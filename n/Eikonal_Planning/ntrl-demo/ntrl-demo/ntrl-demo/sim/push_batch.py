"""Batch the headless push demo over many test cases and score each one.

``push_t_demo.py`` runs one (shape, env, case).  This driver runs ``--cases`` of them per
cell in ONE process, so the checkpoint is loaded and the primitive library measured once
per cell rather than once per case, and it scores every run on:

- **compute time**: wall-clock from the first plan to the end of the run (planning,
  replanning, action selection and physics -- everything but the one-off calibration and
  checkpoint load, which are shared across cases and reported separately);
- **deviation**: distance (world units) from the T's executed trace to the ORIGINAL
  planned reference (the path before any replan), one value per physics frame, reduced to
  a per-case mean and max;
- **collision**: whether the T's footprint ever intersected a wall or a block.  The
  environment is scenery in the sim (see ``--solid-env``), so this is a geometric check
  on every frame rather than a physics event;
- **goal**: whether the controller declared the goal reached (``--goal-dist`` /
  ``--goal-deg``) before ``--max-seconds`` of simulated time.

Success is "reached the goal without ever colliding".  Every run's record is written to
``<out>/<shape>_<env>.json``; ``_make_charts_pushtask.py`` turns those into the table.

Usage (from the ntrl-demo root, inside the container):
    python sim/push_batch.py --envs 2denv1 2denv3 2denv4 --shapes rectangle Lshape3d Ashape3d \\
        --cases 100 --out results/push_task -- --teleport --spacing 1 --push-len 5 --plan-device cuda:1

Everything after ``--`` is handed to ``push_t_demo.build_args`` unchanged.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

import pymunk
import pymunk_viser_push as base
import push_t_demo as demo


def polyline_distance(pts, ref):
    """Distance from each of (N,2) points to the (M,2) polyline -> (N,)."""
    a, b = ref[:-1], ref[1:]                                   # (M-1, 2)
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-12)
    ap = pts[:, None, :] - a[None]                             # (N, M-1, 2)
    t = np.clip((ap * ab[None]).sum(2) / ab2[None], 0.0, 1.0)
    d = ap - t[..., None] * ab[None]
    return np.sqrt((d ** 2).sum(2)).min(1)


class Cell:
    """Everything that is fixed for one (shape, env): meshes, planner, primitives."""

    def __init__(self, shape, env, demo_argv):
        sys.argv = ['push_t_demo.py', '--shape', shape, '--env', env, '--headless'] + demo_argv
        args = demo.build_args()
        args.teleport = True
        demo.resolve_paths(args)
        self.args, self.shape, self.env = args, shape, env

        env_mesh = base.load_mesh(args.env)
        tee_mesh = base.load_mesh(args.shape)
        self.env_polys = sorted(base.footprint(env_mesh), key=lambda p: -p.area)
        tee_polys = base.footprint(tee_mesh)
        assert len(tee_polys) == 1
        c = tee_polys[0].centroid
        self.tee_poly = Polygon([(x - c.x, y - c.y) for x, y in tee_polys[0].exterior.coords])
        Vz = np.array([[float(t) for t in ln.split()[1:4]]
                       for ln in open(args.shape_zup) if ln.startswith('v ')])
        bbox_c = 0.5 * (Vz.min(0) + Vz.max(0))
        self.b2c = np.array([c.x - bbox_c[0], c.y - bbox_c[1]])
        self.obstacles = prep(unary_union(self.env_polys))
        self.bounds = self.env_polys[0].bounds

        with open(os.path.join(args.dataPath, 'meta.json')) as f:
            self.meta = json.load(f)
        t0 = time.time()
        self.womodel, _, _ = demo.load_planner(args.dataPath, args.modelPath, args.ckpt, 0,
                                               args.plan_device)
        self.pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy'))
        self.load_s = time.time() - t0

        c_len = args.c_length if args.c_length else args.spin_friction / args.friction
        if args.transit_approach is None:
            args.transit_approach = args.pusher_radius
        if args.transit_reach is None:
            args.transit_reach = 2.0 * args.pusher_radius
        self.ik = demo.PushIK(self.tee_poly, c_len, n_boundary=args.n_boundary)
        self.push_len = args.push_len if args.push_len else args.spacing * args.action_steps
        push_lens = demo.push_length_ladder(self.push_len, args.push_len_min,
                                            args.push_len_steps)
        self.prims = demo.PushPrimitives(self.ik, n_points=args.n_contacts,
                                         n_dirs=args.n_dirs, spread_deg=args.dir_spread)
        t0 = time.time()
        self.prims.calibrate(args, self.tee_poly, push_lens,
                             cache_dir=os.path.dirname(args.dataPath) or '.')
        self.calib_s = time.time() - t0

    def tee_at(self, x, y, th):
        return translate(rotate(self.tee_poly, th, origin=(0, 0), use_radians=True), x, y)

    def run_case(self, case):
        a = self.args
        start_norm, goal_norm = self.pairs[case][:6].copy(), self.pairs[case][6:].copy()
        t0 = time.time()
        path_norm, dist = demo.plan_from(self.womodel, start_norm, goal_norm,
                                         a.plan_device, a.mppi_steps)
        ref = demo.planner_to_world(path_norm, self.meta['env_scale'],
                                    self.meta['env_center'], self.b2c)
        ref = demo.resample_path(ref, a.spacing, a.smooth)
        ref0 = ref.copy()
        plan_s = time.time() - t0

        tee_start = (float(ref[0, 0]), float(ref[0, 1]))
        heading = ref[min(a.lookahead, len(ref) - 1), :2] - ref[0, :2]
        if np.linalg.norm(heading) < 1e-9:
            heading = np.array([1.0, 0.0])
        heading = heading / np.linalg.norm(heading)
        pusher_start = np.asarray(tee_start) - heading * (
            self.ik.radius + a.pusher_radius + a.standoff)
        sim = demo.PushSim(a, self.env_polys, self.tee_poly, tee_start, float(ref[0, 2]),
                           tuple(pusher_start))
        ctrl = demo.PushController(self.ik, ref, a, list(self.env_polys))
        rep = None
        if not a.no_replan:
            rep = demo.Replanner(self.womodel, goal_norm, a.plan_device, a.replan_steps,
                                 self.meta['env_scale'], self.meta['env_center'], self.b2c,
                                 a.spacing, a.smooth)
        runner = demo.PrimitiveRunner(ctrl, self.prims, self.push_len, a, rep)

        dt = 1.0 / a.fps
        limit = a.max_seconds or 120.0
        t, trace, blew_up = 0.0, [], False
        n_coll = 0
        first_coll = None
        while t < limit and not ctrl.done:
            if demo.escaped(sim, self.bounds):
                blew_up = True
                break
            runner.update(sim)
            for _ in range(a.substeps):
                sim.step(dt / a.substeps)
            x, y, th = sim.tee.position.x, sim.tee.position.y, sim.tee.angle
            trace.append([x, y, th])
            if self.obstacles.intersects(self.tee_at(x, y, th)):
                n_coll += 1
                if first_coll is None:
                    first_coll = t
            t += dt
        total_s = time.time() - t0

        trace = np.array(trace) if trace else ref0[:1]
        dev = polyline_distance(trace[:, :2], ref0[:, :2])
        dp, dth = demo.pose_error(sim, ref)
        collided = n_coll > 0
        return {
            'case': int(case),
            'plan_converged': bool(dist < 0.01),
            'ref_len': float(np.linalg.norm(np.diff(ref0[:, :2], axis=0), axis=1).sum()),
            'plan_s': plan_s,
            'compute_s': total_s,
            'sim_s': t,
            'frames': int(len(trace)),
            'n_actions': int(runner.n_actions),
            'n_replans': int(rep.n_done) if rep else 0,
            'goal': bool(ctrl.done),
            'blew_up': blew_up,
            'collided': collided,
            'collision_frames': int(n_coll),
            'first_collision_s': first_coll,
            'success': bool(ctrl.done and not collided and not blew_up),
            'dev_mean': float(dev.mean()),
            'dev_max': float(dev.max()),
            'dev_final': float(dev[-1]),
            'final_pos_err': float(dp),
            'final_ang_err_deg': float(math.degrees(dth)),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--envs', nargs='+', default=['2denv1', '2denv3', '2denv4'])
    ap.add_argument('--shapes', nargs='+', default=['rectangle', 'Lshape3d', 'Ashape3d'])
    ap.add_argument('--cases', type=int, default=100)
    ap.add_argument('--start', type=int, default=0, help='first test-set index')
    ap.add_argument('--out', default='results/push_task')
    ap.add_argument('--redo', action='store_true', help='rerun cells that already have a json')
    ap.add_argument('demo_args', nargs=argparse.REMAINDER,
                    help='everything after -- goes to push_t_demo')
    opts = ap.parse_args()
    demo_argv = [x for x in opts.demo_args if x != '--']
    os.makedirs(opts.out, exist_ok=True)

    for env in opts.envs:
        for shape in opts.shapes:
            out = os.path.join(opts.out, f'{shape}_{env}.json')
            if os.path.exists(out) and not opts.redo:
                print(f'[batch] {shape}_{env}: exists, skipping ({out})')
                continue
            print(f'\n[batch] ===== {shape} in {env} =====', flush=True)
            cell = Cell(shape, env, demo_argv)
            print(f'[batch] checkpoint load {cell.load_s:.1f}s, primitive calibration '
                  f'{cell.calib_s:.1f}s (cached after the first shape)', flush=True)
            records = []
            t_cell = time.time()
            for case in range(opts.start, opts.start + opts.cases):
                r = cell.run_case(case)
                records.append(r)
                print(f'[batch] {shape}_{env} case {case:3d}: '
                      f'{"GOAL" if r["goal"] else "miss"} '
                      f'{"COLL" if r["collided"] else "free"} '
                      f'{"BLEWUP " if r["blew_up"] else ""}'
                      f'dev {r["dev_mean"]:5.1f} (max {r["dev_max"]:5.1f}) '
                      f'{r["n_actions"]:3d} actions  {r["compute_s"]:5.1f}s',
                      flush=True)
            summary = {
                'shape': shape, 'env': env, 'cases': len(records),
                'ckpt': cell.args.ckpt, 'dataPath': cell.args.dataPath,
                'demo_args': demo_argv,
                'load_s': cell.load_s, 'calib_s': cell.calib_s,
                'wall_s': time.time() - t_cell,
                'records': records,
            }
            with open(out, 'w') as f:
                json.dump(summary, f, indent=1)
            n = len(records)
            print(f'[batch] {shape}_{env}: success {100.0 * sum(r["success"] for r in records) / n:.0f}%'
                  f'  collision {100.0 * sum(r["collided"] for r in records) / n:.0f}%'
                  f'  goal {100.0 * sum(r["goal"] for r in records) / n:.0f}%'
                  f'  compute {sum(r["compute_s"] for r in records):.0f}s -> {out}', flush=True)


if __name__ == '__main__':
    main()
