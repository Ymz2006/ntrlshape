#!/usr/bin/env python3
"""Interactive viser viewer for the NTFields 3-D shape (SE(3)) baseline.

Same viewer ``ntrl-demo/evaluate_training_3d_batched.py`` launches -- the
environment drawn as its own triangle mesh, the moving shape stamped at every
waypoint and colored by progress along the path (viridis, dark = start ..
bright = goal), the start pose in red and the goal pose in green -- but the
paths come from the NTFields baseline planner of ``3d_plan.py`` instead of the
ntrl-demo model, and the GUI carries nothing but the case list.

The rollout, the collision test and the pass/fail rule are ``3d_plan.py``'s own
(imported from it, not re-implemented), so a case labelled ``success`` here is
exactly a case ``3d_plan.py`` counts as a PASS.

Run from this directory:

    python 3d_plan_viser.py --env rectangle_env1 --cases 50 --device cuda:0

then open http://<server-ip>:8090 on your host PC.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from glob import glob
from timeit import default_timer as timer

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
NTRL_ROOT = os.path.abspath(os.path.join(ROOT, os.pardir, os.pardir,
                                         'ntrl-demo', 'ntrl-demo'))
for _p in (ROOT, NTRL_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)


def _load_plan_module():
    """Import ``3d_plan.py`` as a module (its name is not a valid identifier)."""
    path = os.path.join(ROOT, '3d_plan.py')
    spec = importlib.util.spec_from_file_location('ntfields_3d_plan', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


plan = _load_plan_module()

from models.model_3d import Model                                    # noqa: E402
from dataprocessing.preprocess_obj import (                          # noqa: E402
    DEFAULT_TET_SWITCHES, load_obj, sample_surface_points,
    tetrahedralize_shape, _rotvec_to_matrix_np)

DIM = plan.DIM
ENV_COLLISION_POINTS = plan.ENV_COLLISION_POINTS


# ──────────────────────────────────────────────────────────────────────────────
# Scene
# ──────────────────────────────────────────────────────────────────────────────
def build_scene(data_path, mesh_root, seed, collision=True):
    """Load the meshes meta.json names; return the render meshes + the checker.

    ``3d_plan.build_scene`` builds the very same checker but keeps the meshes to
    itself, and the viewer needs them, so the loading is repeated here with the
    identical normalization (shape centred on its bbox and divided by
    ``env_scale``; environment centred on ``env_center`` and divided by the same
    scale) and the identical seeded surface sampling.
    """
    with open(os.path.join(data_path, 'meta.json')) as fh:
        meta = json.load(fh)
    env_scale = float(meta['env_scale'])
    env_center = np.asarray(meta['env_center'], dtype=np.float64)
    shape_scale = float(meta.get('shape_scale', 1.0))

    shape_obj = plan.resolve_mesh(meta['shape_obj'], mesh_root)
    V_sh, F_sh, _ = load_obj(shape_obj)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                                   dtype=np.float64)
    shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
    shape_radius = float(np.linalg.norm(shape_V, axis=1).max())

    env_obj = plan.resolve_mesh(meta['env_obj'], mesh_root)
    V_env, F_env, names_env = load_obj(env_obj)
    V_env_n = np.ascontiguousarray((V_env - env_center) / env_scale, dtype=np.float64)
    # The walls are drawn translucent so the interior stays visible; everything,
    # walls included, still counts as an obstacle for collision.
    wall_mask = np.array([('wall' in str(n).lower()) for n in names_env])
    wall_F = F_env[wall_mask]
    obst_F = F_env[~wall_mask]

    checker = None
    if collision:
        TV, TT, _ = tetrahedralize_shape(shape_V, shape_F,
                                         switches=DEFAULT_TET_SWITCHES + 'Q')
        tets_local = np.asarray(TV, dtype=np.float64)[TT]
        np.random.seed(seed)
        env_pts = np.ascontiguousarray(
            sample_surface_points(V_env_n, F_env, ENV_COLLISION_POINTS),
            dtype=np.float64)
        checker = plan.ShapeCollisionChecker(tets_local, env_pts, shape_radius)

    print('shape       : {}  ({} verts)'.format(shape_obj, len(shape_V)))
    print('environment : {}  ({} tris: {} obstacle, {} wall)'.format(
        env_obj, len(F_env), len(obst_F), len(wall_F)))
    return meta, shape_V, shape_F, V_env_n, obst_F, wall_F, checker


# ──────────────────────────────────────────────────────────────────────────────
# Viewer
# ──────────────────────────────────────────────────────────────────────────────
def _placed_mesh(shape_V, cfg):
    """Transform the shape's local mesh by a config (x,y,z, rotvec in radians)."""
    R = _rotvec_to_matrix_np(cfg[3:6])
    return shape_V @ R.T + cfg[0:3]


def _progress_color(t):
    """Map a progress value t in [0, 1] to an RGB tuple of ints (viridis)."""
    try:
        import matplotlib.cm as cm
        r, g, b, _ = cm.get_cmap('viridis')(float(t))
    except Exception:
        r, g, b = float(t), float(t), 1.0 - float(t)
    return (int(r * 255), int(g * 255), int(b * 255))


def add_environment(server, env_V, obst_F, wall_F):
    """Draw the environment as its actual triangle mesh (static scene)."""
    if len(obst_F) > 0:
        server.scene.add_mesh_simple(
            '/env/obstacles', vertices=env_V, faces=obst_F,
            color=(150, 150, 150), opacity=1.0, flat_shading=True, side='double')
    if len(wall_F) > 0:
        server.scene.add_mesh_simple(
            '/env/walls', vertices=env_V, faces=wall_F,
            color=(173, 216, 230), opacity=0.15, flat_shading=True, side='double')


def render_episode(server, ep, shape_V, shape_F, stride=1):
    """Stamp the shape along one episode's path; return the scene handles.

    ``stride`` > 1 draws every n-th waypoint (the last one is always drawn), for
    paths long enough that one mesh per waypoint bogs the browser down.
    """
    handles = []
    waypoints = ep['waypoints']
    T = len(waypoints)
    idxs = list(range(0, T, max(stride, 1)))
    if T and idxs[-1] != T - 1:
        idxs.append(T - 1)
    for t in idxs:
        Vp = _placed_mesh(shape_V, waypoints[t])
        handles.append(server.scene.add_mesh_simple(
            f'/episode/traj/{t:04d}', vertices=Vp, faces=shape_F,
            color=_progress_color(t / max(T - 1, 1)), opacity=0.5,
            flat_shading=True, side='double'))

    for cfg, col, nm in ((ep['begin_cfg'], (220, 30, 30), 'start'),
                         (ep['end_cfg'], (30, 180, 30), 'goal')):
        if cfg is None:
            continue
        Vp = _placed_mesh(shape_V, cfg)
        handles.append(server.scene.add_mesh_simple(
            f'/episode/{nm}', vertices=Vp, faces=shape_F,
            color=col, opacity=0.9, flat_shading=True, side='double'))
    return handles


def launch_viser(episodes, shape_V, shape_F, env_V, obst_F, wall_F, port, stride):
    """Serve the scene with a GUI that is nothing but the case dropdown."""
    import viser
    server = viser.ViserServer(host='0.0.0.0', port=port)
    server.scene.set_up_direction('+y')

    add_environment(server, env_V, obst_F, wall_F)

    # One entry per case, labelled with the outcome the planner recorded, so the
    # dropdown doubles as the pass/fail list.
    labels = ['{:04d}_{}'.format(ep['idx'], ep['status']) for ep in episodes] \
        or ['(none)']
    dd = server.gui.add_dropdown('Case', options=labels)

    current = []

    def show(i):
        for h in current:
            h.remove()
        current.clear()
        if not episodes:
            return
        current.extend(render_episode(server, episodes[i], shape_V, shape_F,
                                      stride=stride))

    @dd.on_update
    def _(_):
        if not episodes or dd.value == '(none)':
            return
        show(list(dd.options).index(dd.value))

    show(0)

    print('\nServing viser at http://0.0.0.0:{}  '
          '— open this on your host PC'.format(port))
    print("Use 'Case' to browse the planned cases.  Path is colored by progress "
          '(dark=start .. bright=goal); start pose red, goal green.')
    print('Press Ctrl-C to stop.\n', flush=True)
    while True:
        time.sleep(10)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', default='rectangle_env1',
                        help='Environment name; selects both the checkpoint '
                             '(<model-root>/<env>/latest.pt) and the test set '
                             '(<test-root>/<env>).')
    parser.add_argument('--model-root', default=os.path.join(ROOT, 'outputs', '3dshape'))
    parser.add_argument('--checkpoint', default=None,
                        help='Explicit .pt to plan with; overrides --model-root/--env.')
    parser.add_argument('--test-root', default=plan.DEFAULT_TEST_ROOT)
    parser.add_argument('--data', default=None,
                        help='Full test-set directory; overrides --test-root/--env.')
    parser.add_argument('--mesh-root', default=plan.DEFAULT_MESH_ROOT)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--model-size', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--cases', type=int, default=50,
                        help='How many start/goal pairs to plan and offer in the '
                             'dropdown; 0 = all of them (1000 paths is a long '
                             'wait before the viewer comes up).')
    parser.add_argument('--batch', type=int, default=50,
                        help='Episodes whose rollouts run together on the GPU. '
                             'Timing is irrelevant here, so batch freely.')
    parser.add_argument('--steps', type=int, default=200,
                        help='Cap on MPPI iterations per episode.')
    parser.add_argument('--samples', type=int, default=50, help='MPPI samples per step.')
    parser.add_argument('--horizon', type=int, default=5, help='MPPI rollout horizon.')
    parser.add_argument('--step', type=float, default=0.015,
                        help='Per-sample displacement cap in the normalized 6-D '
                             'config space (the convergence ball is 0.01).')
    parser.add_argument('--momentum', type=float, default=2.0,
                        help='Gain on the previous accepted step; 0 disables it.')
    parser.add_argument('--goal-tol', type=float, default=0.01,
                        help='Convergence ball radius in the normalized 6-D space.')
    parser.add_argument('--2d', dest='two_d', default=None, action='store_true',
                        help='Force planar rollouts (x, y, rz only). Read from '
                             'meta.json ("two_d") when not given.')
    parser.add_argument('--no-collision', dest='collision', action='store_false',
                        help='Skip collision checking; the status then reflects '
                             'convergence alone.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Draw every n-th waypoint of a path (1 = all of them).')
    parser.add_argument('--port', type=int, default=8090, help='viser HTTP port.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    data_path = args.data if args.data is not None else os.path.join(args.test_root, args.env)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = plan.resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = plan.find_checkpoint(args)
    meta, shape_V, shape_F, env_V, obst_F, wall_F, checker = build_scene(
        data_path, args.mesh_root, args.seed, collision=args.collision)
    planar = meta.get('two_d', False) if args.two_d is None else args.two_d

    arr = np.load(os.path.join(data_path, 'sampled_points.npy')).astype(np.float32)
    if arr.shape[1] != 2 * DIM:
        raise ValueError('expected an (N, {}) sampled_points.npy, got {}'.format(
            2 * DIM, arr.shape))
    n_cases = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
    arr = arr[:n_cases]

    # Model.__init__ writes nothing until training, so the "model path" only has
    # to exist; point it at the checkpoint's own directory.
    model = Model(os.path.dirname(checkpoint), data_path, DIM, [0.0] * DIM,
                  device=device, model_size=args.model_size)
    model.load(checkpoint)
    model.network.eval()

    print('checkpoint  : {}'.format(checkpoint))
    print('test set    : {}  ({} cases)'.format(data_path, n_cases))
    print('device      : {}   planar: {}   batch: {}'.format(device, planar, args.batch))
    print('collision   : {}'.format('on' if args.collision else 'OFF (convergence only)'))

    episodes = []
    n_succ = 0
    t0 = timer()
    for lo in range(0, n_cases, args.batch):
        hi = min(lo + args.batch, n_cases)
        XP = torch.from_numpy(arr[lo:hi]).to(device)
        with torch.no_grad():
            paths, _, converged, _ = plan.mppi(
                model, XP.clone(), DIM, steps=args.steps, sample_num=args.samples,
                horizon=args.horizon, step=args.step, momentum=args.momentum,
                goal_tol=args.goal_tol, planar=planar)

        for k, idx in enumerate(range(lo, hi)):
            wp = paths[k]
            collision = bool(checker.path_in_collision(wp)) if args.collision else False
            ok = bool(converged[k]) and not collision
            n_succ += int(ok)
            # The rollout works in the normalized frame (rotvec / 2*pi); the
            # viewer wants the rotvec back in radians, so rescale that block.
            deg = wp.copy().astype(np.float64)
            deg[:, 3:6] *= 2 * np.pi
            begin = arr[idx, 0:DIM].astype(np.float64).copy()
            end = arr[idx, DIM:2 * DIM].astype(np.float64).copy()
            begin[3:6] *= 2 * np.pi
            end[3:6] *= 2 * np.pi
            episodes.append({
                'idx': idx,
                'status': 'success' if ok else 'fail',
                'waypoints': deg,
                'begin_cfg': begin,
                'end_cfg': end,
            })
            print('[{:04d}] {}  waypoints={:4d}  converged={}  collision={}'.format(
                idx, 'PASS' if ok else 'FAIL', len(wp), bool(converged[k]), collision),
                flush=True)

    rate = n_succ / len(episodes) if episodes else 0.0
    print('\nplanned {} cases in {:.1f}s   successes {}  ({:.1%})'.format(
        len(episodes), timer() - t0, n_succ, rate))

    launch_viser(episodes, shape_V, shape_F, env_V, obst_F, wall_F,
                 args.port, args.stride)


if __name__ == '__main__':
    main()
