"""RRT-Connect (OMPL) baseline on the SE(3) shape-planning test sets.

Plans the same start/goal pairs that ``evaluate_training_3d.py`` feeds to the
learned planner, but with OMPL's ``RRTConnect`` in SE(3), so the two are
directly comparable.  Reports the success rate and the average wall-clock
planning time per test case.

Everything lives in the *normalized* frame written by
``dataprocessing/preprocess_obj.py``:

    world_normalized = (world_obj - meta.env_center) / meta.env_scale

Start/goal poses come from ``<dataPath>/sampled_points.npy`` -- an (N, 12)
array whose first 6 columns are the start config and last 6 the goal config,
each ``(x, y, z, rx, ry, rz)`` with the rotation vector stored divided by 2*pi.

Collision model (same one that labels the training data,
``preprocess_obj.points_inside_tets``): the environment mesh -- obstacles *and*
walls -- is sampled into a dense surface point cloud, and a pose is in
collision iff any env point falls inside the placed shape's tetrahedral
decomposition.  A KD-tree over the cloud plus the shape's bounding radius gives
the broad phase.

Needs the OMPL python bindings (``pip install ompl``; the source tree under
``baseline_ompl/ompl-1.7.0`` builds the C++ library only).

Run from the nested ntrl-demo root, e.g.

    python ../baseline_ompl/rrt_connect_eval.py \
        --obj datasets/3dshape/Lshape3d.obj \
        --env datasets/3dshape/env1.obj \
        --dataPath testing_data/3dshape/Lshape3d_env1
"""

import os
import sys
import json
import time
import argparse

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# ``dataprocessing`` lives in the sibling ntrl-demo package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_NTRL_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, 'ntrl-demo'))
for _p in (os.getcwd(), _NTRL_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from dataprocessing.preprocess_obj import (          # noqa: E402
    DEFAULT_TET_SWITCHES, load_obj, sample_surface_points, tetrahedralize_shape)

from ompl import base as ob                          # noqa: E402
from ompl import geometric as og                     # noqa: E402
from ompl import util as ou                          # noqa: E402


DIM = 6
TWO_PI = 2.0 * np.pi
# Surface points sampled over the FULL environment mesh for collision checking
# (same count evaluate_training_3d.py uses).
ENV_COLLISION_POINTS = 50000


def resolve(path):
    """Resolve a path against the cwd, falling back to the ntrl-demo root."""
    if os.path.exists(path) or os.path.isabs(path):
        return path
    alt = os.path.join(_NTRL_ROOT, path)
    return alt if os.path.exists(alt) else path


# ──────────────────────────────────────────────────────────────────────────────
# Collision checking
# ──────────────────────────────────────────────────────────────────────────────
class ShapeCollisionChecker:
    """Point-in-tet collision test for a rigid shape moving in a point-cloud env.

    ``tets_local`` is (K, 4, 3) in the shape's local frame.  A world point is
    mapped into that frame with ``local = (world - t) @ R`` (the inverse of the
    placement ``world = local @ R.T + t``) and tested against every tet via
    barycentric coordinates -- exactly ``preprocess_obj.points_inside_tets``.
    """

    def __init__(self, tets_local, env_points, radius, tol=-1e-6):
        self.env_points = np.ascontiguousarray(env_points, dtype=np.float64)
        self.tree = cKDTree(self.env_points)
        self.radius = float(radius)
        self.tol = tol

        tets = np.asarray(tets_local, dtype=np.float64)
        self.v0 = tets[:, 0, :]                                  # (K,3)
        M = np.stack([tets[:, 1, :] - self.v0,
                      tets[:, 2, :] - self.v0,
                      tets[:, 3, :] - self.v0], axis=-1)         # (K,3,3) columns
        self.Minv = np.linalg.inv(M)                             # (K,3,3)
        self.n_checks = 0

    def in_collision(self, t, R):
        """True iff any env point lies inside the shape placed at (t, R)."""
        self.n_checks += 1
        idx = self.tree.query_ball_point(t, self.radius)
        if not idx:
            return False
        near_local = (self.env_points[idx] - t) @ R              # (M,3)
        rhs = near_local[None, :, :] - self.v0[:, None, :]       # (K,M,3)
        bary = np.einsum('kij,kmj->kmi', self.Minv, rhs)         # (K,M,3)
        l0 = 1.0 - bary.sum(axis=-1)
        inside = ((bary >= self.tol).all(axis=-1)) & (l0 >= self.tol)
        return bool(inside.any())


def state_to_pose(state):
    """OMPL SE(3) state -> (translation (3,), rotation matrix (3,3))."""
    t = np.array([state.getX(), state.getY(), state.getZ()], dtype=np.float64)
    q = state.rotation()
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return t, R


def cfg_to_state(space, cfg):
    """Dataset config (x,y,z, rotvec/2pi) -> a newly allocated SE(3) state.

    The state is owned by Python (the binding frees it on garbage collection);
    calling ``space.freeState`` on it double-frees and crashes the interpreter.
    """
    st = space.allocState()
    st.setXYZ(float(cfg[0]), float(cfg[1]), float(cfg[2]))
    rotvec = np.asarray(cfg[3:6], dtype=np.float64) * TWO_PI
    angle = float(np.linalg.norm(rotvec))
    rot = st.rotation()
    if angle < 1e-12:
        rot.setIdentity()
    else:
        axis = rotvec / angle
        rot.setAxisAngle(float(axis[0]), float(axis[1]), float(axis[2]), angle)
    return st


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
def build_scene(args):
    """Load meshes + test pairs and return (checker, bounds, start/goal pairs)."""
    data_path = resolve(args.dataPath)
    with open(os.path.join(data_path, 'meta.json')) as f:
        meta = json.load(f)
    env_scale = float(meta['env_scale'])
    env_center = np.asarray(meta['env_center'], dtype=np.float64)
    shape_scale = float(meta.get('shape_scale', 1.0))

    # Shape: centre on its bbox and normalize, as preprocess_obj / evaluate do.
    V_sh, F_sh, _ = load_obj(resolve(args.obj))
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                                   dtype=np.float64)
    shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
    shape_radius = float(np.linalg.norm(shape_V, axis=1).max())

    # 'Q' = quiet, so tetgen's page of statistics stays out of the log.
    TV, TT, _ = tetrahedralize_shape(shape_V, shape_F,
                                     switches=DEFAULT_TET_SWITCHES + 'Q')
    tets_local = np.asarray(TV, dtype=np.float64)[TT]            # (K,4,3)

    # Environment: obstacles *and* walls all count as obstacles.
    V_env, F_env, _ = load_obj(resolve(args.env))
    V_env_n = (V_env - env_center) / env_scale
    env_pts = np.ascontiguousarray(
        sample_surface_points(V_env_n, F_env, ENV_COLLISION_POINTS),
        dtype=np.float64)

    checker = ShapeCollisionChecker(tets_local, env_pts, shape_radius)

    arr = np.load(os.path.join(data_path, 'sampled_points.npy'))  # (N, 12)
    n = min(args.n, len(arr)) if args.n > 0 else len(arr)
    pairs = [(arr[i, 0:DIM].astype(np.float64), arr[i, DIM:2 * DIM].astype(np.float64))
             for i in range(n)]

    lo = V_env_n.min(axis=0)
    hi = V_env_n.max(axis=0)

    print(f'shape       : {resolve(args.obj)}  '
          f'({len(shape_V)} verts, {len(TT)} tets, radius {shape_radius:.4f})')
    print(f'environment : {resolve(args.env)}  '
          f'({len(F_env)} tris, {len(env_pts)} collision points)')
    print(f'bounds      : low {np.round(lo, 4).tolist()}  high {np.round(hi, 4).tolist()}')
    print(f'test cases  : {n} from {os.path.join(data_path, "sampled_points.npy")}')
    return checker, (lo, hi), pairs


def make_setup(checker, bounds, args):
    """SimpleSetup on SE(3) with an RRTConnect planner and our validity checker."""
    space = ob.SE3StateSpace()
    lo, hi = bounds
    rb = ob.RealVectorBounds(3)
    for i in range(3):
        rb.setLow(i, float(lo[i]))
        rb.setHigh(i, float(hi[i]))
    space.setBounds(rb)

    ss = og.SimpleSetup(space)

    def is_valid(state):
        t, R = state_to_pose(state)
        return not checker.in_collision(t, R)

    ss.setStateValidityChecker(is_valid)
    ss.getSpaceInformation().setStateValidityCheckingResolution(args.resolution)

    planner = og.RRTConnect(ss.getSpaceInformation())
    if args.range > 0:
        planner.setRange(args.range)
    ss.setPlanner(planner)
    return space, ss


def path_in_collision(path, checker):
    """Interpolate the solution and re-check every waypoint (like the NN eval)."""
    path.interpolate()
    for i in range(path.getStateCount()):
        t, R = state_to_pose(path.getState(i))
        if checker.in_collision(t, R):
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='RRT-Connect (OMPL) baseline on the SE(3) shape test sets.')
    parser.add_argument('--obj', required=True,
                        help='Moving shape OBJ, e.g. datasets/3dshape/rectangle.obj')
    parser.add_argument('--env', required=True,
                        help='Environment OBJ, e.g. datasets/3dshape/env1.obj')
    parser.add_argument('--dataPath', required=True,
                        help='Test-data dir holding sampled_points.npy + meta.json')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of start/goal pairs to plan (0 = all)')
    parser.add_argument('--time', type=float, default=5.0,
                        help='Planning time limit per test case, seconds')
    parser.add_argument('--range', type=float, default=0.0,
                        help='RRTConnect extension range (0 = OMPL default)')
    parser.add_argument('--resolution', type=float, default=0.005,
                        help='Motion-validation resolution, fraction of space extent')
    parser.add_argument('--simplify', action='store_true',
                        help='Run path simplification (not counted in solve time)')
    parser.add_argument('--seed', type=int, default=1,
                        help='RNG seed for OMPL and for the env surface sampling')
    parser.add_argument('--out', default='',
                        help='Optional directory for the summary txt/csv')
    args = parser.parse_args()

    ou.setLogLevel(ou.LogLevel.LOG_WARN)
    # Seed both RNGs: OMPL's sampler and numpy's, which draws the environment
    # surface point cloud (a different cloud flips the odd grazing contact).
    ou.RNG.setSeed(args.seed)
    np.random.seed(args.seed)

    checker, bounds, pairs = build_scene(args)
    space, ss = make_setup(checker, bounds, args)

    print(f'planner     : RRTConnect  (time limit {args.time}s per case, '
          f'resolution {args.resolution})')
    print()

    rows = []
    n_success = 0
    n_invalid_endpoint = 0
    n_timeout = 0
    n_collision = 0
    for i, (start_cfg, goal_cfg) in enumerate(pairs):
        start = cfg_to_state(space, start_cfg)
        goal = cfg_to_state(space, goal_cfg)

        # A pair whose endpoints are already in collision is unplannable; count
        # it as a failure but flag it separately.
        endpoints_ok = all(
            not checker.in_collision(*state_to_pose(s)) for s in (start, goal))

        ss.clear()
        ss.setStartAndGoalStates(start, goal)

        t0 = time.perf_counter()
        solved = ss.solve(args.time)
        elapsed = time.perf_counter() - t0

        exact = bool(ss.haveExactSolutionPath())
        collision = False
        length = float('nan')
        if exact:
            if args.simplify:
                ss.simplifySolution()
            path = ss.getSolutionPath()
            length = path.length()
            collision = path_in_collision(path, checker)

        ok = endpoints_ok and exact and not collision
        if ok:
            n_success += 1
        else:
            if not endpoints_ok:
                n_invalid_endpoint += 1
            elif not exact:
                n_timeout += 1
            elif collision:
                n_collision += 1

        rows.append((i, ok, elapsed, length, endpoints_ok, exact, collision))
        print(f'[{i:03d}] {"PASS" if ok else "FAIL"}  '
              f'time={elapsed:7.3f}s  '
              f'status={solved.asString():<18} '
              f'len={length:7.3f}  '
              f'endpoints_valid={endpoints_ok}  collision={collision}',
              flush=True)

    # ── summary ──────────────────────────────────────────────────────────────
    n_total = len(rows)
    times = np.array([r[2] for r in rows], dtype=np.float64)
    succ_times = np.array([r[2] for r in rows if r[1]], dtype=np.float64)
    success_rate = n_success / n_total if n_total else 0.0
    # Secondary rate over the pairs the planner could actually attempt.  A few
    # dataset poses graze an obstacle just enough for this collision model to
    # call the start or goal invalid; those are not planner failures.
    n_plannable = n_total - n_invalid_endpoint
    plannable_rate = n_success / n_plannable if n_plannable else 0.0

    lines = [
        f'shape                     : {resolve(args.obj)}',
        f'environment               : {resolve(args.env)}',
        f'data_path                 : {resolve(args.dataPath)}',
        f'planner                   : RRTConnect (OMPL, SE(3))',
        f'time_limit_per_case       : {args.time:.3f} s',
        f'test_cases                : {n_total}',
        f'successes                 : {n_success}',
        f'failures                  : {n_total - n_success}',
        f'  no_solution_in_time     : {n_timeout}',
        f'  path_in_collision       : {n_collision}',
        f'  invalid_start_or_goal   : {n_invalid_endpoint}',
        f'success_rate              : {success_rate:.4f}  ({success_rate:.1%})',
        f'success_rate|valid_endpts : {plannable_rate:.4f}  ({plannable_rate:.1%})  '
        f'[{n_success}/{n_plannable}]',
        f'avg_time_per_case         : {times.mean():.4f} s',
        f'avg_time_successful_cases : '
        f'{(succ_times.mean() if succ_times.size else float("nan")):.4f} s',
        f'median_time_per_case      : {np.median(times):.4f} s',
        f'total_time                : {times.sum():.2f} s',
        f'collision_checks          : {checker.n_checks}',
    ]
    print()
    print('\n'.join(lines))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        summary = os.path.join(args.out, 'rrt_connect_success_rate.txt')
        with open(summary, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        csv = os.path.join(args.out, 'rrt_connect_cases.csv')
        with open(csv, 'w') as f:
            f.write('idx,success,time_s,path_length,endpoints_valid,exact,collision\n')
            for i, ok, el, ln, ep, ex, col in rows:
                f.write(f'{i},{int(ok)},{el:.6f},{ln:.6f},{int(ep)},{int(ex)},{int(col)}\n')
        print(f'\nWrote {summary}\nWrote {csv}')


if __name__ == '__main__':
    main()
