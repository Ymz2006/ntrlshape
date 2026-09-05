"""LazyPRM (OMPL) baseline on the SE(3) shape-planning test sets.

The RRT-Connect twin of this script is ``rrt_connect_eval.py``; everything
except the planner is reused from it -- the same normalized frame, the same
start/goal pairs out of ``sampled_points.npy``, the same point-in-tet collision
model, and the same scoring rules -- so the two tables are directly comparable.

LazyPRM builds its roadmap without validating edges, and only collision-checks
the edges along a candidate solution.  Like the RRT-Connect run, each test case
is planned from scratch (``SimpleSetup.clear()`` drops the roadmap between
cases), so this measures LazyPRM as a single-query planner.  Pass ``--reuse``
to keep the roadmap across cases instead and measure it the multi-query way.

Needs the OMPL 1.7.0 python bindings -- the 2.0.1 wheel does not expose the
lazy planners.

Run from the nested ntrl-demo root, e.g.

    python ../baseline_ompl/lazy_prm_eval.py \
        --obj datasets/3dshape/Lshape3d.obj \
        --env datasets/3dshape/env1.obj \
        --dataPath testing_data/3dshape/Lshape3d_env1
"""

import os
import sys
import time
import argparse

import numpy as np

from rrt_connect_eval import (                       # noqa: E402
    TWO_PI, build_scene, mean, median, path_in_collision, resolve,
    state_to_pose, std)

from ompl import base as ob                          # noqa: E402
from ompl import geometric as og                     # noqa: E402
from ompl import util as ou                          # noqa: E402


def cfg_to_state(space, cfg):
    """Dataset config (x,y,z, rotvec/2pi) -> a ScopedState.

    ``rrt_connect_eval.cfg_to_state`` hands back a raw ``State*``, which the
    2.0.1 bindings accept; the 1.7.0 ones only take a ``ScopedState`` here, so
    build one and reach through it with ``st()`` to set the pose.
    """
    st = ob.State(space)
    inner = st()
    inner.setXYZ(float(cfg[0]), float(cfg[1]), float(cfg[2]))
    rotvec = np.asarray(cfg[3:6], dtype=np.float64) * TWO_PI
    angle = float(np.linalg.norm(rotvec))
    rot = inner.rotation()
    if angle < 1e-12:
        rot.setIdentity()
    else:
        axis = rotvec / angle
        rot.setAxisAngle(float(axis[0]), float(axis[1]), float(axis[2]), angle)
    return st


def make_setup(checker, bounds, args):
    """SimpleSetup on SE(3) with a LazyPRM planner and our validity checker."""
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

    # The 1.7.0 Boost.Python bindings want the checker wrapped in the typed
    # functor; a bare python callable does not match the C++ signature.
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
    ss.getSpaceInformation().setStateValidityCheckingResolution(args.resolution)

    planner = og.LazyPRM(ss.getSpaceInformation())
    if args.max_nn > 0:
        planner.setMaxNearestNeighbors(args.max_nn)
    ss.setPlanner(planner)
    return space, ss


def main():
    parser = argparse.ArgumentParser(
        description='LazyPRM (OMPL) baseline on the SE(3) shape test sets.')
    parser.add_argument('--obj', required=True,
                        help='Moving shape OBJ, e.g. datasets/3dshape/rectangle.obj')
    parser.add_argument('--env', required=True,
                        help='Environment OBJ, e.g. datasets/3dshape/env1.obj')
    parser.add_argument('--dataPath', required=True,
                        help='Test-data dir holding sampled_points.npy + meta.json')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of start/goal pairs to plan (0 = all)')
    parser.add_argument('--time', type=float, default=30.0,
                        help='Planning time limit per test case, seconds; a case '
                             'that takes longer than this counts as a failure')
    parser.add_argument('--max_nn', type=int, default=0,
                        help='LazyPRM max nearest neighbors (0 = OMPL default)')
    parser.add_argument('--resolution', type=float, default=0.005,
                        help='Motion-validation resolution, fraction of space extent')
    parser.add_argument('--reuse', action='store_true',
                        help='Keep the roadmap across test cases (multi-query); '
                             'the default clears it, matching the RRT-Connect run')
    parser.add_argument('--simplify', action='store_true',
                        help='Run path simplification (not counted in solve time)')
    parser.add_argument('--seed', type=int, default=1,
                        help='RNG seed for OMPL and for the env surface sampling')
    parser.add_argument('--out', default='',
                        help='Optional directory for the summary txt/csv')
    args = parser.parse_args()

    ou.setLogLevel(ou.LogLevel.LOG_WARN)
    ou.RNG.setSeed(args.seed)
    np.random.seed(args.seed)

    checker, bounds, pairs = build_scene(args)
    space, ss = make_setup(checker, bounds, args)

    print(f'planner     : LazyPRM  (time limit {args.time}s per case, '
          f'resolution {args.resolution}, '
          f'roadmap {"reused across cases" if args.reuse else "cleared per case"})')
    print()

    rows = []
    n_success = 0
    n_invalid_endpoint = 0
    n_timeout = 0
    n_over_limit = 0
    n_collision = 0
    for i, (start_cfg, goal_cfg) in enumerate(pairs):
        start = cfg_to_state(space, start_cfg)
        goal = cfg_to_state(space, goal_cfg)

        endpoints_ok = all(
            not checker.in_collision(*state_to_pose(s())) for s in (start, goal))

        # ``clearQuery`` drops the start/goal but keeps the roadmap; ``clear``
        # throws the roadmap away too, which is what makes each case a fresh
        # single-query solve like the RRT-Connect baseline.
        if args.reuse:
            ss.getPlanner().clearQuery()
        else:
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

        over_limit = elapsed > args.time

        ok = endpoints_ok and exact and not collision and not over_limit
        if not endpoints_ok:
            n_invalid_endpoint += 1
        elif ok:
            n_success += 1
        elif not exact:
            n_timeout += 1
        elif over_limit:
            n_over_limit += 1
        else:
            n_collision += 1

        rows.append((i, ok, elapsed, length, endpoints_ok, exact, collision,
                     over_limit))
        print(f'[{i:03d}] {"PASS" if ok else "FAIL"}  '
              f'time={elapsed:7.3f}s  '
              f'status={solved.asString():<18} '
              f'len={length:7.3f}  '
              f'endpoints_valid={endpoints_ok}  collision={collision}  '
              f'over_limit={over_limit}',
              flush=True)

    # ── summary ──────────────────────────────────────────────────────────────
    n_total = len(rows)
    valid = [r for r in rows if r[4]]
    n_valid = len(valid)
    times = np.array([r[2] for r in valid], dtype=np.float64)
    succ_times = np.array([r[2] for r in valid if r[1]], dtype=np.float64)
    lengths = np.array([r[3] for r in valid if r[1]], dtype=np.float64)
    success_rate = n_success / n_valid if n_valid else 0.0

    lines = [
        f'shape                     : {resolve(args.obj)}',
        f'environment               : {resolve(args.env)}',
        f'data_path                 : {resolve(args.dataPath)}',
        f'planner                   : LazyPRM (OMPL, SE(3))',
        f'roadmap                   : {"reused across cases" if args.reuse else "cleared per case"}',
        f'time_limit_per_case       : {args.time:.3f} s  (a case over this = fail)',
        f'path_simplification       : {"on" if args.simplify else "off"}',
        '',
        f'test_cases_total          : {n_total}',
        f'invalid_start_or_goal     : {n_invalid_endpoint}  (excluded, unplannable)',
        f'test_cases_scored         : {n_valid}  [valid start+goal only]',
        f'successes                 : {n_success}',
        f'failures                  : {n_valid - n_success}',
        f'  no_solution_in_time     : {n_timeout}',
        f'  over_time_limit         : {n_over_limit}',
        f'  path_in_collision       : {n_collision}',
        f'success_rate              : {success_rate:.4f}  ({success_rate:.1%})  '
        f'[{n_success}/{n_valid}]',
        '',
        f'time_mean                 : {mean(times):.4f} s   [all {n_valid} scored cases]',
        f'time_std                  : {std(times):.4f} s',
        f'time_mean_successful      : {mean(succ_times):.4f} s   '
        f'[{succ_times.size} successful cases]',
        f'time_std_successful       : {std(succ_times):.4f} s',
        f'time_median               : {median(times):.4f} s',
        '',
        f'path_length_mean          : {mean(lengths):.4f}   '
        f'[{lengths.size} successful cases]',
        f'path_length_std           : {std(lengths):.4f}',
        f'path_length_total         : {lengths.sum():.4f}',
        '',
        f'total_time                : {times.sum():.2f} s',
        f'collision_checks          : {checker.n_checks}',
    ]
    print()
    print('\n'.join(lines))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        summary = os.path.join(args.out, 'lazy_prm_success_rate.txt')
        with open(summary, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        csv = os.path.join(args.out, 'lazy_prm_cases.csv')
        with open(csv, 'w') as f:
            f.write('idx,success,time_s,path_length,endpoints_valid,exact,'
                    'collision,over_limit\n')
            for i, ok, el, ln, ep, ex, col, ovr in rows:
                f.write(f'{i},{int(ok)},{el:.6f},{ln:.6f},{int(ep)},{int(ex)},'
                        f'{int(col)},{int(ovr)}\n')
        print(f'\nWrote {summary}\nWrote {csv}')

    # The 1.7.0 bindings corrupt the heap on interpreter teardown; everything is
    # written and flushed by now, so leave without running destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
