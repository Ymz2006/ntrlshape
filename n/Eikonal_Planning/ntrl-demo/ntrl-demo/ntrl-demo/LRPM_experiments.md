# LazyPRM (OMPL) Baseline Experiments

LazyPRM in SE(3) run on the same 3-D test sets the learned planner is evaluated
on (`testing_data/3dshape/<shape>_<env>`), so the numbers line up with
`experiments.md` and `RRT_experiments.md` case for case.  Produced by
`baseline_ompl/lazy_prm_eval.py`.

| Env | Success Rate | Path Time mean ± sd (s) | Path Length mean ± sd |
| --- | --- | --- | --- |
| rectangle_env1 | 99.6% | 7.099 ± 5.006 | 7.504 ± 3.991 |
| Lshape3d_env1 | 92.7% | 11.516 ± 7.601 | 7.570 ± 4.264 |
| Fshape3d_env1 | 73.3% | 13.970 ± 8.278 | 6.909 ± 4.016 |
| Ashape3d_env1 | 37.8% | 14.463 ± 9.071 | 4.938 ± 3.123 |
| Vshape3d_env1 | 51.1% | 15.120 ± 8.392 | 6.176 ± 3.462 |
| 4shape3d_env1 | 66.4% | 14.394 ± 8.364 | 6.704 ± 3.682 |
| rectangle_env2 | 99.7% | 6.185 ± 4.751 | 7.180 ± 4.042 |
| Lshape3d_env2 | 95.0% | 10.530 ± 7.157 | 7.436 ± 4.071 |
| Fshape3d_env2 | 77.1% | 13.493 ± 8.008 | 7.281 ± 4.233 |
| Ashape3d_env2 | 43.6% | 13.713 ± 8.618 | 5.298 ± 3.461 |
| Vshape3d_env2 | 57.9% | 14.531 ± 8.291 | 6.539 ± 3.892 |
| 4shape3d_env2 | 73.4% | 13.682 ± 8.281 | 7.127 ± 4.234 |
| rectangle_env3 | 99.3% | 2.795 ± 2.928 | 5.827 ± 3.175 |
| Lshape3d_env3 | 98.7% | 4.214 ± 3.852 | 5.500 ± 3.134 |
| Fshape3d_env3 | 98.0% | 6.429 ± 5.209 | 5.551 ± 3.189 |
| Ashape3d_env3 | 92.5% | 10.787 ± 7.639 | 5.291 ± 3.065 |
| Vshape3d_env3 | 96.4% | 8.676 ± 6.375 | 5.560 ± 3.191 |
| 4shape3d_env3 | 98.6% | 6.916 ± 5.473 | 5.504 ± 3.152 |
| rectangle_env4 | 99.5% | 3.433 ± 3.131 | 5.838 ± 3.172 |
| Lshape3d_env4 | 98.3% | 5.951 ± 4.721 | 5.629 ± 3.109 |
| Fshape3d_env4 | 98.2% | 8.627 ± 6.220 | 5.688 ± 3.065 |
| Ashape3d_env4 | 83.0% | 12.578 ± 8.278 | 5.337 ± 3.033 |
| Vshape3d_env4 | 89.4% | 10.497 ± 7.538 | 5.380 ± 3.107 |
| 4shape3d_env4 | 95.6% | 9.936 ± 6.988 | 5.826 ± 3.314 |

## How these were produced

```
python ../baseline_ompl/lazy_prm_eval.py \
    --obj      datasets/3dshape/<shape>.obj \
    --env      datasets/3dshape/<env>.obj \
    --dataPath testing_data/3dshape/<shape>_<env> \
    --n 0 --time 30 \
    --out      results/ompl_lazyprm/<shape>_<env>
```

Sweep driver: `lazyprm_logs/run_sweep.sh` (all 24 configs in parallel, one
process each, `OMP_NUM_THREADS=1`).  Per-config logs land in
`lazyprm_logs/<name>.log` and the full summaries in
`results/ompl_lazyprm/<name>/`.

**OMPL version**: this needs the **1.7.0** python bindings, in a venv at
`/opt/ompl17venv` (`pip install ompl==1.7.0`).  The 2.0.1 wheel used for
`RRT_experiments.md` exposes only 14 planners and has no lazy variants.  The
1.7.0 Boost.Python bindings also differ from 2.0.1 in two places the script
handles: the validity checker must be wrapped in `ob.StateValidityCheckerFn`,
and `setStartAndGoalStates` takes a `ScopedState` rather than a raw `State*`.
They additionally corrupt the heap on interpreter teardown, so the script
`os._exit(0)`s once its output files are flushed.

## Notes on the numbers

- **Test cases**: all 1000 start/goal pairs per config, the same
  `sampled_points.npy` the learned planner and RRT-Connect are scored on.
- **Success rate**: fraction of *scored* pairs solved.  Pairs whose start or
  goal is already in collision are unplannable by construction and are excluded
  from the denominator (0-2 per config, identical to the RRT-Connect run).
- **Time**: wall-clock `ss.solve()` time over the successful cases only, so it
  is directly comparable with the path-length column.  Measured with 24
  concurrent single-threaded processes on a 36-core host, so treat the timings
  as relative rather than best-case single-run latency.
- **Path length**: OMPL's SE(3) metric (weighted translation + rotation) in the
  normalized frame, measured on the raw planner output -- no path
  simplification (`--simplify` off), matching the RRT-Connect run.  Raw roadmap
  paths are jagged, so these lengths are roughly 2-3x the RRT-Connect ones and
  should not be read as a path-quality verdict on LazyPRM.
- **Single-query protocol**: the roadmap is cleared between test cases
  (`SimpleSetup.clear()`), so each case is planned from scratch exactly like the
  RRT-Connect baseline.  This deliberately gives up LazyPRM's multi-query
  advantage in a fixed environment; `lazy_prm_eval.py --reuse` keeps the roadmap
  across cases if the amortized number is wanted instead.

## How the failures fail

Every failure in the sweep is the 30 s budget running out.  No returned path was
ever in collision -- `path_in_collision` is 0 in all 24 configs, as it was for
RRT-Connect.

| failure mode | count across all 24 configs | what it means |
| --- | --- | --- |
| `no_solution_in_time` | 3634 | `solve()` used the full 30 s and returned no exact solution |
| `over_time_limit` | 215 | an exact path came back, but wall clock passed 30 s getting there |
| `path_in_collision` | 0 | never happened |

Failed cases cluster in a razor-thin band around the cap: min 30.00 s, median
~30.2 s, max 35.0 s.  Nothing gives up early; it only runs out of clock.

**The 30 s budget is binding for LazyPRM, unlike for RRT-Connect.**  Successful
cases run right up to the wire -- median success time is 2-13 s, but the slowest
*successful* case is 27-30 s in essentially every config.  The success-time
distribution is still live at the cap, so a longer budget would convert a real
fraction of these failures into successes.  Read the table as "success within
30 s", not "success, period".  By contrast RRT-Connect recorded exactly one
timeout in 24 000 cases, so the same budget barely constrains it -- part of the
head-to-head gap is the budget, not planner quality alone.
