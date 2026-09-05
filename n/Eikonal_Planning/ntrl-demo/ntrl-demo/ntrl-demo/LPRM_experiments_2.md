# LazyPRM (OMPL) Baseline Experiments -- env1 Re-run

LazyPRM in SE(3) run on the same 3-D test sets the learned planner is evaluated
on (`testing_data/3dshape/<shape>_<env>`), so the numbers line up with
`experiments.md` and `RRT_experiments.md` case for case.  Produced by
`baseline_ompl/lazy_prm_eval.py`.

This is a **re-run of env1 only**, with settings identical to `LRPM_experiments.md`
(30 s budget, roadmap cleared per case, seed 1).  The env2/env3/env4 rows are left
blank because they were not part of this re-run; their numbers are in
`LRPM_experiments.md`.  Results land in `results/ompl_lazyprm_rerun/` so the
original sweep is untouched.

| Env | Success Rate | Path Time mean ± sd (s) | Path Length mean ± sd |
| --- | --- | --- | --- |
| rectangle_env1 | 100.0% | 2.628 ± 1.872 | 7.514 ± 3.998 |
| Lshape3d_env1 | 99.8% | 5.439 ± 3.987 | 7.507 ± 4.070 |
| Fshape3d_env1 | 99.3% | 8.804 ± 6.328 | 7.699 ± 4.288 |
| Ashape3d_env1 | 62.7% | 13.468 ± 8.445 | 6.117 ± 3.741 |
| Vshape3d_env1 | 80.9% | 12.540 ± 8.241 | 7.022 ± 4.172 |
| 4shape3d_env1 | 97.1% | 10.025 ± 6.857 | 7.641 ± 4.306 |
| rectangle_env2 |  |  |  |
| Lshape3d_env2 |  |  |  |
| Fshape3d_env2 |  |  |  |
| Ashape3d_env2 |  |  |  |
| Vshape3d_env2 |  |  |  |
| 4shape3d_env2 |  |  |  |
| rectangle_env3 |  |  |  |
| Lshape3d_env3 |  |  |  |
| Fshape3d_env3 |  |  |  |
| Ashape3d_env3 |  |  |  |
| Vshape3d_env3 |  |  |  |
| 4shape3d_env3 |  |  |  |
| rectangle_env4 |  |  |  |
| Lshape3d_env4 |  |  |  |
| Fshape3d_env4 |  |  |  |
| Ashape3d_env4 |  |  |  |
| Vshape3d_env4 |  |  |  |
| 4shape3d_env4 |  |  |  |

## How these were produced

```
python ../baseline_ompl/lazy_prm_eval.py \
    --obj      datasets/3dshape/<shape>.obj \
    --env      datasets/3dshape/<env>.obj \
    --dataPath testing_data/3dshape/<shape>_<env> \
    --n 0 --time 30 \
    --out      results/ompl_lazyprm_rerun/<shape>_<env>
```

Sweep driver: `lazyprm_logs/run_sweep_env1_rerun.sh` (the 6 env1 configs in
parallel, one process each, `OMP_NUM_THREADS=1`).  Per-config logs land in
`lazyprm_logs/rerun_env1/<name>.log` and the full summaries in
`results/ompl_lazyprm_rerun/<name>/`.

**OMPL version**: this needs the **1.7.0** python bindings, in a venv at
`/opt/ompl17venv` (`pip install ompl==1.7.0`).  The 2.0.1 wheel used for
`RRT_experiments.md` exposes only 14 planners and has no lazy variants.  The
1.7.0 Boost.Python bindings also differ from 2.0.1 in two places the script
handles: the validity checker must be wrapped in `ob.StateValidityCheckerFn`,
and `setStartAndGoalStates` takes a `ScopedState` rather than a raw `State*`.
They additionally corrupt the heap on interpreter teardown, so the script
`os._exit(0)`s once its output files are flushed.

## Notes on the numbers

- **Test cases**: all 1000 start/goal pairs per env1 config (6000 scored cases
  in total), the same `sampled_points.npy` the learned planner and RRT-Connect
  are scored on.
- **Success rate**: fraction of *scored* pairs solved.  Pairs whose start or
  goal is already in collision are unplannable by construction and are excluded
  from the denominator -- zero of them in the env1 test sets.
- **Time**: wall-clock `ss.solve()` time over the successful cases only, so it
  is directly comparable with the path-length column.  Measured with **6**
  concurrent single-threaded processes on a 36-core host -- not the 24 of the
  original sweep.  That difference matters a lot here; see "What changed versus
  the first run" below.
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

Every failure in this re-run is the 30 s budget running out.  No returned path
was ever in collision -- `path_in_collision` is 0 in all 6 configs, as it was in
the original sweep and for RRT-Connect.

| failure mode | count across the 6 env1 configs | what it means |
| --- | --- | --- |
| `no_solution_in_time` | 563 | `solve()` used the full 30 s and returned no exact solution |
| `over_time_limit` | 39 | an exact path came back, but wall clock passed 30 s getting there |
| `path_in_collision` | 0 | never happened |

Failed cases cluster in a razor-thin band around the cap: min 30.00 s, median
30.17 s, max 32.77 s.  Nothing gives up early; it only runs out of clock.

**The 30 s budget is binding for LazyPRM, unlike for RRT-Connect.**  Successful
cases run right up to the wire -- median success time is 2-13 s, but the slowest
*successful* case is 26-30 s in five of the six configs (only `rectangle_env1`,
at 15.07 s, finishes with real headroom).  The success-time
distribution is still live at the cap, so a longer budget would convert a real
fraction of these failures into successes.  Read the table as "success within
30 s", not "success, period".  By contrast RRT-Connect recorded exactly one
timeout in 24 000 cases, so the same budget barely constrains it -- part of the
head-to-head gap is the budget, not planner quality alone.

## What changed versus the first run

Identical settings, identical seed, identical test cases -- the only difference
is that this re-run had **6 processes competing for the host instead of 24**.
The success rates move enormously:

| Env | SR first run | SR re-run | Δ | time mean first run | time mean re-run |
| --- | --- | --- | --- | --- | --- |
| rectangle_env1 | 99.6% | 100.0% | +0.4 | 7.099 | 2.630 |
| Lshape3d_env1 | 92.7% | 99.8% | +7.1 | 11.516 | 5.440 |
| Fshape3d_env1 | 73.3% | 99.3% | +26.0 | 13.970 | 8.797 |
| Ashape3d_env1 | 37.8% | 62.7% | +24.9 | 14.463 | 13.468 |
| Vshape3d_env1 | 51.1% | 80.9% | +29.8 | 15.120 | 12.539 |
| 4shape3d_env1 | 66.4% | 97.1% | +30.7 | 14.394 | 10.030 |

Because the 30 s limit is wall-clock, a contended process gets less actual CPU
inside the budget and so explores a smaller roadmap before the clock runs out.
Mean solve time on successes fell by up to 2.7x, and the freed time converted
directly into solved cases.

Two consequences worth taking seriously:

1. **The env1 numbers in `LRPM_experiments.md` understate LazyPRM badly** -- by
   up to 30 points of success rate.  They measure LazyPRM-under-24-way-contention,
   not LazyPRM.  The same caveat applies to its env2/env3/env4 rows, which were
   produced under the same 24-way load and have not been re-run.
2. **Path length is the one stable quantity.**  It moved by at most ~1.2 units
   and is unchanged for the easy shapes, because it is a property of the path
   found rather than of how much compute was available.  Cross-run success-rate
   and timing comparisons need matched parallelism; path length does not.

`RRT_experiments.md` is far less exposed to this: it recorded one timeout in
24 000 cases, so its success rates are not budget-limited.  Its *timings* were
still measured 24-way and are inflated by the same effect.
