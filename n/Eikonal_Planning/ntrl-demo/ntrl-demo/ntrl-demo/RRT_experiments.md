# RRT-Connect (OMPL) Baseline Experiments

RRT-Connect in SE(3) run on the same 3-D test sets the learned planner is
evaluated on (`testing_data/3dshape/<shape>_<env>`), so the numbers line up with
`experiments.md` case for case.  Produced by `baseline_ompl/rrt_connect_eval.py`.

| Env | Success Rate | Path Time mean ± sd (s) | Path Length mean ± sd |
| --- | --- | --- | --- |
| rectangle_env1 | 100.0% | 0.452 ± 0.373 | 2.898 ± 1.212 |
| Lshape3d_env1 | 100.0% | 0.939 ± 0.759 | 3.047 ± 1.297 |
| Fshape3d_env1 | 100.0% | 1.703 ± 1.374 | 3.039 ± 1.310 |
| Ashape3d_env1 | 99.9% | 3.495 ± 3.093 | 3.152 ± 1.387 |
| Vshape3d_env1 | 100.0% | 2.558 ± 2.170 | 3.162 ± 1.383 |
| 4shape3d_env1 | 100.0% | 1.981 ± 1.713 | 3.058 ± 1.330 |
| rectangle_env2 | 100.0% | 0.256 ± 0.227 | 2.821 ± 1.187 |
| Lshape3d_env2 | 100.0% | 0.529 ± 0.461 | 3.043 ± 1.334 |
| Fshape3d_env2 | 100.0% | 1.037 ± 0.978 | 3.105 ± 1.405 |
| Ashape3d_env2 | 100.0% | 2.499 ± 2.405 | 3.147 ± 1.420 |
| Vshape3d_env2 | 100.0% | 1.723 ± 1.393 | 3.221 ± 1.363 |
| 4shape3d_env2 | 100.0% | 1.174 ± 1.094 | 3.039 ± 1.330 |
| rectangle_env3 | 100.0% | 0.152 ± 0.185 | 2.442 ± 1.012 |
| Lshape3d_env3 | 100.0% | 0.218 ± 0.275 | 2.404 ± 0.992 |
| Fshape3d_env3 | 100.0% | 0.383 ± 0.502 | 2.384 ± 0.961 |
| Ashape3d_env3 | 100.0% | 1.098 ± 1.361 | 2.400 ± 0.971 |
| Vshape3d_env3 | 100.0% | 0.659 ± 0.806 | 2.417 ± 1.022 |
| 4shape3d_env3 | 100.0% | 0.463 ± 0.578 | 2.394 ± 0.936 |
| rectangle_env4 | 100.0% | 0.134 ± 0.132 | 2.415 ± 0.858 |
| Lshape3d_env4 | 100.0% | 0.220 ± 0.246 | 2.468 ± 0.934 |
| Fshape3d_env4 | 100.0% | 0.379 ± 0.417 | 2.441 ± 0.891 |
| Ashape3d_env4 | 100.0% | 1.064 ± 1.090 | 2.452 ± 0.932 |
| Vshape3d_env4 | 100.0% | 0.625 ± 0.636 | 2.407 ± 0.894 |
| 4shape3d_env4 | 100.0% | 0.456 ± 0.492 | 2.470 ± 0.932 |

## How these were produced

```
python ../baseline_ompl/rrt_connect_eval.py \
    --obj      datasets/3dshape/<shape>.obj \
    --env      datasets/3dshape/<env>.obj \
    --dataPath testing_data/3dshape/<shape>_<env> \
    --n 0 --time 30 \
    --out      results/ompl_rrtconnect/<shape>_<env>
```

Sweep driver: `rrt_logs/run_sweep.sh` (all 24 configs in parallel, one process
each, `OMP_NUM_THREADS=1`).  Per-config logs land in `rrt_logs/<name>.log` and
the full summaries in `results/ompl_rrtconnect/<name>/`.

Needs the OMPL python bindings in the container: `pip install ompl` (2.0.1 used
here); the source tree under `baseline_ompl/ompl-1.7.0` builds the C++ library
only.

## Notes on the numbers

- **Test cases**: all 1000 start/goal pairs per config, the same
  `sampled_points.npy` the learned planner is scored on.
- **Success rate**: fraction of *scored* pairs solved.  A pair whose start or
  goal is already in collision under this collision model is unplannable by
  construction and is excluded from the denominator rather than counted against
  the planner.  That happened for 0-2 pairs per config here (1 in
  `Lshape3d_env3`, `Fshape3d_env3`, `Ashape3d_env3`, `Vshape3d_env3`,
  `Ashape3d_env4`; 2 in `4shape3d_env3`; 0 everywhere else), so every
  denominator is 998-1000.
- **Time**: wall-clock `ss.solve()` time, over the successful cases only, so it
  is directly comparable with the path-length column.  A case is capped at 30 s
  and counts as a failure past that.  Timings come from 24 concurrent
  single-threaded processes on a 36-core host, so they carry some contention;
  treat them as relative, not as best-case single-run latency.
- **Path length**: OMPL's SE(3) metric (weighted translation + rotation) in the
  normalized frame, measured on the raw planner output -- no path
  simplification (`--simplify` off).
- **Only failure in the sweep**: one case in `Ashape3d_env1` returned no exact
  solution within 30 s.  No case exceeded the time limit, and no returned path
  was in collision.
- **Collision model**: the environment mesh (obstacles *and* walls) sampled to
  50 000 surface points; a pose collides iff any env point falls inside the
  placed shape's tetrahedral decomposition -- the same test that labels the
  training data.
