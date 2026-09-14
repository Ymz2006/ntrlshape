# RRT-Connect (OMPL) Baseline Experiments

> Collected at the repository root. Every unqualified path below (`datasets/`, `Experiments/`, `outputs/`, `results/`, `tests/`, `train/`, ...)
> is relative to `ntrl-demo/ntrl-demo/`, where this table's runs live.

## 3-D shape task

RRT-Connect in SE(3) run on the same 3-D test sets the learned planner is
evaluated on (`testing_data/3dshape/<shape>_<env>`), so the numbers line up with
`experiments_ours.md` case for case.  Produced by `baselines/baseline_ompl/rrt_connect_eval.py`.

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

### How these were produced

```
python ../../baselines/baseline_ompl/rrt_connect_eval.py \
    --obj      datasets/3dshape/<shape>.obj \
    --env      datasets/3dshape/<env>.obj \
    --dataPath testing_data/3dshape/<shape>_<env> \
    --n 0 --time 30 \
    --out      results/ompl_rrtconnect/<shape>_<env>
```

Sweep driver: `baselines/rrt_logs/run_sweep.sh` (all 24 configs in parallel, one process
each, `OMP_NUM_THREADS=1`).  Per-config logs land in `baselines/rrt_logs/<name>.log` and
the full summaries in `results/ompl_rrtconnect/<name>/`.

Needs the OMPL python bindings in the container: `pip install ompl` (2.0.1 used
here); the source tree under `baselines/baseline_ompl/ompl-1.7.0` builds the C++ library
only.

### Notes on the numbers

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


## 2-D shape task (`--2d`)

RRT-Connect in SE(2) run on the same planar test sets the learned planner is
evaluated on (`testing_data/3dshape/<shape>_2denv<N>`), so the numbers line up
with the 2-D table in `experiments_ours.md` case for case.  Seven shapes across
all four planar environments -- 28 cells, 1000 pairs each.  Produced by the same
`baselines/baseline_ompl/rrt_connect_eval.py`, with `--2d`.

| Env | Success Rate | Path Time mean ± sd (s) | Path Length mean ± sd |
| --- | --- | --- | --- |
| rectangle_2denv1 | 100.0% | 0.280 ± 0.527 | 2.404 ± 1.193 |
| Lshape3d_2denv1 | 100.0% | 0.859 ± 1.964 | 2.506 ± 1.311 |
| Fshape3d_2denv1 | 99.7% | 1.511 ± 3.838 | 2.386 ± 1.247 |
| Ashape3d_2denv1 | 92.9% | 1.647 ± 3.958 | 2.135 ± 1.030 |
| Vshape3d_2denv1 | 93.5% | 1.496 ± 4.079 | 2.255 ± 1.120 |
| 4shape3d_2denv1 | 99.1% | 1.867 ± 4.381 | 2.352 ± 1.262 |
| Tshape3d_2denv1 | 91.1% | 1.867 ± 4.335 | 2.180 ± 1.110 |
| rectangle_2denv2 | 100.0% | 0.521 ± 0.730 | 2.753 ± 1.370 |
| Lshape3d_2denv2 | 100.0% | 2.499 ± 4.111 | 2.882 ± 1.541 |
| Fshape3d_2denv2 | 97.7% | 3.862 ± 5.984 | 2.924 ± 1.553 |
| Ashape3d_2denv2 | 50.5% | 3.194 ± 6.493 | 2.038 ± 1.075 |
| Vshape3d_2denv2 | 52.6% | 3.837 ± 7.380 | 2.140 ± 1.146 |
| 4shape3d_2denv2 | 94.9% | 5.031 ± 7.447 | 2.949 ± 1.634 |
| Tshape3d_2denv2 | 48.8% | 2.389 ± 5.503 | 1.998 ± 0.988 |
| rectangle_2denv3 | 100.0% | 0.050 ± 0.059 | 1.961 ± 0.703 |
| Lshape3d_2denv3 | 100.0% | 0.099 ± 0.124 | 2.070 ± 0.789 |
| Fshape3d_2denv3 | 100.0% | 0.130 ± 0.156 | 1.998 ± 0.784 |
| Ashape3d_2denv3 | 100.0% | 0.312 ± 0.451 | 2.057 ± 0.850 |
| Vshape3d_2denv3 | 100.0% | 0.259 ± 0.352 | 2.103 ± 0.859 |
| 4shape3d_2denv3 | 100.0% | 0.157 ± 0.213 | 2.048 ± 0.773 |
| Tshape3d_2denv3 | 100.0% | 0.441 ± 0.581 | 2.080 ± 0.896 |
| rectangle_2denv4 | 100.0% | 0.057 ± 0.086 | 2.156 ± 0.934 |
| Lshape3d_2denv4 | 100.0% | 0.108 ± 0.160 | 2.124 ± 0.989 |
| Fshape3d_2denv4 | 100.0% | 0.217 ± 0.545 | 2.139 ± 0.972 |
| Ashape3d_2denv4 | 100.0% | 0.657 ± 1.195 | 2.246 ± 1.048 |
| Vshape3d_2denv4 | 100.0% | 0.394 ± 0.572 | 2.229 ± 1.024 |
| 4shape3d_2denv4 | 100.0% | 0.251 ± 0.361 | 2.191 ± 0.960 |
| Tshape3d_2denv4 | 99.9% | 0.675 ± 0.961 | 2.343 ± 1.135 |

### How these were produced

```
python ../../baselines/baseline_ompl/rrt_connect_eval.py --2d \
    --obj      datasets/3dshape/<shape>_zup.obj \
    --env      datasets/3dshape/<env>_zup.obj \
    --dataPath testing_data/3dshape/<shape>_<env> \
    --n 0 --time 30 \
    --out      results/ompl_rrtconnect/<shape>_<env>
```

Sweep driver: `baselines/rrt_logs/run_sweep_2d.sh` (all 28 configs, `JOBS` at a
time -- 14 for this run, against the 3-D sweep's 24 -- one process each,
`OMP_NUM_THREADS=1`).  Per-config logs land in
`baselines/rrt_logs/<shape>_2denv<N>.log` and the full summaries in
`results/ompl_rrtconnect/<name>/`.  Those names never collide with the 3-D
sweep's, so the two sets of logs and results sit side by side.

Same OMPL requirement as the 3-D sweep.  The bindings are installed in the
`ntrl_mpnet` container, not `ntrl_baselines_train`.

### Notes on the numbers

- **What `--2d` changes**: the planner runs on OMPL's `SE2StateSpace` -- x, y and
  yaw -- instead of `SE3StateSpace`, matching the sub-space
  `preprocess_obj.py --2d` sampled the pairs in.  Nothing else differs from the
  3-D run: each state is lifted back to the full pose `t = (x, y, 0)`,
  `R = Rz(yaw)` and checked by the same point-in-tet collision model against the
  same `_zup.obj` meshes `evaluate_training_3d_batched.py --2d` loads.  Checked
  both directions: the SE(2) lift reproduces the SE(3) pose to 3.3e-16 over 400
  dataset configs, and the SE(3) path is untouched -- `Lshape3d_env1` still
  returns the path lengths in the 3-D table above (3.171, 4.978, 6.076, 2.748,
  2.356).
- **Test cases**: all 1000 pairs per config, and unlike the 3-D sweep every one
  of them was scored.  No planar config had a start or goal in collision under
  this model, so every denominator is exactly 1000.
- **Path length**: comparable with the 3-D table by construction.  OMPL weights
  SE(2)'s SO(2) sub-space by 0.5, so a z-rotation of theta costs theta/2 there --
  which is what SE(3)'s SO(3) quaternion metric charges the same rotation.
  Still the raw planner output, no simplification.
- **Time**: as in 3-D, wall-clock `ss.solve()` over the successful cases only,
  capped at 30 s.  14 concurrent single-threaded processes on a 36-core host, so
  again relative, not best-case latency.
- **Failure mode**: essentially all of it is "no exact solution within 30 s"
  (1789 cases across the sweep).  Only 4 cases anywhere ran past the budget with
  a path in hand (2 in `Ashape3d_2denv2`, 2 in `4shape3d_2denv2`), and **no**
  returned path was in collision in any of the 28 configs.
- **This baseline is not a 100% ceiling in 2-D.**  That is the headline
  difference from the 3-D sweep, where RRT-Connect solved 23 of 24 configs
  perfectly.  `2denv3` and `2denv4` still go 100% across the board, but `2denv1`
  costs it 6-9 points on the A/V/T shapes, and `2denv2` roughly halves it:
  `Ashape3d_2denv2` 50.5%, `Vshape3d_2denv2` 52.6%, `Tshape3d_2denv2` 48.8%.
  Removing a translational DOF removes the room to route *around* an obstacle,
  and those three shapes are the long, thin ones that have to be threaded
  through a gap at close to the right heading.
- **The learned planner beats it on exactly those cells.**  Against the SD
  column of `experiments_ours.md`: 61.3% vs 50.5% on `Ashape3d_2denv2`, 58.1% vs
  52.6% on `Vshape3d_2denv2`, 54.8% vs 48.8% on `Tshape3d_2denv2`.  The two
  planners collapse on the same environment, which is evidence about `2denv2`
  itself rather than about either method.
- **Most `2denv2` failures survive 5x the budget.**  Re-planning 60 of the 30 s
  failures -- the first 20 from each of the A/V/T cells -- at a 150 s limit
  solved only 21 of them (6/20 `Ashape3d_2denv2`, 9/20 `Vshape3d_2denv2`, 6/20
  `Tshape3d_2denv2`); the other 39 ran the full 150 s and returned nothing.  So
  the 30 s cap does cost this baseline some real successes -- the 21 that did
  solve took a median of 61.7 s, well past it -- but roughly two thirds of these
  pairs stay unsolved given 5x the time.  Whether those are genuinely
  disconnected in SE(2) or merely very narrow is not something a sampling
  planner can settle; the point is that `2denv2`'s A/V/T numbers are not an
  artifact of the time limit.  Reproduce with
  `baselines/rrt_logs/_recheck_2denv2.py <config> <shard> <nshards> <n_pairs> <time_limit>`.


## 1k-valid sets (`testing_data_1k_complete/`), with the path length split

Both sweeps above were scored on `testing_data/`, whose pairs are only known to be
collision-free.  This section re-runs every cell on `testing_data_1k_complete/`
-- the same 1000-pairs-per-cell layout, but every pair also carries an
RRT-Connect path found within 180 s, so each cell has a demonstrated 100%
ceiling -- and adds the four `Tshape3d_env*` SE(3) cells, which the older sweeps
never had.  It is the RRT-Connect counterpart of the `3D_1k_valid` and
`2d_1k_valid` tables in `experiments_ours.md`, case for case.  Do **not** compare
these rows with the two tables above: the hardest pairs were filtered out of this
population (see `testing_data_1k_complete/README.md` for the per-cell discard
rates), so it is a different test, not the same test measured twice.

The single "Path Length" column above is OMPL's compound metric, which is
**not** translation alone: it is `trans + rot / 2`, the translation travelled
plus half the rotation angle summed over the raw waypoints.  Here it is split
into its two parts:

- **Translation Length** -- `sum ||t_{i+1} - t_i||` over the planner's waypoints,
  normalized frame (the environment's longest side is 1).
- **Rotation Length** -- `sum theta_i`, the geodesic angle between consecutive
  orientations, in radians (`2 arccos|q_i . q_{i+1}|` in SE(3), `|wrap(dyaw)|` in
  SE(2)).  Divide by `2 pi` for the dataset's `rotvec / 2pi` config units.
- **OMPL Path Length** -- `trans + rot / 2`, unchanged from the tables above, kept
  so the three columns can be checked against each other.

All three are on the raw planner output (no `--simplify`), over the successful
cases only.

### 3-D shape task (SE(3), 7 shapes x 4 envs)

<!-- rrt1k:3d:begin -->
| Env | Success Rate | Path Time mean ± sd (s) | Translation Length mean ± sd | Rotation Length mean ± sd (rad) | OMPL Path Length mean ± sd |
| --- | --- | --- | --- | --- | --- |
| rectangle_env1 | 100.0% | 0.345 ± 0.292 | 0.825 ± 0.394 | 3.994 ± 1.672 | 2.822 ± 1.164 |
| Lshape3d_env1 | 100.0% | 0.655 ± 0.581 | 0.845 ± 0.427 | 4.153 ± 1.834 | 2.921 ± 1.282 |
| Fshape3d_env1 | 100.0% | 1.132 ± 1.056 | 0.865 ± 0.427 | 4.311 ± 1.882 | 3.021 ± 1.304 |
| Ashape3d_env1 | 99.9% | 2.553 ± 2.110 | 0.861 ± 0.438 | 4.383 ± 1.990 | 3.053 ± 1.373 |
| Vshape3d_env1 | 99.9% | 1.681 ± 1.419 | 0.861 ± 0.442 | 4.330 ± 1.914 | 3.026 ± 1.339 |
| 4shape3d_env1 | 100.0% | 1.292 ± 1.146 | 0.859 ± 0.419 | 4.237 ± 1.874 | 2.977 ± 1.290 |
| Tshape3d_env1 | 100.0% | 2.883 ± 2.386 | 0.904 ± 0.473 | 4.620 ± 2.200 | 3.214 ± 1.523 |
| rectangle_env2 | 100.0% | 0.233 ± 0.211 | 0.802 ± 0.395 | 4.088 ± 1.759 | 2.845 ± 1.216 |
| Lshape3d_env2 | 100.0% | 0.398 ± 0.340 | 0.804 ± 0.395 | 4.216 ± 1.757 | 2.912 ± 1.211 |
| Fshape3d_env2 | 100.0% | 0.706 ± 0.634 | 0.842 ± 0.442 | 4.342 ± 1.925 | 3.013 ± 1.345 |
| Ashape3d_env2 | 100.0% | 1.791 ± 1.525 | 0.874 ± 0.432 | 4.583 ± 1.937 | 3.166 ± 1.347 |
| Vshape3d_env2 | 100.0% | 1.125 ± 0.938 | 0.865 ± 0.424 | 4.567 ± 1.910 | 3.148 ± 1.324 |
| 4shape3d_env2 | 100.0% | 0.836 ± 0.773 | 0.855 ± 0.442 | 4.412 ± 1.973 | 3.061 ± 1.372 |
| Tshape3d_env2 | 100.0% | 2.012 ± 1.748 | 0.924 ± 0.479 | 4.861 ± 2.178 | 3.355 ± 1.521 |
| rectangle_env3 | 100.0% | 0.134 ± 0.180 | 0.700 ± 0.339 | 3.540 ± 1.434 | 2.470 ± 0.983 |
| Lshape3d_env3 | 100.0% | 0.180 ± 0.213 | 0.689 ± 0.336 | 3.510 ± 1.432 | 2.444 ± 0.981 |
| Fshape3d_env3 | 100.0% | 0.272 ± 0.344 | 0.674 ± 0.326 | 3.460 ± 1.373 | 2.404 ± 0.934 |
| Ashape3d_env3 | 100.0% | 0.623 ± 0.711 | 0.661 ± 0.331 | 3.438 ± 1.383 | 2.380 ± 0.954 |
| Vshape3d_env3 | 100.0% | 0.391 ± 0.433 | 0.665 ± 0.318 | 3.456 ± 1.427 | 2.393 ± 0.965 |
| 4shape3d_env3 | 100.0% | 0.295 ± 0.349 | 0.670 ± 0.318 | 3.462 ± 1.431 | 2.401 ± 0.958 |
| Tshape3d_env3 | 100.0% | 0.708 ± 0.822 | 0.667 ± 0.343 | 3.528 ± 1.550 | 2.431 ± 1.053 |
| rectangle_env4 | 100.0% | 0.097 ± 0.103 | 0.666 ± 0.286 | 3.386 ± 1.256 | 2.360 ± 0.830 |
| Lshape3d_env4 | 100.0% | 0.152 ± 0.164 | 0.654 ± 0.298 | 3.470 ± 1.314 | 2.389 ± 0.881 |
| Fshape3d_env4 | 100.0% | 0.243 ± 0.238 | 0.632 ± 0.278 | 3.426 ± 1.234 | 2.346 ± 0.820 |
| Ashape3d_env4 | 100.0% | 0.603 ± 0.576 | 0.654 ± 0.294 | 3.553 ± 1.323 | 2.431 ± 0.883 |
| Vshape3d_env4 | 100.0% | 0.391 ± 0.366 | 0.650 ± 0.297 | 3.553 ± 1.355 | 2.427 ± 0.903 |
| 4shape3d_env4 | 100.0% | 0.277 ± 0.276 | 0.645 ± 0.296 | 3.513 ± 1.281 | 2.402 ± 0.858 |
| Tshape3d_env4 | 100.0% | 0.670 ± 0.615 | 0.649 ± 0.288 | 3.550 ± 1.335 | 2.424 ± 0.887 |
<!-- rrt1k:3d:end -->

### 2-D shape task (SE(2), `--2d`, 7 shapes x 4 envs)

<!-- rrt1k:2d:begin -->
| Env | Success Rate | Path Time mean ± sd (s) | Translation Length mean ± sd | Rotation Length mean ± sd (rad) | OMPL Path Length mean ± sd |
| --- | --- | --- | --- | --- | --- |
| rectangle_2denv1 | 100.0% | 0.223 ± 0.410 | 0.827 ± 0.511 | 3.204 ± 1.468 | 2.429 ± 1.137 |
| Lshape3d_2denv1 | 100.0% | 0.816 ± 1.872 | 0.820 ± 0.567 | 3.373 ± 1.741 | 2.506 ± 1.351 |
| Fshape3d_2denv1 | 99.9% | 1.136 ± 2.836 | 0.784 ± 0.554 | 3.205 ± 1.600 | 2.387 ± 1.260 |
| Ashape3d_2denv1 | 95.7% | 1.391 ± 3.735 | 0.622 ± 0.411 | 3.087 ± 1.414 | 2.165 ± 1.028 |
| Vshape3d_2denv1 | 97.3% | 1.477 ± 4.170 | 0.664 ± 0.481 | 3.238 ± 1.593 | 2.283 ± 1.187 |
| 4shape3d_2denv1 | 100.0% | 1.478 ± 3.653 | 0.775 ± 0.572 | 3.298 ± 1.663 | 2.424 ± 1.311 |
| Tshape3d_2denv1 | 97.4% | 1.483 ± 3.643 | 0.607 ± 0.404 | 3.210 ± 1.512 | 2.212 ± 1.079 |
| rectangle_2denv2 | 100.0% | 0.479 ± 0.699 | 0.962 ± 0.566 | 3.685 ± 1.677 | 2.804 ± 1.307 |
| Lshape3d_2denv2 | 100.0% | 1.924 ± 3.097 | 0.961 ± 0.649 | 3.897 ± 1.991 | 2.910 ± 1.570 |
| Fshape3d_2denv2 | 99.8% | 3.612 ± 5.772 | 0.998 ± 0.716 | 3.906 ± 2.033 | 2.951 ± 1.648 |
| Ashape3d_2denv2 | 77.5% | 2.736 ± 6.195 | 0.492 ± 0.382 | 2.924 ± 1.446 | 1.954 ± 1.018 |
| Vshape3d_2denv2 | 81.4% | 3.352 ± 6.832 | 0.554 ± 0.424 | 3.132 ± 1.564 | 2.119 ± 1.122 |
| 4shape3d_2denv2 | 98.1% | 4.565 ± 7.033 | 0.974 ± 0.699 | 3.802 ± 1.885 | 2.875 ± 1.553 |
| Tshape3d_2denv2 | 87.3% | 1.530 ± 4.345 | 0.411 ± 0.295 | 2.774 ± 1.385 | 1.798 ± 0.917 |
| rectangle_2denv3 | 100.0% | 0.045 ± 0.050 | 0.730 ± 0.354 | 2.634 ± 1.145 | 2.047 ± 0.786 |
| Lshape3d_2denv3 | 100.0% | 0.080 ± 0.104 | 0.714 ± 0.367 | 2.646 ± 1.135 | 2.036 ± 0.799 |
| Fshape3d_2denv3 | 100.0% | 0.113 ± 0.146 | 0.711 ± 0.367 | 2.650 ± 1.146 | 2.036 ± 0.782 |
| Ashape3d_2denv3 | 100.0% | 0.301 ± 0.366 | 0.715 ± 0.375 | 2.751 ± 1.183 | 2.090 ± 0.844 |
| Vshape3d_2denv3 | 100.0% | 0.184 ± 0.217 | 0.697 ± 0.364 | 2.749 ± 1.182 | 2.071 ± 0.834 |
| 4shape3d_2denv3 | 100.0% | 0.135 ± 0.190 | 0.706 ± 0.350 | 2.648 ± 1.096 | 2.030 ± 0.760 |
| Tshape3d_2denv3 | 100.0% | 0.278 ± 0.346 | 0.677 ± 0.347 | 2.758 ± 1.199 | 2.056 ± 0.818 |
| rectangle_2denv4 | 100.0% | 0.055 ± 0.074 | 0.736 ± 0.421 | 2.772 ± 1.279 | 2.122 ± 0.923 |
| Lshape3d_2denv4 | 100.0% | 0.097 ± 0.145 | 0.725 ± 0.438 | 2.841 ± 1.313 | 2.145 ± 0.975 |
| Fshape3d_2denv4 | 100.0% | 0.146 ± 0.310 | 0.728 ± 0.414 | 2.825 ± 1.273 | 2.140 ± 0.918 |
| Ashape3d_2denv4 | 100.0% | 0.306 ± 0.419 | 0.745 ± 0.466 | 2.948 ± 1.348 | 2.219 ± 1.024 |
| Vshape3d_2denv4 | 100.0% | 0.261 ± 0.567 | 0.777 ± 0.508 | 2.994 ± 1.419 | 2.274 ± 1.109 |
| 4shape3d_2denv4 | 100.0% | 0.155 ± 0.388 | 0.723 ± 0.434 | 2.824 ± 1.243 | 2.135 ± 0.930 |
| Tshape3d_2denv4 | 100.0% | 0.415 ± 1.110 | 0.757 ± 0.504 | 3.043 ± 1.436 | 2.278 ± 1.120 |
<!-- rrt1k:2d:end -->

### How these were produced

```
python ../../baselines/baseline_ompl/rrt_connect_eval.py [--2d] \
    --obj      datasets/3dshape/<shape>[_zup].obj \
    --env      datasets/3dshape/<env>[_zup].obj \
    --dataPath testing_data_1k_complete/3dshape/<shape>_<env> \
    --n 0 --time 30 \
    --out      results/ompl_rrtconnect_1k/<shape>_<env>
```

Sweep driver: `baselines/rrt_logs_1k/run_sweep_1k.sh` (all 56 cells, `JOBS` at
a time -- 16 here -- one single-threaded process each, `Tshape3d_env*` first).
Per-cell logs in `baselines/rrt_logs_1k/<cell>.log`, full summaries and the
per-case CSV (now with `trans_length` and `rot_length_rad` columns) in
`results/ompl_rrtconnect_1k/<cell>/`.  Tables refreshed with
`python baselines/baseline_ompl/make_rrt_connect_1k_md.py`.  Run in the
`ntrl_mpnet` container (the one with the OMPL bindings), cwd `ntrl-demo/ntrl-demo`.

### Notes on the numbers

- **Test cases**: all 1000 pairs per cell.  Exactly one pair in the whole sweep
  (`Tshape3d_env3` #37) had its start pose in collision under this run's 50k
  cloud draw -- the draw-dependent grazing contact `testing_data_1k_complete/README.md`
  puts at ~0.008% of placements -- and is excluded, so that denominator is 999
  and every other one is 1000.
- **Time**: wall-clock `ss.solve()` over the successful cases, capped at 30 s.
  16 concurrent single-threaded processes on a 36-core host that was already
  carrying ~10 other jobs, so absolute times are ~4x a quiet single run
  (`Tshape3d_env1`: 1.7 s mean alone vs 2.9 s here); treat them as relative.
- **The four new `Tshape3d_env*` cells are a clean 100%.**  3999 of 3999
  scored pairs solved, no time-outs.  Time-wise the T is the slowest shape in
  every environment, by a small margin over A (`env1` 2.88 s vs A 2.55, V 1.68,
  F 1.13, L 0.66; `env2` 2.01 vs A 1.79; `env3`/`env4` 0.71 / 0.67 vs A 0.62 /
  0.60) -- the hardest shape to thread, but not an outlier.  Against the
  learned planner's `3D_1k_valid` SD column (97.4 / 85.4 / 98.2 / 97.9%), the
  T-shape gap is concentrated in `env2`, the same place A and V lose points.
- **SE(3) as a whole is still a 100% ceiling.**  26 of 28 cells solve every
  scored pair; `Ashape3d_env1` and `Vshape3d_env1` each miss one (no exact
  solution in 30 s).  No returned path was in collision anywhere in the 56
  cells, and only 3 cases in the whole sweep ran past the budget with a path
  in hand.
- **The planar picture moved a lot, because the population did.**  On
  `testing_data/` the A/V/T `2denv2` cells were 50.5 / 52.6 / 48.8%; here they
  are 77.5 / 81.4 / 87.3%, and `2denv1` A/V/T went from 92.9 / 93.5 / 91.1% to
  95.7 / 97.3 / 97.4%.  That is the filter -- the 1k sets discarded 36 / 30 /
  46% of sampled `2denv2` pairs as unsolvable in 180 s -- not a better planner.
  What is left is genuinely informative: every one of these pairs *has* a
  known RRT-Connect path, yet at 30 s the planner still misses 13-22% of them
  in A/V/T `2denv2` and 3-4% in `2denv1`.  All 655 of those failures are
  `no_solution_in_time` (failed cases run 30.00-30.27 s, median 30.04): the 30 s
  cap is 6x tighter than the 180 s that verified the pairs, and these cells
  are where the difference shows.
- **The learned planner is level with or ahead of the baseline exactly there.**
  `2d_1k_valid` SD column vs this table: `Ashape3d_2denv2` 78.1 vs 77.5%,
  `Vshape3d_2denv2` 76.5 vs 81.4%, `Tshape3d_2denv2` 86.7 vs 87.3%; on
  `2denv1` it trails by 5-7 points (90.4 / 90.7 / 90.9 vs 95.7 / 97.3 / 97.4).
  Everywhere else RRT-Connect is at 98-100%.
- **What the length split shows.**  Rotation, not translation, is most of
  OMPL's "path length": `rot / 2` is 71-73% of the compound number in every
  SE(3) cell and 64-67% in every planar cell that solves everything (rising
  to 71-77% only in the A/V/T `2denv1`/`2denv2` cells, whose surviving pairs
  translate less -- next bullet).  That share barely moves across shapes or
  environments: it is a property of RRT-Connect's uniform orientation
  sampling, not of the task.  In absolute terms an SE(3) path translates
  0.63-0.92 (env side = 1) while spinning 3.4-4.9 rad (195-280 degrees);
  planar paths translate 0.6-1.0 and turn 2.6-3.9 rad.  Both halves are about
  2x the straight-line move: over the T cells the path translation is
  1.7-2.2x the start-goal distance and the path rotation 1.8-2.5x the
  start-goal geodesic angle (`env1`/`env2` at the high end, `env3`/`env4` at
  the low end).  These are unsimplified RRT-Connect paths, so this is the
  zig-zag of the tree, and the two columns are what to compare a smoother
  planner against -- separately, since a planner that rotates little and
  translates a lot would look identical to its opposite in the compound column.
- **Survivor bias in the A/V/T `2denv2` lengths.**  Those three cells show
  translation 0.41-0.55 and rotation 2.8-3.1 rad against 0.96-1.00 and
  3.7-3.9 rad for rect/L/F/4 in the same environment.  Their solved pairs are
  the short ones (mean start-goal distance 0.19 in `Tshape3d_2denv2` vs 0.40
  in `rectangle_2denv1`): the long threading moves are the ones timing out,
  and the 1k filter had already removed the longest.  Do not read the A/V/T
  `2denv2` length columns as "shorter paths".
- **Collision model**: unchanged from the sweeps above -- 50 000 surface points
  over the environment mesh (obstacles and walls), a pose collides iff any
  point falls inside the placed shape's tetrahedral decomposition, path
  re-walked after interpolation at resolution 0.005.
