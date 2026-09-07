# Experiments -- `models/metric` on the 3-D shape task

> Collected at the repository root. Every unqualified path below (`datasets/`, `Experiments/`, `outputs/`, `results/`, `tests/`, `train/`, ...)
> is relative to `baselines/ntrl-demo/`, where this table's runs live. Regenerate with `python train/make_experiments_md.py --package metric ... --out ../../experiments_metric_ntfields.md` from that directory.

Trained with `train/train_3dshape_metric_all.sh` (4000 epochs per environment, dim = 6, source = origin) on the same datasets, test sets and controller as `experiments_metric_arm.md`, which tabulates `models/metric_arm` over the same environments. The two packages are the same single-route network and differ only in hyperparameters -- the 0.2 vs 0.1 output scale, the unscaled vs 0.2-scaled Fourier matrix `B`, the 1e-3 vs 0 normal-loss weight and the 0.03 vs 0.02 ray length -- so the two tables are directly comparable row by row.

This is the `baselines/ntrl-demo` copy of `models/metric`, which has since diverged from the main repo's `models/metric`; the numbers in `experiments_ours.md` are not this network.

Training passes `--alpha 1.0`. `models/metric` defaults to a speed stretch of `alpha = 1.025`, applied as `alpha*speed + 1 - alpha`, which assumes speeds never fall below `(alpha-1)/alpha = 0.0244`. These datasets do not clear that floor -- after the quartic transform `s^2 (2-s)^2` about 14% of `rectangle_env1` sits below it -- so the stretched speed goes negative and the `sqrt` in `Function.Loss` returns NaN on the first batch. 1.0 is also what `models/metric_arm` hardcodes, which keeps the comparison on equal footing.

`Valid` counts pairs whose rollout reached the goal AND stayed collision-free
at every waypoint, over the pairs left after screening out any whose start or
goal pose already collides. The collision test is the one in the main repo's
`evaluate_training_3d_batched.py`, so the column is closest to its `SR Forward`,
though the controller here is the simpler `tests/arm_plan_stat.py` one (no
taper, no local-weight term, no rescue pass).

The planning columns come from `tests/3d_plan.py`: the MPPI controller of
`tests/arm_plan_stat.py` run on each environment's 1000 held-out start/goal
pairs, one path at a time. A pair converges when the rollout reaches within
0.01 of the goal inside 200 steps; plan time and path length are averaged over
the valid paths, and length is the sum of segment norms in the 6-D
configuration space (translation plus 2*pi-scaled rotation). Plan time covers
the rollout alone -- collision checking and bookkeeping run afterwards.

`Valid (tight)` is the same measurement on the `<env>_tight` test sets, which
`dataprocessing/make_tight_testing_data.py` regenerates at `--offset 0.005`
instead of 0.02: start and goal poses may sit four times closer to obstacles,
so the queries are harder and more of them are screened out for an endpoint
that already collides. Per-set timing and length for those runs are in
`results/3d_plan_metric_tight/<env>/summary.json`.

| Env | Model | Epochs | Final Loss | Train Time (s) | Valid | Plan Time (s) | Path Length | Valid (tight) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./Experiments/3dshape_metric/rectangle_env1_09_05_23_44/latest.pt` | 4000 | 3.6956e-03 | 1328 | 81.8% (818/1000) | 0.088 ± 0.034 | 0.746 ± 0.262 | 72.6% (698/962) |
| Lshape3d_env1 | `./Experiments/3dshape_metric/Lshape3d_env1_09_06_00_21/latest.pt` | 4000 | 4.1156e-03 | 2147 | 83.4% (834/1000) | 0.090 ± 0.035 | 0.752 ± 0.273 | 73.9% (715/967) |
| Fshape3d_env1 | `./Experiments/3dshape_metric/Fshape3d_env1_09_06_00_21/latest.pt` | 4000 | 4.1785e-03 | 2180 | 83.2% (832/1000) | 0.087 ± 0.033 | 0.749 ± 0.270 | 75.0% (724/965) |
| Ashape3d_env1 | `./Experiments/3dshape_metric/Ashape3d_env1_09_06_00_21/latest.pt` | 4000 | 4.3874e-03 | 2182 | 81.9% (819/1000) | 0.090 ± 0.034 | 0.735 ± 0.267 | 76.5% (720/941) |
| Vshape3d_env1 | `./Experiments/3dshape_metric/Vshape3d_env1_09_06_00_21/latest.pt` | 4000 | 4.5240e-03 | 2192 | 78.9% (789/1000) | 0.091 ± 0.035 | 0.739 ± 0.267 | 71.6% (677/945) |
| 4shape3d_env1 | `./Experiments/3dshape_metric/4shape3d_env1_09_06_00_21/latest.pt` | 4000 | 4.1265e-03 | 2188 | 78.8% (788/1000) | 0.090 ± 0.034 | 0.752 ± 0.270 | 74.5% (713/957) |
| rectangle_env2 | `./Experiments/3dshape_metric/rectangle_env2_09_06_00_21/latest.pt` | 4000 | 3.6980e-03 | 2159 | 82.7% (827/1000) | 0.084 ± 0.033 | 0.709 ± 0.262 | 74.0% (716/967) |
| Lshape3d_env2 | `./Experiments/3dshape_metric/Lshape3d_env2_09_06_00_56/latest.pt` | 4000 | 4.4142e-03 | 2158 | 81.6% (814/998) | 0.095 ± 0.038 | 0.739 ± 0.270 | 73.8% (711/963) |
| Fshape3d_env2 | `./Experiments/3dshape_metric/Fshape3d_env2_09_06_00_57/latest.pt` | 4000 | 4.1649e-03 | 2163 | 85.2% (852/1000) | 0.095 ± 0.038 | 0.734 ± 0.273 | 74.4% (709/953) |
| Ashape3d_env2 | `./Experiments/3dshape_metric/Ashape3d_env2_09_06_00_57/latest.pt` | 4000 | 4.6322e-03 | 2165 | 74.3% (743/1000) | 0.082 ± 0.030 | 0.709 ± 0.244 | 62.9% (594/945) |
| Vshape3d_env2 | `./Experiments/3dshape_metric/Vshape3d_env2_09_06_00_57/latest.pt` | 4000 | 4.6209e-03 | 2162 | 73.7% (736/999) | 0.083 ± 0.030 | 0.728 ± 0.261 | 65.0% (610/939) |
| 4shape3d_env2 | `./Experiments/3dshape_metric/4shape3d_env2_09_06_00_57/latest.pt` | 4000 | 4.3324e-03 | 2180 | 78.7% (787/1000) | 0.095 ± 0.039 | 0.721 ± 0.263 | 69.7% (655/940) |
| rectangle_env3 | `./Experiments/3dshape_metric/rectangle_env3_09_06_00_57/latest.pt` | 4000 | 3.6257e-03 | 2155 | 90.7% (907/1000) | 0.091 ± 0.036 | 0.664 ± 0.238 | 83.7% (799/955) |
| Lshape3d_env3 | `./Experiments/3dshape_metric/Lshape3d_env3_09_06_01_32/latest.pt` | 4000 | 4.0311e-03 | 2220 | 89.2% (891/999) | 0.079 ± 0.033 | 0.653 ± 0.250 | 79.7% (754/946) |
| Fshape3d_env3 | `./Experiments/3dshape_metric/Fshape3d_env3_09_06_01_33/latest.pt` | 4000 | 4.0209e-03 | 2214 | 92.6% (925/999) | 0.077 ± 0.031 | 0.648 ± 0.244 | 82.9% (781/942) |
| Ashape3d_env3 | `./Experiments/3dshape_metric/Ashape3d_env3_09_06_01_33/latest.pt` | 4000 | 4.0924e-03 | 2208 | 93.2% (931/999) | 0.090 ± 0.038 | 0.651 ± 0.247 | 86.2% (797/925) |
| Vshape3d_env3 | `./Experiments/3dshape_metric/Vshape3d_env3_09_06_01_33/latest.pt` | 4000 | 4.2454e-03 | 2218 | 91.7% (916/999) | 0.089 ± 0.037 | 0.656 ± 0.246 | 83.5% (771/923) |
| 4shape3d_env3 | `./Experiments/3dshape_metric/4shape3d_env3_09_06_01_33/latest.pt` | 4000 | 4.3087e-03 | 2221 | 93.1% (930/999) | 0.079 ± 0.031 | 0.653 ± 0.240 | 83.1% (781/940) |
| rectangle_env4 | `./Experiments/3dshape_metric/rectangle_env4_09_06_01_34/latest.pt` | 4000 | 4.0877e-03 | 2260 | 85.9% (859/1000) | 0.080 ± 0.031 | 0.663 ± 0.238 | 79.9% (762/954) |
| Lshape3d_env4 | `./Experiments/3dshape_metric/Lshape3d_env4_09_06_02_10/latest.pt` | 4000 | 4.1917e-03 | 2198 | 89.4% (893/999) | 0.089 ± 0.036 | 0.655 ± 0.229 | 80.8% (759/939) |
| Fshape3d_env4 | `./Experiments/3dshape_metric/Fshape3d_env4_09_06_02_10/latest.pt` | 4000 | 4.1212e-03 | 2185 | 89.6% (895/999) | 0.089 ± 0.034 | 0.674 ± 0.233 | 80.3% (752/936) |
| Ashape3d_env4 | `./Experiments/3dshape_metric/Ashape3d_env4_09_06_02_10/latest.pt` | 4000 | 4.6228e-03 | 2184 | 87.9% (878/999) | 0.077 ± 0.031 | 0.657 ± 0.246 | 79.1% (726/918) |
| Vshape3d_env4 | `./Experiments/3dshape_metric/Vshape3d_env4_09_06_02_10/latest.pt` | 4000 | 4.3933e-03 | 2162 | 90.6% (906/1000) | 0.077 ± 0.030 | 0.653 ± 0.242 | 79.5% (737/927) |
| 4shape3d_env4 | `./Experiments/3dshape_metric/4shape3d_env4_09_06_02_10/latest.pt` | 4000 | 4.2272e-03 | 2183 | 88.1% (881/1000) | 0.091 ± 0.035 | 0.659 ± 0.221 | 77.9% (732/940) |
| Tshape3d_env4 | `./Experiments/3dshape_metric/Tshape3d_env4_09_06_02_11/latest.pt` | 4000 | 2.6893e-03 | 2153 | 90.4% (904/1000) | 0.093 ± 0.047 | 0.676 ± 0.315 | 82.1% (821/1000) |
| Tshape3d_env1 | `./Experiments/3dshape_metric/Tshape3d_env1_09_06_02_46/latest.pt` | 4000 | 4.4010e-03 | 1114 | 71.6% (710/992) | 0.091 ± 0.035 | 0.778 ± 0.284 | 66.0% (636/963) |
| Lcouch_Corozal | `./Experiments/3dshape_metric/Lcouch_Corozal_09_06_02_46/latest.pt` | 4000 | 3.5759e-03 | 1113 | 43.1% (373/865) | 0.079 ± 0.038 | 0.633 ± 0.285 | 35.8% (234/654) |
