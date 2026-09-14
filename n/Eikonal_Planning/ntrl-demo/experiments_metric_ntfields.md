# Experiments -- `models/metric` on the 3-D shape task

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

## 3-D section &mdash; SE(3), `env1`-`env4`

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
| Tshape3d_env1 | `./Experiments/3dshape_metric/Tshape3d_env1_09_06_02_46/latest.pt` | 4000 | 4.4010e-03 | 1114 | 71.6% (710/992) | 0.091 ± 0.035 | 0.778 ± 0.284 | 66.0% (636/963) |
| Tshape3d_env2 | `./Experiments/3dshape_metric/Tshape3d_env2_09_12_13_53/latest.pt` | 4000 | 4.5381e-03 | 2312 | -- | -- | -- | -- |
| Tshape3d_env3 | `./Experiments/3dshape_metric/Tshape3d_env3_09_12_13_53/latest.pt` | 4000 | 4.0769e-03 | 2326 | -- | -- | -- | -- |
| Tshape3d_env4 | `./Experiments/3dshape_metric/Tshape3d_env4_09_12_13_53/latest.pt` | 4000 | 4.9179e-03 | 2299 | -- | -- | -- | -- |
| Lcouch_Corozal | `./Experiments/3dshape_metric/Lcouch_Corozal_09_06_02_46/latest.pt` | 4000 | 3.5759e-03 | 1113 | 43.1% (373/865) | 0.079 ± 0.038 | 0.633 ± 0.285 | 35.8% (234/654) |

## 2-D section &mdash; SE(2), `2denv1`-`2denv4` x 7 shapes

All four planar mazes, all seven shapes: 28 cells, trained by
`train/train_3dshape_metric.py` on the same shared
`../../ntrl-demo/ntrl-demo/datasets/3dshape/<cell>` dirs the 3-D rows use.

The planning columns for these rows come from `tests/3d_plan.py --2d`, which
masks the MPPI displacement to (x, y, rz) before the magnitude clamp so z /
rx / ry hold their start value of 0 for the whole rollout.  Without it the
rollout leaves the planar slice the field was fit on.  The flag is redundant
with the `"two_d": true` each 2-D `meta.json` carries; it is passed anyway so
the mode is explicit in every log and in `summary.json` (`planar_rollout`).

`Tshape3d_2denv4` was trained under the pre-env-tag name `Tshape3d_env4`
(run `Tshape3d_2denv4_09_06_02_11`, renamed 2026-09-12); the `Tshape3d_env4`
row of the 3-D table is the SE(3) cell trained after the rename.

| Env | Model | Epochs | Final Loss | Train Time (s) | Valid | Plan Time (s) | Path Length | Valid (tight) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv1 | `./Experiments/3dshape_metric/rectangle_2denv1_09_10_09_05/latest.pt` | 4000 | 2.3626e-03 | 1772 | -- | -- | -- | -- |
| Lshape3d_2denv1 | `./Experiments/3dshape_metric/Lshape3d_2denv1_09_10_09_05/latest.pt` | 4000 | 2.6950e-03 | 1782 | -- | -- | -- | -- |
| Fshape3d_2denv1 | `./Experiments/3dshape_metric/Fshape3d_2denv1_09_10_09_05/latest.pt` | 4000 | 2.6772e-03 | 1792 | -- | -- | -- | -- |
| Ashape3d_2denv1 | `./Experiments/3dshape_metric/Ashape3d_2denv1_09_10_09_05/latest.pt` | 4000 | 2.5691e-03 | 1760 | -- | -- | -- | -- |
| Vshape3d_2denv1 | `./Experiments/3dshape_metric/Vshape3d_2denv1_09_10_09_05/latest.pt` | 4000 | 3.0825e-03 | 1769 | -- | -- | -- | -- |
| 4shape3d_2denv1 | `./Experiments/3dshape_metric/4shape3d_2denv1_09_10_09_05/latest.pt` | 4000 | 2.7720e-03 | 1771 | -- | -- | -- | -- |
| Tshape3d_2denv1 | `./Experiments/3dshape_metric/Tshape3d_2denv1_09_10_09_34/latest.pt` | 4000 | 2.9753e-03 | 1823 | -- | -- | -- | -- |
| rectangle_2denv2 | `./Experiments/3dshape_metric/rectangle_2denv2_09_10_09_35/latest.pt` | 4000 | 2.7842e-03 | 1848 | -- | -- | -- | -- |
| Lshape3d_2denv2 | `./Experiments/3dshape_metric/Lshape3d_2denv2_09_10_09_35/latest.pt` | 4000 | 2.8746e-03 | 1888 | -- | -- | -- | -- |
| Fshape3d_2denv2 | `./Experiments/3dshape_metric/Fshape3d_2denv2_09_10_09_35/latest.pt` | 4000 | 2.8882e-03 | 1883 | -- | -- | -- | -- |
| Ashape3d_2denv2 | `./Experiments/3dshape_metric/Ashape3d_2denv2_09_10_09_35/latest.pt` | 4000 | 2.5235e-03 | 1830 | -- | -- | -- | -- |
| Vshape3d_2denv2 | `./Experiments/3dshape_metric/Vshape3d_2denv2_09_10_09_35/latest.pt` | 4000 | 2.4958e-03 | 1808 | -- | -- | -- | -- |
| 4shape3d_2denv2 | `./Experiments/3dshape_metric/4shape3d_2denv2_09_10_10_05/latest.pt` | 4000 | 2.8584e-03 | 1776 | -- | -- | -- | -- |
| Tshape3d_2denv2 | `./Experiments/3dshape_metric/Tshape3d_2denv2_09_10_10_05/latest.pt` | 4000 | 2.5681e-03 | 1786 | -- | -- | -- | -- |
| rectangle_2denv3 | `./Experiments/3dshape_metric/rectangle_2denv3_09_10_10_05/latest.pt` | 4000 | 1.6422e-03 | 1854 | -- | -- | -- | -- |
| Lshape3d_2denv3 | `./Experiments/3dshape_metric/Lshape3d_2denv3_09_10_10_05/latest.pt` | 4000 | 1.9953e-03 | 1850 | -- | -- | -- | -- |
| Fshape3d_2denv3 | `./Experiments/3dshape_metric/Fshape3d_2denv3_09_10_10_06/latest.pt` | 4000 | 2.1743e-03 | 1801 | -- | -- | -- | -- |
| Ashape3d_2denv3 | `./Experiments/3dshape_metric/Ashape3d_2denv3_09_10_10_06/latest.pt` | 4000 | 2.2252e-03 | 1825 | -- | -- | -- | -- |
| Vshape3d_2denv3 | `./Experiments/3dshape_metric/Vshape3d_2denv3_09_10_10_35/latest.pt` | 4000 | 2.1388e-03 | 1759 | -- | -- | -- | -- |
| 4shape3d_2denv3 | `./Experiments/3dshape_metric/4shape3d_2denv3_09_10_10_35/latest.pt` | 4000 | 2.0625e-03 | 1815 | -- | -- | -- | -- |
| Tshape3d_2denv3 | `./Experiments/3dshape_metric/Tshape3d_2denv3_09_10_10_36/latest.pt` | 4000 | 2.8756e-03 | 1802 | -- | -- | -- | -- |
| rectangle_2denv4 | `./Experiments/3dshape_metric/rectangle_2denv4_09_10_10_36/latest.pt` | 4000 | 1.9354e-03 | 1841 | -- | -- | -- | -- |
| Lshape3d_2denv4 | `./Experiments/3dshape_metric/Lshape3d_2denv4_09_10_10_36/latest.pt` | 4000 | 2.1426e-03 | 1801 | -- | -- | -- | -- |
| Fshape3d_2denv4 | `./Experiments/3dshape_metric/Fshape3d_2denv4_09_10_10_37/latest.pt` | 4000 | 2.0501e-03 | 1795 | -- | -- | -- | -- |
| Ashape3d_2denv4 | `./Experiments/3dshape_metric/Ashape3d_2denv4_09_10_11_04/latest.pt` | 4000 | 2.1601e-03 | 1179 | -- | -- | -- | -- |
| Vshape3d_2denv4 | `./Experiments/3dshape_metric/Vshape3d_2denv4_09_10_11_05/latest.pt` | 4000 | 2.5261e-03 | 1129 | -- | -- | -- | -- |
| 4shape3d_2denv4 | `./Experiments/3dshape_metric/4shape3d_2denv4_09_10_11_06/latest.pt` | 4000 | 2.0302e-03 | 1077 | -- | -- | -- | -- |
| Tshape3d_2denv4 | `./Experiments/3dshape_metric/Tshape3d_2denv4_09_06_02_11/latest.pt` | 4000 | 2.6893e-03 | 2153 | 90.4% (904/1000) | 0.093 ± 0.047 | 0.676 ± 0.315 | 82.1% (821/1000) |

## 3D_1k_valid

The same SE(3) cells as the 3-D section above -- same checkpoints, same
`tests/3d_plan.py`, same controller settings, same 1000 cases -- scored on
`../../ntrl-demo/ntrl-demo/testing_data_1k_complete/3dshape/<cell>` instead
of `testing_data/`.  Every pair in that set is collision-free *and* carries
an RRT-Connect path found within 180 s, so unlike the table above it has a
demonstrated 100% ceiling: a failure here is the planner's, not a
start/goal pair that nothing could solve.

Because the unsolvable pairs are gone, these numbers are **not** comparable
row-for-row with the 3-D section -- the two are different test populations,
not the same test measured twice.  See `testing_data_1k_complete/README.md`
for how the set was built and what fraction of sampled pairs it discarded.
`Lcouch_Corozal` has no 1k set and is absent.  `Tshape3d_env2..env4` were
trained 2026-09-12 on the SE(3) `datasets/3dshape/Tshape3d_env*` sets and
have no `testing_data/` run, only this one.

`Valid`, `Plan Time` and `Path Length` are the same three measurements as
the 3-D section (endpoint screening, then success over the pairs left;
rollout wall clock and Euclidean 6-D length over the valid paths).
`Trans Length` and `Rot Length` split the same polyline the way
`experiments_rrt_connect.md` does: translation travelled (`sum ||dt||`,
normalized frame) and rotation travelled as the geodesic angle in radians
(`sum 2*acos|q.q'|`).  Unlike `Path Length`, these two are directly
comparable with the NTFields and RRT-Connect tables.  These runs went four
to two GPUs, so read the timings as amortized.  `test_cases` is the number
of pairs left after screening.

Produced by `tests/run_3d_plan_1k.sh`; summaries in `results/3d_plan_metric_1k/<cell>/summary.json`.

| Env | Model | Valid | Plan Time (s) | Path Length | Trans Length | Rot Length (rad) | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./Experiments/3dshape_metric/rectangle_env1_09_05_23_44/latest.pt` | 82.0% (820/1000) | 0.096 ± 0.041 | 0.724 ± 0.273 | 0.534 ± 0.227 | 2.570 ± 1.060 | 1000 |
| Lshape3d_env1 | `./Experiments/3dshape_metric/Lshape3d_env1_09_06_00_21/latest.pt` | 84.3% (843/1000) | 0.089 ± 0.036 | 0.737 ± 0.274 | 0.546 ± 0.232 | 2.609 ± 1.047 | 1000 |
| Fshape3d_env1 | `./Experiments/3dshape_metric/Fshape3d_env1_09_06_00_21/latest.pt` | 83.2% (832/1000) | 0.096 ± 0.036 | 0.744 ± 0.259 | 0.556 ± 0.223 | 2.607 ± 1.049 | 1000 |
| Ashape3d_env1 | `./Experiments/3dshape_metric/Ashape3d_env1_09_06_00_21/latest.pt` | 84.3% (843/1000) | 0.094 ± 0.035 | 0.749 ± 0.266 | 0.541 ± 0.218 | 2.746 ± 1.084 | 1000 |
| Vshape3d_env1 | `./Experiments/3dshape_metric/Vshape3d_env1_09_06_00_21/latest.pt` | 80.3% (803/1000) | 0.093 ± 0.034 | 0.732 ± 0.258 | 0.533 ± 0.215 | 2.667 ± 1.052 | 1000 |
| 4shape3d_env1 | `./Experiments/3dshape_metric/4shape3d_env1_09_06_00_21/latest.pt` | 77.8% (778/1000) | 0.090 ± 0.033 | 0.732 ± 0.257 | 0.542 ± 0.220 | 2.602 ± 1.022 | 1000 |
| rectangle_env2 | `./Experiments/3dshape_metric/rectangle_env2_09_06_00_21/latest.pt` | 82.5% (825/1000) | 0.095 ± 0.039 | 0.710 ± 0.261 | 0.530 ± 0.228 | 2.479 ± 1.026 | 1000 |
| Lshape3d_env2 | `./Experiments/3dshape_metric/Lshape3d_env2_09_06_00_56/latest.pt` | 81.7% (817/1000) | 0.090 ± 0.035 | 0.716 ± 0.256 | 0.522 ± 0.221 | 2.588 ± 1.015 | 1000 |
| Fshape3d_env2 | `./Experiments/3dshape_metric/Fshape3d_env2_09_06_00_57/latest.pt` | 85.9% (859/1000) | 0.096 ± 0.037 | 0.741 ± 0.271 | 0.544 ± 0.233 | 2.653 ± 1.078 | 1000 |
| Ashape3d_env2 | `./Experiments/3dshape_metric/Ashape3d_env2_09_06_00_57/latest.pt` | 71.8% (718/1000) | 0.093 ± 0.033 | 0.736 ± 0.252 | 0.531 ± 0.211 | 2.728 ± 1.025 | 1000 |
| Vshape3d_env2 | `./Experiments/3dshape_metric/Vshape3d_env2_09_06_00_57/latest.pt` | 73.8% (738/1000) | 0.094 ± 0.035 | 0.734 ± 0.259 | 0.532 ± 0.211 | 2.709 ± 1.076 | 1000 |
| 4shape3d_env2 | `./Experiments/3dshape_metric/4shape3d_env2_09_06_00_57/latest.pt` | 79.6% (796/1000) | 0.097 ± 0.037 | 0.738 ± 0.263 | 0.545 ± 0.229 | 2.632 ± 1.063 | 1000 |
| rectangle_env3 | `./Experiments/3dshape_metric/rectangle_env3_09_06_00_57/latest.pt` | 90.5% (905/1000) | 0.095 ± 0.035 | 0.674 ± 0.233 | 0.495 ± 0.217 | 2.405 ± 0.970 | 1000 |
| Lshape3d_env3 | `./Experiments/3dshape_metric/Lshape3d_env3_09_06_01_32/latest.pt` | 88.9% (889/1000) | 0.090 ± 0.035 | 0.663 ± 0.240 | 0.480 ± 0.209 | 2.411 ± 1.030 | 1000 |
| Fshape3d_env3 | `./Experiments/3dshape_metric/Fshape3d_env3_09_06_01_33/latest.pt` | 90.6% (906/1000) | 0.091 ± 0.035 | 0.664 ± 0.245 | 0.485 ± 0.218 | 2.391 ± 1.011 | 1000 |
| Ashape3d_env3 | `./Experiments/3dshape_metric/Ashape3d_env3_09_06_01_33/latest.pt` | 92.4% (924/1000) | 0.087 ± 0.034 | 0.650 ± 0.243 | 0.466 ± 0.202 | 2.391 ± 1.043 | 1000 |
| Vshape3d_env3 | `./Experiments/3dshape_metric/Vshape3d_env3_09_06_01_33/latest.pt` | 90.8% (908/1000) | 0.088 ± 0.033 | 0.641 ± 0.233 | 0.461 ± 0.197 | 2.368 ± 1.017 | 1000 |
| 4shape3d_env3 | `./Experiments/3dshape_metric/4shape3d_env3_09_06_01_33/latest.pt` | 95.4% (953/999) | 0.091 ± 0.035 | 0.665 ± 0.244 | 0.487 ± 0.216 | 2.374 ± 1.009 | 999 |
| rectangle_env4 | `./Experiments/3dshape_metric/rectangle_env4_09_06_01_34/latest.pt` | 87.5% (875/1000) | 0.090 ± 0.035 | 0.652 ± 0.230 | 0.475 ± 0.198 | 2.371 ± 0.999 | 1000 |
| Lshape3d_env4 | `./Experiments/3dshape_metric/Lshape3d_env4_09_06_02_10/latest.pt` | 89.1% (891/1000) | 0.085 ± 0.033 | 0.655 ± 0.230 | 0.461 ± 0.189 | 2.468 ± 1.008 | 1000 |
| Fshape3d_env4 | `./Experiments/3dshape_metric/Fshape3d_env4_09_06_02_10/latest.pt` | 91.8% (918/1000) | 0.090 ± 0.036 | 0.657 ± 0.240 | 0.462 ± 0.192 | 2.477 ± 1.045 | 1000 |
| Ashape3d_env4 | `./Experiments/3dshape_metric/Ashape3d_env4_09_06_02_10/latest.pt` | 87.8% (878/1000) | 0.088 ± 0.033 | 0.658 ± 0.235 | 0.452 ± 0.194 | 2.534 ± 1.040 | 1000 |
| Vshape3d_env4 | `./Experiments/3dshape_metric/Vshape3d_env4_09_06_02_10/latest.pt` | 89.1% (891/1000) | 0.088 ± 0.033 | 0.664 ± 0.240 | 0.458 ± 0.194 | 2.556 ± 1.059 | 1000 |
| 4shape3d_env4 | `./Experiments/3dshape_metric/4shape3d_env4_09_06_02_10/latest.pt` | 90.9% (909/1000) | 0.086 ± 0.033 | 0.657 ± 0.237 | 0.463 ± 0.197 | 2.463 ± 1.031 | 1000 |
| Tshape3d_env1 | `./Experiments/3dshape_metric/Tshape3d_env1_09_06_02_46/latest.pt` | 78.7% (787/1000) | 0.091 ± 0.035 | 0.729 ± 0.256 | 0.533 ± 0.208 | 2.658 ± 1.077 | 1000 |
| Tshape3d_env2 | `./Experiments/3dshape_metric/Tshape3d_env2_09_12_13_53/latest.pt` | 67.7% (677/1000) | 0.141 ± 0.058 | 0.719 ± 0.250 | 0.522 ± 0.214 | 2.638 ± 1.012 | 1000 |
| Tshape3d_env3 | `./Experiments/3dshape_metric/Tshape3d_env3_09_12_13_53/latest.pt` | 92.7% (927/1000) | 0.140 ± 0.079 | 0.659 ± 0.251 | 0.473 ± 0.211 | 2.417 ± 1.066 | 1000 |
| Tshape3d_env4 | `./Experiments/3dshape_metric/Tshape3d_env4_09_12_13_53/latest.pt` | 88.4% (884/1000) | 0.140 ± 0.055 | 0.645 ± 0.220 | 0.447 ± 0.184 | 2.474 ± 0.989 | 1000 |

## 2d_1k_valid

The same 28 planar cells as the 2-D section above -- same checkpoints, same
`tests/3d_plan.py --2d`, same 1000 cases -- scored on the
`testing_data_1k_complete/` sets.  The filter matters far more here than in
3-D: it discarded 46% of sampled pairs in `Tshape3d_2denv2`, 36% in
`Ashape3d_2denv2` and 30% in `Vshape3d_2denv2`, all of them the hardest
pairs in those cells, so expect those rows to read substantially higher
than above and do **not** compare them row-for-row.  Per-cell discard rates
are in `testing_data_1k_complete/README.md`.

`Tshape3d_2denv4` is planned from run `Tshape3d_2denv4_09_06_02_11`, trained
under the pre-env-tag name `Tshape3d_env4` and renamed 2026-09-12.

| Env | Model | Valid | Plan Time (s) | Path Length | Trans Length | Rot Length (rad) | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv1 | `./Experiments/3dshape_metric/rectangle_2denv1_09_10_09_05/latest.pt` | 91.8% (918/1000) | 0.077 ± 0.037 | 0.687 ± 0.333 | 0.514 ± 0.322 | 2.286 ± 1.407 | 1000 |
| Lshape3d_2denv1 | `./Experiments/3dshape_metric/Lshape3d_2denv1_09_10_09_05/latest.pt` | 91.8% (918/1000) | 0.074 ± 0.040 | 0.646 ± 0.348 | 0.459 ± 0.306 | 2.337 ± 1.466 | 1000 |
| Fshape3d_2denv1 | `./Experiments/3dshape_metric/Fshape3d_2denv1_09_10_09_05/latest.pt` | 91.4% (914/1000) | 0.073 ± 0.039 | 0.637 ± 0.346 | 0.451 ± 0.307 | 2.302 ± 1.476 | 1000 |
| Ashape3d_2denv1 | `./Experiments/3dshape_metric/Ashape3d_2denv1_09_10_09_05/latest.pt` | 86.5% (865/1000) | 0.062 ± 0.031 | 0.548 ± 0.267 | 0.358 ± 0.217 | 2.182 ± 1.425 | 1000 |
| Vshape3d_2denv1 | `./Experiments/3dshape_metric/Vshape3d_2denv1_09_10_09_05/latest.pt` | 88.0% (880/1000) | 0.062 ± 0.031 | 0.558 ± 0.279 | 0.362 ± 0.221 | 2.242 ± 1.456 | 1000 |
| 4shape3d_2denv1 | `./Experiments/3dshape_metric/4shape3d_2denv1_09_10_09_05/latest.pt` | 88.8% (888/1000) | 0.065 ± 0.036 | 0.620 ± 0.346 | 0.439 ± 0.300 | 2.251 ± 1.517 | 1000 |
| Tshape3d_2denv1 | `./Experiments/3dshape_metric/Tshape3d_2denv1_09_10_09_34/latest.pt` | 84.9% (849/1000) | 0.059 ± 0.030 | 0.518 ± 0.262 | 0.335 ± 0.204 | 2.090 ± 1.372 | 1000 |
| rectangle_2denv2 | `./Experiments/3dshape_metric/rectangle_2denv2_09_10_09_35/latest.pt` | 93.1% (931/1000) | 0.082 ± 0.037 | 0.769 ± 0.342 | 0.556 ± 0.296 | 2.700 ± 1.415 | 1000 |
| Lshape3d_2denv2 | `./Experiments/3dshape_metric/Lshape3d_2denv2_09_10_09_35/latest.pt` | 83.7% (837/1000) | 0.091 ± 0.050 | 0.762 ± 0.420 | 0.499 ± 0.306 | 3.008 ± 1.805 | 1000 |
| Fshape3d_2denv2 | `./Experiments/3dshape_metric/Fshape3d_2denv2_09_10_09_35/latest.pt` | 75.8% (758/1000) | 0.083 ± 0.046 | 0.696 ± 0.381 | 0.454 ± 0.298 | 2.752 ± 1.639 | 1000 |
| Ashape3d_2denv2 | `./Experiments/3dshape_metric/Ashape3d_2denv2_09_10_09_35/latest.pt` | 67.2% (672/1000) | 0.059 ± 0.035 | 0.478 ± 0.283 | 0.246 ± 0.175 | 2.283 ± 1.526 | 1000 |
| Vshape3d_2denv2 | `./Experiments/3dshape_metric/Vshape3d_2denv2_09_10_09_35/latest.pt` | 68.3% (683/1000) | 0.060 ± 0.035 | 0.509 ± 0.294 | 0.267 ± 0.185 | 2.402 ± 1.551 | 1000 |
| 4shape3d_2denv2 | `./Experiments/3dshape_metric/4shape3d_2denv2_09_10_10_05/latest.pt` | 69.6% (696/1000) | 0.082 ± 0.043 | 0.640 ± 0.345 | 0.408 ± 0.274 | 2.583 ± 1.512 | 1000 |
| Tshape3d_2denv2 | `./Experiments/3dshape_metric/Tshape3d_2denv2_09_10_10_05/latest.pt` | 73.8% (738/1000) | 0.054 ± 0.032 | 0.421 ± 0.245 | 0.202 ± 0.125 | 2.095 ± 1.470 | 1000 |
| rectangle_2denv3 | `./Experiments/3dshape_metric/rectangle_2denv3_09_10_10_05/latest.pt` | 95.1% (951/1000) | 0.087 ± 0.034 | 0.627 ± 0.241 | 0.468 ± 0.220 | 2.127 ± 1.435 | 1000 |
| Lshape3d_2denv3 | `./Experiments/3dshape_metric/Lshape3d_2denv3_09_10_10_05/latest.pt` | 97.9% (979/1000) | 0.077 ± 0.030 | 0.620 ± 0.241 | 0.461 ± 0.219 | 2.126 ± 1.440 | 1000 |
| Fshape3d_2denv3 | `./Experiments/3dshape_metric/Fshape3d_2denv3_09_10_10_06/latest.pt` | 96.6% (966/1000) | 0.085 ± 0.034 | 0.618 ± 0.243 | 0.460 ± 0.221 | 2.116 ± 1.445 | 1000 |
| Ashape3d_2denv3 | `./Experiments/3dshape_metric/Ashape3d_2denv3_09_10_10_06/latest.pt` | 98.4% (984/1000) | 0.083 ± 0.033 | 0.608 ± 0.238 | 0.450 ± 0.212 | 2.115 ± 1.434 | 1000 |
| Vshape3d_2denv3 | `./Experiments/3dshape_metric/Vshape3d_2denv3_09_10_10_35/latest.pt` | 99.0% (990/1000) | 0.080 ± 0.032 | 0.612 ± 0.243 | 0.453 ± 0.217 | 2.121 ± 1.432 | 1000 |
| 4shape3d_2denv3 | `./Experiments/3dshape_metric/4shape3d_2denv3_09_10_10_35/latest.pt` | 96.6% (966/1000) | 0.081 ± 0.032 | 0.612 ± 0.239 | 0.449 ± 0.219 | 2.146 ± 1.470 | 1000 |
| Tshape3d_2denv3 | `./Experiments/3dshape_metric/Tshape3d_2denv3_09_10_10_36/latest.pt` | 98.6% (986/1000) | 0.082 ± 0.033 | 0.602 ± 0.243 | 0.441 ± 0.215 | 2.123 ± 1.436 | 1000 |
| rectangle_2denv4 | `./Experiments/3dshape_metric/rectangle_2denv4_09_10_10_36/latest.pt` | 95.2% (952/1000) | 0.085 ± 0.035 | 0.624 ± 0.255 | 0.463 ± 0.247 | 2.120 ± 1.418 | 1000 |
| Lshape3d_2denv4 | `./Experiments/3dshape_metric/Lshape3d_2denv4_09_10_10_36/latest.pt` | 98.5% (985/1000) | 0.083 ± 0.035 | 0.638 ± 0.272 | 0.477 ± 0.264 | 2.146 ± 1.440 | 1000 |
| Fshape3d_2denv4 | `./Experiments/3dshape_metric/Fshape3d_2denv4_09_10_10_37/latest.pt` | 97.1% (971/1000) | 0.082 ± 0.035 | 0.629 ± 0.273 | 0.469 ± 0.262 | 2.116 ± 1.458 | 1000 |
| Ashape3d_2denv4 | `./Experiments/3dshape_metric/Ashape3d_2denv4_09_10_11_04/latest.pt` | 99.1% (991/1000) | 0.088 ± 0.041 | 0.653 ± 0.302 | 0.486 ± 0.295 | 2.201 ± 1.486 | 1000 |
| Vshape3d_2denv4 | `./Experiments/3dshape_metric/Vshape3d_2denv4_09_10_11_05/latest.pt` | 99.2% (992/1000) | 0.084 ± 0.039 | 0.654 ± 0.303 | 0.489 ± 0.297 | 2.183 ± 1.470 | 1000 |
| 4shape3d_2denv4 | `./Experiments/3dshape_metric/4shape3d_2denv4_09_10_11_06/latest.pt` | 98.4% (984/1000) | 0.080 ± 0.035 | 0.631 ± 0.281 | 0.473 ± 0.276 | 2.111 ± 1.468 | 1000 |
| Tshape3d_2denv4 | `./Experiments/3dshape_metric/Tshape3d_2denv4_09_06_02_11/latest.pt` | 97.4% (974/1000) | 0.082 ± 0.040 | 0.640 ± 0.309 | 0.469 ± 0.300 | 2.187 ± 1.424 | 1000 |
