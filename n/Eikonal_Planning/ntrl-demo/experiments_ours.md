# Experiments

> Collected at the repository root. Every unqualified path below (`datasets/`, `Experiments/`, `outputs/`, `results/`, `tests/`, `train/`, ...)
> is relative to `ntrl-demo/ntrl-demo/`, where this table's runs live. Regenerate with `_make_experiments_2d.py --out ../../experiments_ours.md` from that directory.

## 3-D shape task

Seven shapes across four SE(3) environments (28 cells), plus the legacy planar
(`Tshape3d`, `2denv4`) row at the bottom, which sat in this table as
`Tshape3d_env4 (2D)` until the 3-D `Tshape3d_env4` cell took that name on
2026-09-10 (its dirs are now `*/Tshape3d_2denv4_legacy`; it is superseded by
the `2denv4` sweep below).  The Gibson cell `Lcouch_Corozal` has its own
"Gibson" section further down.

Every row is 800k training pairs (`--margin 0.05 --offset 0.001`), **10000
epochs**, scored on 1000 held-out pairs at `--offset 0.02`.  The `Tshape3d_env*`
rows were added 2026-09-11 by `_run_3d_tshape.sh` (README: "Running the 3-D
T-shape cells"); their datasets were sampled at `--batch_size 250` instead of
2000, which only changes how the rejection sampler is chunked.  The earlier
rows predate the `--name` flag and keep their `3dshape_<timestamp>` run folders.
**`Tshape3d_env3` is the exception on epochs**: its `latest.pt` is the epoch-5500
model (`Model_Epoch_05500_*.pt` in the same folder), evaluated mid-run; it already
sat inside the band of the finished env3 rows, so the run was stopped at epoch 7500
rather than taken to 10000.

`SR Alternate Bellman Horizon SD` is the hlB planner with the executed MPPI step
biased along the least-squares gradient of the sampled speeds
(`_spread_planner.py --steer-bias 0.5 --no-gate`, driver `_run_spread_sd.sh`,
1000 cases each).  This script reads that column from `results/spread_sd/` itself;
do not chain `_make_spread_sd.py` after it -- that rewrites the SD column of every
table in the file, including `3D_1k_valid` (SD from `spread_sd_1k/`) and `3D_tight`
(SD from `spread_sd_tight/`).  Refresh
this whole section with `_make_experiments_3d.py`.

| Env | Model | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | SR Alternate Bellman Horizon SD | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./Experiments/3dshape/3dshape_08_06_17_06/latest.pt` | 97.6% | 96.8% | 99.9% | 97.2% | 98.9% | 98.9% | 98.5% | 1000 |
| Lshape3d_env1 | `./Experiments/3dshape/3dshape_08_15_20_19/latest.pt` | 88.9% | 89.8% | 97.5% | 89.8% | 94.7% | 95.2% | 98.2% | 1000 |
| Fshape3d_env1 | `./Experiments/3dshape/3dshape_08_16_11_35/latest.pt` | 92.5% | 92.3% | 98.7% | 95.5% | 97.9% | 98.3% | 99.3% | 1000 |
| Ashape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_49/latest.pt` | 78.8% | 77.9% | 92.6% | 82.9% | 89.1% | 91.0% | 96.6% | 1000 |
| Vshape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_50/latest.pt` | 81.5% | 80.1% | 93.7% | 85.1% | 91.0% | 90.3% | 96.4% | 1000 |
| 4shape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_54/latest.pt` | 89.6% | 89.3% | 96.9% | 90.8% | 94.1% | 95.2% | 96.6% | 1000 |
| Tshape3d_env1 | `./Experiments/3dshape/Tshape3d_env1/latest.pt` | 84.1% | 82.5% | 94.8% | 86.7% | 91.3% | 92.8% | 97.6% | 1000 |
| rectangle_env2 | `./Experiments/3dshape/3dshape_08_29_15_41/latest.pt` | 95.8% | 93.9% | 99.3% | 96.6% | 98.0% | 98.0% | 98.7% | 1000 |
| Lshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_04/latest.pt` | 79.8% | 80.6% | 93.5% | 83.0% | 87.5% | 86.3% | 94.1% | 998 |
| Fshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_05/latest.pt` | 80.8% | 80.7% | 92.4% | 82.9% | 87.5% | 86.0% | 94.5% | 1000 |
| Ashape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_09/latest.pt` | 70.7% | 72.0% | 87.4% | 72.8% | 78.9% | 78.0% | 90.1% | 1000 |
| Vshape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_11/latest.pt` | 64.1% | 63.5% | 79.3% | 66.1% | 70.1% | 71.1% | 85.0% | 999 |
| 4shape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_13/latest.pt` | 82.6% | 82.3% | 93.9% | 84.9% | 88.8% | 87.4% | 95.8% | 1000 |
| Tshape3d_env2 | `./Experiments/3dshape/Tshape3d_env2/latest.pt` | 63.5% | 64.6% | 80.3% | 68.0% | 72.0% | 73.2% | 86.0% | 1000 |
| rectangle_env3 | `./Experiments/3dshape/3dshape_09_04_09_14/latest.pt` | 94.9% | 95.0% | 99.1% | 97.8% | 98.3% | 98.1% | 99.1% | 1000 |
| Lshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_15/latest.pt` | 95.8% | 96.6% | 99.3% | 97.4% | 97.6% | 98.2% | 99.1% | 999 |
| Fshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_16/latest.pt` | 94.7% | 95.9% | 99.4% | 96.5% | 98.4% | 98.4% | 98.5% | 999 |
| Ashape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_21/latest.pt` | 93.6% | 94.2% | 98.2% | 95.4% | 97.5% | 97.2% | 98.8% | 999 |
| Vshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_22/latest.pt` | 92.0% | 92.5% | 97.6% | 95.6% | 97.8% | 97.8% | 99.2% | 999 |
| 4shape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_23/latest.pt` | 92.8% | 94.2% | 98.7% | 96.4% | 98.1% | 98.0% | 99.0% | 1000 |
| Tshape3d_env3 | `./Experiments/3dshape/Tshape3d_env3/latest.pt` | 92.1% | 90.9% | 97.9% | 93.7% | 96.6% | 96.6% | 98.4% | 998 |
| rectangle_env4 | `./Experiments/3dshape/3dshape_09_04_09_26/latest.pt` | 95.2% | 97.3% | 99.2% | 97.4% | 98.2% | 97.9% | 98.8% | 1000 |
| Lshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_29/latest.pt` | 92.7% | 91.8% | 98.4% | 94.8% | 95.3% | 96.4% | 98.4% | 999 |
| Fshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_30/latest.pt` | 91.6% | 92.0% | 98.1% | 95.2% | 95.5% | 95.4% | 98.3% | 999 |
| Ashape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_49/latest.pt` | 87.6% | 86.5% | 95.8% | 91.6% | 93.2% | 92.4% | 98.7% | 999 |
| Vshape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_52/latest.pt` | 90.0% | 88.4% | 96.6% | 91.4% | 93.9% | 94.3% | 98.9% | 1000 |
| 4shape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_55/latest.pt` | 92.7% | 92.4% | 98.7% | 93.8% | 95.2% | 95.6% | 97.5% | 1000 |
| Tshape3d_env4 | `./Experiments/3dshape/Tshape3d_env4/latest.pt` | 86.8% | 86.8% | 94.9% | 89.8% | 92.7% | 93.3% | 98.0% | 1000 |
| Tshape3d_2denv4_legacy (2D) | `./Experiments/3dshape/3dshape_08_19_12_31/latest.pt` | 98.8% | 98.4% | 99.6% | 99.1% | 99.1% | 99.4% | 99.4% | 1000 |

## 3D_1k_valid

The same SE(3) cells as the 3-D shape task above -- same checkpoints,
same planners, same 1000 cases -- scored on `testing_data_1k_complete/`
instead of `testing_data/`.  Every pair in that set is collision-free *and*
carries an RRT-Connect path found within 180 s, so unlike the table above it
has a demonstrated 100% ceiling: a failure here is the planner's, not a
start/goal pair that nothing could solve.

Because the unsolvable pairs are gone, these numbers are **not** comparable
row-for-row with the 3-D table -- the two are different test populations, not
the same test measured twice.  See `testing_data_1k_complete/README.md` for
how the set was built and what fraction of sampled pairs it discarded.

The `Tshape3d_env*` rows were added 2026-09-12 (1k sets generated the same day;
the `Tshape3d_env3` checkpoint is the epoch-5500 model, as above).  The 3-D table's
`Tshape3d_2denv4_legacy (2D)` row has no counterpart here -- it is a planar
cell, and no legacy 1k set was generated for it.

Produced by `_run_3d_eval_1k.sh`; refresh this table alone with
`_make_3d_1k_table.py`.

| Env | Model | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | SR Alternate Bellman Horizon SD | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./Experiments/3dshape/3dshape_08_06_17_06/latest.pt` | 97.3% | 96.7% | 99.5% | 97.1% | 98.8% | 99.1% | 99.3% | 1000 |
| Lshape3d_env1 | `./Experiments/3dshape/3dshape_08_15_20_19/latest.pt` | 89.8% | 90.3% | 98.2% | 91.1% | 95.2% | 95.1% | 98.7% | 1000 |
| Fshape3d_env1 | `./Experiments/3dshape/3dshape_08_16_11_35/latest.pt` | 93.1% | 92.6% | 99.0% | 94.9% | 97.4% | 96.9% | 98.7% | 1000 |
| Ashape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_49/latest.pt` | 78.6% | 78.3% | 91.9% | 83.5% | 88.9% | 88.9% | 97.0% | 1000 |
| Vshape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_50/latest.pt` | 82.0% | 80.1% | 93.9% | 86.7% | 89.5% | 91.1% | 96.4% | 1000 |
| 4shape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_54/latest.pt` | 90.9% | 89.5% | 97.7% | 93.2% | 95.5% | 95.6% | 97.2% | 1000 |
| Tshape3d_env1 | `./Experiments/3dshape/Tshape3d_env1/latest.pt` | 84.9% | 83.7% | 95.3% | 88.0% | 92.6% | 93.4% | 97.4% | 1000 |
| rectangle_env2 | `./Experiments/3dshape/3dshape_08_29_15_41/latest.pt` | 96.0% | 95.7% | 99.7% | 95.0% | 97.9% | 96.7% | 98.9% | 1000 |
| Lshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_04/latest.pt` | 80.1% | 82.7% | 93.4% | 83.1% | 85.5% | 86.7% | 93.2% | 1000 |
| Fshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_05/latest.pt` | 78.7% | 82.2% | 91.5% | 79.9% | 84.2% | 84.6% | 93.1% | 1000 |
| Ashape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_09/latest.pt` | 71.6% | 70.2% | 86.7% | 72.9% | 78.3% | 78.8% | 87.5% | 1000 |
| Vshape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_11/latest.pt` | 62.9% | 63.0% | 79.6% | 66.6% | 73.3% | 72.2% | 85.4% | 1000 |
| 4shape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_13/latest.pt` | 82.3% | 83.1% | 94.4% | 84.0% | 88.0% | 87.9% | 94.7% | 1000 |
| Tshape3d_env2 | `./Experiments/3dshape/Tshape3d_env2/latest.pt` | 65.4% | 64.1% | 82.7% | 67.9% | 73.2% | 72.8% | 85.4% | 999 |
| rectangle_env3 | `./Experiments/3dshape/3dshape_09_04_09_14/latest.pt` | 95.4% | 94.4% | 99.3% | 97.4% | 98.3% | 98.1% | 98.9% | 1000 |
| Lshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_15/latest.pt` | 95.6% | 95.2% | 98.8% | 97.6% | 98.3% | 98.2% | 99.1% | 1000 |
| Fshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_16/latest.pt` | 93.8% | 95.2% | 98.8% | 96.5% | 97.8% | 97.7% | 99.3% | 1000 |
| Ashape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_21/latest.pt` | 92.9% | 93.5% | 98.6% | 94.9% | 97.3% | 97.4% | 99.1% | 1000 |
| Vshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_22/latest.pt` | 93.5% | 93.9% | 98.4% | 97.4% | 98.5% | 98.7% | 99.5% | 1000 |
| 4shape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_23/latest.pt` | 95.9% | 94.5% | 99.5% | 97.0% | 98.3% | 98.1% | 99.1% | 1000 |
| Tshape3d_env3 | `./Experiments/3dshape/Tshape3d_env3/latest.pt` | 91.2% | 90.5% | 97.5% | 94.4% | 95.5% | 97.1% | 98.2% | 1000 |
| rectangle_env4 | `./Experiments/3dshape/3dshape_09_04_09_26/latest.pt` | 97.2% | 96.6% | 99.8% | 98.3% | 98.4% | 98.4% | 98.9% | 1000 |
| Lshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_29/latest.pt` | 93.3% | 93.4% | 98.6% | 95.3% | 96.7% | 96.4% | 99.0% | 1000 |
| Fshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_30/latest.pt` | 94.2% | 91.7% | 98.4% | 95.0% | 96.6% | 97.1% | 99.0% | 1000 |
| Ashape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_49/latest.pt` | 85.5% | 86.1% | 95.4% | 88.3% | 92.4% | 91.7% | 97.3% | 1000 |
| Vshape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_52/latest.pt` | 88.6% | 88.8% | 97.6% | 92.0% | 93.2% | 94.0% | 97.2% | 1000 |
| 4shape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_55/latest.pt` | 92.5% | 93.0% | 98.7% | 94.5% | 95.8% | 96.4% | 98.6% | 1000 |
| Tshape3d_env4 | `./Experiments/3dshape/Tshape3d_env4/latest.pt` | 85.2% | 86.0% | 95.8% | 89.8% | 94.1% | 93.0% | 97.9% | 1000 |

## 3D_tight

The same SE(3) cells as the 3-D shape task above -- same checkpoints,
same planners, same 1000 cases -- scored on the `_tight` test sets in
`testing_data/3dshape/` instead of the regular ones.  A tight set is the
same rejection sampler run at `--offset 0.005` instead of `0.02`
(`dataprocessing/make_tight_testing_data.py`), so start and goal poses may
sit four times closer to the obstacles; the environments and the models are
unchanged, only the queries are harder.  Like the 3-D table (and unlike
`3D_1k_valid`) nothing guarantees every pair is solvable, so these numbers
are a different test population again, not a re-measurement of either
table above.  Note `test_cases`: the evaluator's own endpoint check rejects
a start or goal it finds in collision, and at this offset that drops 4-8% of
the 1000 sampled pairs per cell (0-1 on the regular sets); every SR is
over the pairs that remain.

The `Tshape3d_env2..4` tight sets were generated 2026-09-12 by the `gen`
stage of the driver; `Tshape3d_env1_tight` is the 2026-08-19 tight set (same
mesh, env and offset as the cell, an older draw -- it is the set kept as
`Tshape3d_env1_aug18_tight` in `testing_data_old_dontuse/`); the
`Tshape3d_env3` checkpoint is the epoch-5500 model, as above.  The 3-D
table's `Tshape3d_2denv4_legacy (2D)` row has no counterpart here.

Produced by `_run_3d_eval_tight.sh`; refresh this table alone with
`_make_3d_tight_table.py`.

| Env | Model | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | SR Alternate Bellman Horizon SD | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./Experiments/3dshape/3dshape_08_06_17_06/latest.pt` | 88.7% | 88.9% | 97.7% | 96.6% | 95.6% | 96.1% | 97.8% | 958 |
| Lshape3d_env1 | `./Experiments/3dshape/3dshape_08_15_20_19/latest.pt` | 77.3% | 78.7% | 93.1% | 88.8% | 90.2% | 90.5% | 96.3% | 965 |
| Fshape3d_env1 | `./Experiments/3dshape/3dshape_08_16_11_35/latest.pt` | 83.7% | 83.6% | 95.6% | 92.8% | 93.1% | 92.6% | 97.3% | 963 |
| Ashape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_49/latest.pt` | 68.6% | 68.8% | 85.5% | 81.1% | 83.1% | 82.3% | 93.8% | 951 |
| Vshape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_50/latest.pt` | 72.7% | 67.7% | 88.0% | 83.1% | 83.2% | 83.7% | 92.9% | 951 |
| 4shape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_54/latest.pt` | 81.5% | 81.7% | 95.3% | 90.8% | 91.3% | 91.7% | 96.4% | 952 |
| Tshape3d_env1 | `./Experiments/3dshape/Tshape3d_env1/latest.pt` | 72.7% | 73.3% | 89.5% | 80.8% | 84.0% | 85.0% | 94.4% | 955 |
| rectangle_env2 | `./Experiments/3dshape/3dshape_08_29_15_41/latest.pt` | 89.7% | 88.1% | 98.0% | 96.2% | 95.3% | 95.2% | 97.6% | 965 |
| Lshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_04/latest.pt` | 73.3% | 70.4% | 88.4% | 81.6% | 82.4% | 81.5% | 91.8% | 960 |
| Fshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_05/latest.pt` | 73.6% | 70.1% | 87.5% | 79.5% | 80.3% | 81.8% | 91.5% | 951 |
| Ashape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_09/latest.pt` | 60.3% | 63.5% | 79.8% | 71.8% | 70.4% | 72.8% | 85.3% | 942 |
| Vshape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_11/latest.pt` | 55.0% | 57.5% | 75.8% | 64.2% | 66.6% | 67.8% | 82.0% | 938 |
| 4shape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_13/latest.pt` | 74.7% | 71.1% | 89.0% | 81.1% | 81.8% | 81.9% | 92.0% | 940 |
| Tshape3d_env2 | `./Experiments/3dshape/Tshape3d_env2/latest.pt` | 55.5% | 54.5% | 73.6% | 64.5% | 69.1% | 66.9% | 82.5% | 928 |
| rectangle_env3 | `./Experiments/3dshape/3dshape_09_04_09_14/latest.pt` | 85.1% | 85.6% | 97.4% | 96.7% | 95.7% | 94.6% | 98.5% | 957 |
| Lshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_15/latest.pt` | 88.2% | 87.4% | 97.2% | 95.2% | 94.9% | 94.9% | 98.0% | 950 |
| Fshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_16/latest.pt` | 86.9% | 88.4% | 97.3% | 95.8% | 94.3% | 94.0% | 97.2% | 937 |
| Ashape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_21/latest.pt` | 85.3% | 82.4% | 94.2% | 91.6% | 92.2% | 91.9% | 96.3% | 919 |
| Vshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_22/latest.pt` | 84.7% | 82.0% | 94.5% | 92.8% | 93.1% | 93.6% | 97.2% | 927 |
| 4shape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_23/latest.pt` | 83.4% | 85.8% | 94.8% | 92.5% | 92.0% | 93.6% | 96.5% | 941 |
| Tshape3d_env3 | `./Experiments/3dshape/Tshape3d_env3/latest.pt` | 80.2% | 79.4% | 93.3% | 90.8% | 89.1% | 89.7% | 94.4% | 921 |
| rectangle_env4 | `./Experiments/3dshape/3dshape_09_04_09_26/latest.pt` | 89.2% | 88.3% | 97.5% | 95.8% | 95.5% | 95.6% | 97.9% | 955 |
| Lshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_29/latest.pt` | 83.1% | 83.5% | 95.1% | 92.5% | 91.1% | 92.0% | 96.8% | 936 |
| Fshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_30/latest.pt` | 83.3% | 79.8% | 94.0% | 91.9% | 91.3% | 89.5% | 96.0% | 936 |
| Ashape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_49/latest.pt` | 76.2% | 76.2% | 90.0% | 86.3% | 89.5% | 88.2% | 96.7% | 923 |
| Vshape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_52/latest.pt` | 77.8% | 76.6% | 91.6% | 86.3% | 87.8% | 88.1% | 96.8% | 929 |
| 4shape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_55/latest.pt` | 82.5% | 78.9% | 94.1% | 91.4% | 89.7% | 88.4% | 94.9% | 932 |
| Tshape3d_env4 | `./Experiments/3dshape/Tshape3d_env4/latest.pt` | 74.3% | 74.9% | 89.3% | 86.2% | 88.4% | 87.0% | 96.2% | 938 |

## Gibson (Lcouch_Corozal)

A real indoor scan instead of a synthetic obstacle field: the `Lcouch` couch
body (12 verts, 18 tets) in the Gibson `Corozal` house mesh (85k triangles,
normalised to a 0.58 x 1.0 x 0.33 box, so the couch spans ~0.1 of the long
axis).  Trained like every SE(3) cell above -- 800k pairs at `--margin 0.05
--offset 0.001`, 10000 epochs -- on 2026-08-31 (README: "Corozal").

Scored on `testing_data_1k_complete/3dshape/Lcouch_Corozal`, an RRT-verified
set built the same way as the 1k sets (`generate_testing_data_rrt.py`,
`--offset 0.02`, 180 s RRT-Connect budget, three 50k-point collision audits)
but with **500 pairs** instead of 1000; see `testing_data_1k_complete/README.md`
for the acceptance rate.  Start/goal poses are sampled uniformly over the
mesh bounding box, so pairs can sit outside the building envelope as well as
in its rooms -- what RRT-Connect can connect, including around the outside of
the walls, is what the set contains.

Columns: the six success-rate planners of the 3-D table, the SD planner
(`_spread_planner.py --steer-bias 0.5 --no-gate`), and the `er_opt.py`
length / gen-time figures (CUDA-graph mode, mean +- std over the converged
cases, same 500 pairs) that `CHARTS_PAPER.md` quotes for the Ours row.
Produced by `_run_gibson_eval_1k.sh` (2026-09-13); refresh this table alone
with `_make_gibson_table.py`.

The failures are almost all **non-convergence**, not collision: forward
planner 156 no-converge vs 21 collisions, hlB 136 vs 7, SD 134 vs 1, and the
median closest approach of a non-converged rollout is 0.67 -- the walk stalls
far from the goal rather than grazing it.  Pairs whose straight-line
interpolation is collision-free succeed at 97% (195/201); the rest at 47%
(142/299), so the field is reliable locally and unreliable across the
building.  `results/output_3d_1k/Lcouch_Corozal/success_rate.txt` has the
full breakdown.

| Env | Model | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | SR Alternate Bellman Horizon SD | test_cases | Trans len | Rot len (rad) | Gen time (s) | er_opt cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Lcouch_Corozal | `./Experiments/3dshape/3dshape_08_31_13_53/latest.pt` | 67.4% | 69.8% | 82.6% | 75.8% | 71.8% | 71.8% | 73.0% | 500 | 0.882 +- 0.513 | 4.149 +- 1.742 | 0.149 +- 0.146 | 500 (367 converged) |

## 2-D shape task (`--2d`)

Seven shapes across all four planar environments -- `2denv1_zup.obj` (12 bodies),
`2denv2_zup.obj` (13), `2denv3_zup.obj` (6) and `2denv4_zup.obj` (8), each a
350 x 350 footprint -- 28 cells, trained for 5000 epochs and scored on 1000
held-out start/goal pairs with `evaluate_training_3d_batched.py --2d`.
The commands are in `README.md`; preprocessing ran per env
(`_run_2denv{1,2,3,4}_preprocess.sh`, times in `2d_gen_times.md`), training via
`_run_2d_train_ours.sh`, evaluation via `_run_2d_eval.sh`.

`Train Time (s)` is the wall clock of that cell's 5000-epoch run, taken from
the trainer's closing `Training time:` line -- the same quantity
`experiments_ntfields.md` reports, so the two are directly comparable.
`Training Time` is that figure in h/m/s. A cell still training shows `--`.
The June-3 table has no such column -- that pipeline has not been run on any
2-D cell.

Both pipelines are scored on the SAME test sets (`testing_data/3dshape/<ds>`,
generated once by the current preprocessor at `--offset 0.02`), so the two tables
differ only in how the training data was generated and which network was fit.

### Current pipeline (`preprocess_obj.py --2d` + `models/metric`)

800k training pairs, `--margin 0.05 --offset 0.001`. `Tshape3d_2denv4` trains
from `datasets/3dshape/Tshape3d_2denv4` (built 2026-08-18 as `Tshape3d_env4`,
renamed 2026-09-10 when the 3-D `Tshape3d_env4` cell took that name).

| Env | Model | Train Time (s) | Training Time | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | SR Alternate Bellman Horizon SD | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv1 | `./Experiments/3dshape_2d/rectangle_2denv1/latest.pt` | 4458 | 1h 14m 18s | 94.0% | 95.1% | 98.3% | 95.9% | 97.1% | 98.2% | 95.8% | 1000 |
| Lshape3d_2denv1 | `./Experiments/3dshape_2d/Lshape3d_2denv1/latest.pt` | 5777 | 1h 36m 17s | 90.3% | 90.9% | 96.7% | 90.9% | 96.0% | 95.5% | 94.6% | 1000 |
| Fshape3d_2denv1 | `./Experiments/3dshape_2d/Fshape3d_2denv1/latest.pt` | 5772 | 1h 36m 12s | 90.9% | 89.7% | 96.1% | 90.8% | 93.5% | 94.5% | 94.1% | 1000 |
| Ashape3d_2denv1 | `./Experiments/3dshape_2d/Ashape3d_2denv1/latest.pt` | 5317 | 1h 28m 37s | 86.2% | 84.4% | 89.6% | 86.9% | 87.9% | 88.2% | 88.6% | 1000 |
| Vshape3d_2denv1 | `./Experiments/3dshape_2d/Vshape3d_2denv1/latest.pt` | 5323 | 1h 28m 43s | 83.7% | 85.5% | 88.2% | 85.0% | 87.3% | 87.9% | 87.6% | 1000 |
| 4shape3d_2denv1 | `./Experiments/3dshape_2d/4shape3d_2denv1/latest.pt` | 4399 | 1h 13m 19s | 90.9% | 91.5% | 96.0% | 94.2% | 94.1% | 95.2% | 94.9% | 1000 |
| Tshape3d_2denv1 | `./Experiments/3dshape_2d/Tshape3d_2denv1/latest.pt` | 5796 | 1h 36m 36s | 83.1% | 82.5% | 87.6% | 81.5% | 83.6% | 83.8% | 84.2% | 1000 |
| rectangle_2denv2 | `./Experiments/3dshape_2d/rectangle_2denv2/latest.pt` | 5776 | 1h 36m 16s | 92.8% | 91.9% | 97.9% | 95.4% | 94.1% | 95.0% | 90.0% | 1000 |
| Lshape3d_2denv2 | `./Experiments/3dshape_2d/Lshape3d_2denv2/latest.pt` | 5319 | 1h 28m 39s | 80.0% | 81.5% | 92.6% | 81.7% | 89.4% | 90.7% | 91.4% | 1000 |
| Fshape3d_2denv2 | `./Experiments/3dshape_2d/Fshape3d_2denv2/latest.pt` | 5308 | 1h 28m 28s | 82.0% | 81.4% | 92.3% | 83.4% | 89.1% | 90.2% | 91.8% | 1000 |
| Ashape3d_2denv2 | `./Experiments/3dshape_2d/Ashape3d_2denv2/latest.pt` | 4358 | 1h 12m 38s | 41.2% | 40.3% | 45.5% | 41.1% | 49.9% | 49.6% | 61.3% | 1000 |
| Vshape3d_2denv2 | `./Experiments/3dshape_2d/Vshape3d_2denv2/latest.pt` | 4914 | 1h 21m 54s | 39.4% | 40.3% | 45.3% | 40.4% | 50.7% | 54.4% | 58.1% | 1000 |
| 4shape3d_2denv2 | `./Experiments/3dshape_2d/4shape3d_2denv2/latest.pt` | 4898 | 1h 21m 38s | 73.9% | 72.4% | 84.3% | 73.5% | 82.1% | 82.7% | 85.4% | 1000 |
| Tshape3d_2denv2 | `./Experiments/3dshape_2d/Tshape3d_2denv2/latest.pt` | 4915 | 1h 21m 55s | 41.7% | 41.4% | 46.3% | 41.3% | 49.3% | 50.2% | 54.8% | 1000 |
| rectangle_2denv3 | `./Experiments/3dshape_2d/rectangle_2denv3/latest.pt` | 5005 | 1h 23m 25s | 94.9% | 94.8% | 99.5% | 99.5% | 98.7% | 98.9% | 97.0% | 1000 |
| Lshape3d_2denv3 | `./Experiments/3dshape_2d/Lshape3d_2denv3/latest.pt` | 4039 | 1h 07m 19s | 98.2% | 98.0% | 99.9% | 100.0% | 100.0% | 100.0% | 99.8% | 1000 |
| Fshape3d_2denv3 | `./Experiments/3dshape_2d/Fshape3d_2denv3/latest.pt` | 4719 | 1h 18m 39s | 98.7% | 98.8% | 99.9% | 100.0% | 99.9% | 99.9% | 99.8% | 1000 |
| Ashape3d_2denv3 | `./Experiments/3dshape_2d/Ashape3d_2denv3/latest.pt` | 4693 | 1h 18m 13s | 96.2% | 95.3% | 98.9% | 96.6% | 97.9% | 97.4% | 98.0% | 1000 |
| Vshape3d_2denv3 | `./Experiments/3dshape_2d/Vshape3d_2denv3/latest.pt` | 4731 | 1h 18m 51s | 98.7% | 99.1% | 100.0% | 99.5% | 99.7% | 99.7% | 99.8% | 1000 |
| 4shape3d_2denv3 | `./Experiments/3dshape_2d/4shape3d_2denv3/latest.pt` | 4745 | 1h 19m 05s | 99.6% | 99.0% | 100.0% | 99.5% | 99.7% | 99.6% | 99.7% | 1000 |
| Tshape3d_2denv3 | `./Experiments/3dshape_2d/Tshape3d_2denv3/latest.pt` | 3957 | 1h 05m 57s | 98.6% | 99.2% | 99.9% | 98.9% | 99.1% | 98.9% | 98.6% | 1000 |
| rectangle_2denv4 | `./Experiments/3dshape_2d/rectangle_2denv4/latest.pt` | 4590 | 1h 16m 30s | 97.9% | 98.0% | 99.9% | 99.7% | 99.0% | 98.8% | 98.1% | 1000 |
| Lshape3d_2denv4 | `./Experiments/3dshape_2d/Lshape3d_2denv4/latest.pt` | 4582 | 1h 16m 22s | 98.3% | 97.8% | 99.4% | 98.4% | 99.9% | 99.9% | 99.7% | 1000 |
| Fshape3d_2denv4 | `./Experiments/3dshape_2d/Fshape3d_2denv4/latest.pt` | 4719 | 1h 18m 39s | 98.7% | 98.5% | 100.0% | 98.7% | 99.6% | 99.5% | 99.5% | 1000 |
| Ashape3d_2denv4 | `./Experiments/3dshape_2d/Ashape3d_2denv4/latest.pt` | 4695 | 1h 18m 15s | 93.4% | 94.5% | 98.8% | 92.6% | 96.5% | 96.1% | 98.2% | 1000 |
| Vshape3d_2denv4 | `./Experiments/3dshape_2d/Vshape3d_2denv4/latest.pt` | 3902 | 1h 05m 02s | 99.2% | 99.2% | 99.8% | 99.2% | 99.8% | 99.7% | 99.6% | 1000 |
| 4shape3d_2denv4 | `./Experiments/3dshape_2d/4shape3d_2denv4/latest.pt` | 3981 | 1h 06m 21s | 99.3% | 99.4% | 100.0% | 99.3% | 99.4% | 99.6% | 99.3% | 1000 |
| Tshape3d_2denv4 | `./Experiments/3dshape_2d/Tshape3d_2denv4/latest.pt` | 3990 | 1h 06m 30s | 98.7% | 98.2% | 99.9% | 97.1% | 99.0% | 99.3% | 99.2% | 1000 |

### Recovered June-3 pipeline (`models/metric_june03`)

The frozen single-route June-3 network, 5000 epochs, trained by
`_run_2d_june03_train.sh` on the **shared current-pipeline datasets** --
the same `datasets/3dshape/<shape>_<env>` dirs the current rows above use,
not regenerated `_june03` data.  `models/metric_june03/data_mlp.py` reads only
`sampled_points` / `speed` / `normal`, which those dirs already carry, so the
generator is held fixed and a difference here is a statement about the network
(single embedding route, Fourier `B` at std 0.86 vs 0.2) rather than the data.

`preprocess_obj_june03.py` (400k pairs, `--margin 0.1 --offset 0.01`, 8000 env
points) was NOT used for these rows; it remains available if the generator
ablation is wanted separately.

| Env | Model | Train Time (s) | Training Time | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | SR Alternate Bellman Horizon SD | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv1 | `./Experiments/3dshape_2d_june03/rectangle_2denv1/latest.pt` | 2585 | 0h 43m 05s | 87.2% | 89.2% | 95.2% | 89.0% | 91.3% | 91.8% | 95.8% | 1000 |
| Lshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Lshape3d_2denv1/latest.pt` | 2106 | 0h 35m 06s | 92.4% | 92.5% | 97.1% | 94.6% | 94.4% | 95.1% | 94.6% | 1000 |
| Fshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Fshape3d_2denv1/latest.pt` | 2115 | 0h 35m 15s | 85.7% | 88.6% | 94.0% | 90.9% | 92.5% | 93.0% | 94.1% | 1000 |
| Ashape3d_2denv1 | `./Experiments/3dshape_2d_june03/Ashape3d_2denv1/latest.pt` | 2746 | 0h 45m 46s | 81.2% | 80.2% | 86.5% | 83.5% | 82.9% | 83.2% | 88.6% | 1000 |
| Vshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Vshape3d_2denv1/latest.pt` | 2726 | 0h 45m 26s | 86.5% | 84.5% | 89.1% | 87.7% | 88.1% | 88.7% | 87.6% | 1000 |
| 4shape3d_2denv1 | `./Experiments/3dshape_2d_june03/4shape3d_2denv1/latest.pt` | 2485 | 0h 41m 25s | -- | -- | -- | -- | -- | -- | 94.9% | -- |
| Tshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Tshape3d_2denv1/latest.pt` | 1957 | 0h 32m 37s | -- | -- | -- | -- | -- | -- | 84.2% | -- |
| rectangle_2denv2 | `./Experiments/3dshape_2d_june03/rectangle_2denv2/latest.pt` | 1966 | 0h 32m 46s | -- | -- | -- | -- | -- | -- | 90.0% | -- |
| Lshape3d_2denv2 | `./Experiments/3dshape_2d_june03/Lshape3d_2denv2/latest.pt` | 2615 | 0h 43m 35s | -- | -- | -- | -- | -- | -- | 91.4% | -- |
| Fshape3d_2denv2 | `./Experiments/3dshape_2d_june03/Fshape3d_2denv2/latest.pt` | 2650 | 0h 44m 10s | -- | -- | -- | -- | -- | -- | 91.8% | -- |
| Ashape3d_2denv2 | `./Experiments/3dshape_2d_june03/Ashape3d_2denv2/latest.pt` | 2410 | 0h 40m 10s | -- | -- | -- | -- | -- | -- | 61.3% | -- |
| Vshape3d_2denv2 | `./Experiments/3dshape_2d_june03/Vshape3d_2denv2/latest.pt` | 2171 | 0h 36m 11s | -- | -- | -- | -- | -- | -- | 58.1% | -- |
| 4shape3d_2denv2 | `./Experiments/3dshape_2d_june03/4shape3d_2denv2/latest.pt` | 2182 | 0h 36m 22s | -- | -- | -- | -- | -- | -- | 85.4% | -- |
| Tshape3d_2denv2 | `./Experiments/3dshape_2d_june03/Tshape3d_2denv2/latest.pt` | 2720 | 0h 45m 20s | -- | -- | -- | -- | -- | -- | 54.8% | -- |
| rectangle_2denv3 | `./Experiments/3dshape_2d_june03/rectangle_2denv3/latest.pt` | 2706 | 0h 45m 06s | -- | -- | -- | -- | -- | -- | 97.0% | -- |
| Lshape3d_2denv3 | `./Experiments/3dshape_2d_june03/Lshape3d_2denv3/latest.pt` | 1801 | 0h 30m 01s | -- | -- | -- | -- | -- | -- | 99.8% | -- |
| Fshape3d_2denv3 | `./Experiments/3dshape_2d_june03/Fshape3d_2denv3/latest.pt` | 1983 | 0h 33m 03s | -- | -- | -- | -- | -- | -- | 99.8% | -- |
| Ashape3d_2denv3 | `./Experiments/3dshape_2d_june03/Ashape3d_2denv3/latest.pt` | 1972 | 0h 32m 52s | -- | -- | -- | -- | -- | -- | 98.0% | -- |
| Vshape3d_2denv3 | `./Experiments/3dshape_2d_june03/Vshape3d_2denv3/latest.pt` | 2539 | 0h 42m 19s | -- | -- | -- | -- | -- | -- | 99.8% | -- |
| 4shape3d_2denv3 | `./Experiments/3dshape_2d_june03/4shape3d_2denv3/latest.pt` | 2534 | 0h 42m 14s | -- | -- | -- | -- | -- | -- | 99.7% | -- |
| Tshape3d_2denv3 | `./Experiments/3dshape_2d_june03/Tshape3d_2denv3/latest.pt` | 1764 | 0h 29m 24s | -- | -- | -- | -- | -- | -- | 98.6% | -- |
| rectangle_2denv4 | `./Experiments/3dshape_2d_june03/rectangle_2denv4/latest.pt` | 1890 | 0h 31m 30s | -- | -- | -- | -- | -- | -- | 98.1% | -- |
| Lshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Lshape3d_2denv4/latest.pt` | 1889 | 0h 31m 29s | -- | -- | -- | -- | -- | -- | 99.7% | -- |
| Fshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Fshape3d_2denv4/latest.pt` | 2122 | 0h 35m 22s | -- | -- | -- | -- | -- | -- | 99.5% | -- |
| Ashape3d_2denv4 | `./Experiments/3dshape_2d_june03/Ashape3d_2denv4/latest.pt` | 1891 | 0h 31m 31s | -- | -- | -- | -- | -- | -- | 98.2% | -- |
| Vshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Vshape3d_2denv4/latest.pt` | 1846 | 0h 30m 46s | -- | -- | -- | -- | -- | -- | 99.6% | -- |
| 4shape3d_2denv4 | `./Experiments/3dshape_2d_june03/4shape3d_2denv4/latest.pt` | 1878 | 0h 31m 18s | -- | -- | -- | -- | -- | -- | 99.3% | -- |
| Tshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Tshape3d_2denv4/latest.pt` | 1883 | 0h 31m 23s | -- | -- | -- | -- | -- | -- | 99.2% | -- |

### Head to head (`SR Forward`)

| Env | current | June-3 | delta |
| --- | --- | --- | --- |
| rectangle_2denv1 | 94.0% | 87.2% | +6.8 |
| Lshape3d_2denv1 | 90.3% | 92.4% | -2.1 |
| Fshape3d_2denv1 | 90.9% | 85.7% | +5.2 |
| Ashape3d_2denv1 | 86.2% | 81.2% | +5.0 |
| Vshape3d_2denv1 | 83.7% | 86.5% | -2.8 |
| 4shape3d_2denv1 | 90.9% | -- | -- |
| Tshape3d_2denv1 | 83.1% | -- | -- |
| rectangle_2denv2 | 92.8% | -- | -- |
| Lshape3d_2denv2 | 80.0% | -- | -- |
| Fshape3d_2denv2 | 82.0% | -- | -- |
| Ashape3d_2denv2 | 41.2% | -- | -- |
| Vshape3d_2denv2 | 39.4% | -- | -- |
| 4shape3d_2denv2 | 73.9% | -- | -- |
| Tshape3d_2denv2 | 41.7% | -- | -- |
| rectangle_2denv3 | 94.9% | -- | -- |
| Lshape3d_2denv3 | 98.2% | -- | -- |
| Fshape3d_2denv3 | 98.7% | -- | -- |
| Ashape3d_2denv3 | 96.2% | -- | -- |
| Vshape3d_2denv3 | 98.7% | -- | -- |
| 4shape3d_2denv3 | 99.6% | -- | -- |
| Tshape3d_2denv3 | 98.6% | -- | -- |
| rectangle_2denv4 | 97.9% | -- | -- |
| Lshape3d_2denv4 | 98.3% | -- | -- |
| Fshape3d_2denv4 | 98.7% | -- | -- |
| Ashape3d_2denv4 | 93.4% | -- | -- |
| Vshape3d_2denv4 | 99.2% | -- | -- |
| 4shape3d_2denv4 | 99.3% | -- | -- |
| Tshape3d_2denv4 | 98.7% | -- | -- |
| **MEAN** | **87.2%** | **86.6%** | **+2.4** |

## 2d_1k_valid

The same 28 planar cells as the current pipeline above -- same
checkpoints, same planners, same 1000 cases -- scored on
`testing_data_1k_complete/` instead of `testing_data/`.  Every pair in that
set is collision-free *and* carries an RRT-Connect path found within 180 s,
so the table has a demonstrated 100% ceiling: a failure here is the
planner's, not a start/goal pair that nothing could solve.

**This matters far more in 2-D than it did in 3-D.**  The SE(3) sets lost
almost nothing to the filter (RRT-Connect already solved 100% of 23 of 24
cells), so `3D_1k_valid` came out within a quarter point of the 3-D table.
Here the filter is doing real work: it discarded 46% of sampled pairs in
`Tshape3d_2denv2`, 36% in `Ashape3d_2denv2` and 30% in `Vshape3d_2denv2`,
all of them the hardest pairs in those cells.  Expect those rows to read
substantially higher than above, and do **not** compare them row-for-row:
the two tables measure different populations.  Per-cell discard rates are
in `testing_data_1k_complete/README.md`.

`Train Time (s)` / `Training Time` are copied from the table above -- they
describe the checkpoint, which is the same file, not the test set.

The SD column keeps `--no-gate`, matching the shipped 2-D table, even
though a 2026-09-09 sweep found the cv gate is the better 2-D setting
(+2.88 mean vs +1.60).  Changing the planner and the test set at once would
make this table uninterpretable as a comparison.

Produced by `_run_2d_eval_1k.sh`; refresh this table alone with
`_make_2d_1k_table.py`.  Its SD column comes from `results/spread_sd_2d_1k/`,
so do not chain `_make_spread_sd.py` after it.

| Env | Model | Train Time (s) | Training Time | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | SR Alternate Bellman Horizon SD | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv1 | `./Experiments/3dshape_2d/rectangle_2denv1/latest.pt` | 4458 | 1h 14m 18s | 94.1% | 94.6% | 98.3% | 95.8% | 97.7% | 98.3% | 96.5% | 1000 |
| Lshape3d_2denv1 | `./Experiments/3dshape_2d/Lshape3d_2denv1/latest.pt` | 5777 | 1h 36m 17s | 91.7% | 90.7% | 97.3% | 91.6% | 95.7% | 95.7% | 94.4% | 1000 |
| Fshape3d_2denv1 | `./Experiments/3dshape_2d/Fshape3d_2denv1/latest.pt` | 5772 | 1h 36m 12s | 92.0% | 90.6% | 96.3% | 93.2% | 94.7% | 95.7% | 95.5% | 1000 |
| Ashape3d_2denv1 | `./Experiments/3dshape_2d/Ashape3d_2denv1/latest.pt` | 5317 | 1h 28m 37s | 87.9% | 88.3% | 91.4% | 89.3% | 90.4% | 90.9% | 90.4% | 1000 |
| Vshape3d_2denv1 | `./Experiments/3dshape_2d/Vshape3d_2denv1/latest.pt` | 5323 | 1h 28m 43s | 86.9% | 88.1% | 91.0% | 87.5% | 90.2% | 90.7% | 90.7% | 1000 |
| 4shape3d_2denv1 | `./Experiments/3dshape_2d/4shape3d_2denv1/latest.pt` | 4399 | 1h 13m 19s | 90.2% | 91.5% | 96.0% | 93.1% | 95.2% | 95.1% | 95.3% | 1000 |
| Tshape3d_2denv1 | `./Experiments/3dshape_2d/Tshape3d_2denv1/latest.pt` | 5796 | 1h 36m 36s | 87.9% | 88.4% | 92.5% | 88.8% | 90.9% | 91.3% | 90.9% | 1000 |
| rectangle_2denv2 | `./Experiments/3dshape_2d/rectangle_2denv2/latest.pt` | 5776 | 1h 36m 16s | 91.9% | 92.9% | 97.8% | 96.1% | 96.4% | 96.0% | 91.0% | 1000 |
| Lshape3d_2denv2 | `./Experiments/3dshape_2d/Lshape3d_2denv2/latest.pt` | 5319 | 1h 28m 39s | 81.4% | 80.9% | 93.5% | 80.6% | 90.6% | 90.9% | 91.8% | 1000 |
| Fshape3d_2denv2 | `./Experiments/3dshape_2d/Fshape3d_2denv2/latest.pt` | 5308 | 1h 28m 28s | 83.1% | 81.3% | 93.0% | 82.3% | 89.4% | 89.6% | 91.6% | 1000 |
| Ashape3d_2denv2 | `./Experiments/3dshape_2d/Ashape3d_2denv2/latest.pt` | 4358 | 1h 12m 38s | 63.6% | 64.3% | 71.1% | 64.7% | 74.1% | 75.1% | 78.1% | 1000 |
| Vshape3d_2denv2 | `./Experiments/3dshape_2d/Vshape3d_2denv2/latest.pt` | 4914 | 1h 21m 54s | 62.4% | 62.7% | 71.4% | 64.3% | 70.3% | 73.3% | 76.5% | 1000 |
| 4shape3d_2denv2 | `./Experiments/3dshape_2d/4shape3d_2denv2/latest.pt` | 4898 | 1h 21m 38s | 74.0% | 72.5% | 85.5% | 74.5% | 80.8% | 83.3% | 86.4% | 1000 |
| Tshape3d_2denv2 | `./Experiments/3dshape_2d/Tshape3d_2denv2/latest.pt` | 4915 | 1h 21m 55s | 77.2% | 76.6% | 81.6% | 75.0% | 81.0% | 82.7% | 86.7% | 1000 |
| rectangle_2denv3 | `./Experiments/3dshape_2d/rectangle_2denv3/latest.pt` | 5005 | 1h 23m 25s | 95.1% | 94.9% | 99.1% | 99.3% | 99.0% | 98.9% | 98.2% | 1000 |
| Lshape3d_2denv3 | `./Experiments/3dshape_2d/Lshape3d_2denv3/latest.pt` | 4039 | 1h 07m 19s | 98.1% | 97.9% | 99.9% | 100.0% | 99.9% | 99.9% | 99.9% | 1000 |
| Fshape3d_2denv3 | `./Experiments/3dshape_2d/Fshape3d_2denv3/latest.pt` | 4719 | 1h 18m 39s | 98.9% | 98.7% | 100.0% | 99.9% | 99.9% | 99.8% | 99.5% | 1000 |
| Ashape3d_2denv3 | `./Experiments/3dshape_2d/Ashape3d_2denv3/latest.pt` | 4693 | 1h 18m 13s | 95.2% | 96.5% | 99.3% | 96.7% | 97.0% | 96.5% | 97.3% | 1000 |
| Vshape3d_2denv3 | `./Experiments/3dshape_2d/Vshape3d_2denv3/latest.pt` | 4731 | 1h 18m 51s | 98.5% | 99.0% | 100.0% | 99.5% | 99.8% | 99.8% | 99.9% | 1000 |
| 4shape3d_2denv3 | `./Experiments/3dshape_2d/4shape3d_2denv3/latest.pt` | 4745 | 1h 19m 05s | 99.5% | 98.7% | 100.0% | 99.6% | 99.6% | 99.8% | 99.5% | 1000 |
| Tshape3d_2denv3 | `./Experiments/3dshape_2d/Tshape3d_2denv3/latest.pt` | 3957 | 1h 05m 57s | 99.0% | 99.4% | 99.9% | 98.9% | 98.9% | 99.0% | 99.2% | 1000 |
| rectangle_2denv4 | `./Experiments/3dshape_2d/rectangle_2denv4/latest.pt` | 4590 | 1h 16m 30s | 97.8% | 97.8% | 99.6% | 99.0% | 99.5% | 99.6% | 98.2% | 1000 |
| Lshape3d_2denv4 | `./Experiments/3dshape_2d/Lshape3d_2denv4/latest.pt` | 4582 | 1h 16m 22s | 97.4% | 97.8% | 99.2% | 98.1% | 99.5% | 99.8% | 99.8% | 1000 |
| Fshape3d_2denv4 | `./Experiments/3dshape_2d/Fshape3d_2denv4/latest.pt` | 4719 | 1h 18m 39s | 99.1% | 98.9% | 99.8% | 99.4% | 99.7% | 99.7% | 99.4% | 1000 |
| Ashape3d_2denv4 | `./Experiments/3dshape_2d/Ashape3d_2denv4/latest.pt` | 4695 | 1h 18m 15s | 94.9% | 93.9% | 99.4% | 94.5% | 96.4% | 96.2% | 97.7% | 1000 |
| Vshape3d_2denv4 | `./Experiments/3dshape_2d/Vshape3d_2denv4/latest.pt` | 3902 | 1h 05m 02s | 99.8% | 99.4% | 100.0% | 99.8% | 99.7% | 99.7% | 99.5% | 1000 |
| 4shape3d_2denv4 | `./Experiments/3dshape_2d/4shape3d_2denv4/latest.pt` | 3981 | 1h 06m 21s | 99.4% | 98.8% | 99.9% | 99.1% | 99.6% | 99.5% | 99.3% | 1000 |
| Tshape3d_2denv4 | `./Experiments/3dshape_2d/Tshape3d_2denv4/latest.pt` | 3990 | 1h 06m 30s | 98.1% | 97.9% | 99.5% | 98.2% | 99.5% | 99.5% | 99.0% | 1000 |
