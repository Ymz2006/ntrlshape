# Experiments

> Collected at the repository root. Every unqualified path below (`datasets/`, `Experiments/`, `outputs/`, `results/`, `tests/`, `train/`, ...)
> is relative to `ntrl-demo/ntrl-demo/`, where this table's runs live. Regenerate with `_make_experiments_2d.py --out ../../experiments_ours.md` from that directory.

## 3-D shape task

Six shapes across four SE(3) environments plus `Lcouch_Corozal`, and the
planar `Tshape3d_env4` (superseded by the `2denv4` sweep below).

| Env | Model | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./Experiments/3dshape/3dshape_08_06_17_06/latest.pt` | 97.6% | 96.8% | 99.9% | 97.2% | 98.9% | 98.9% | 1000 |
| Lshape3d_env1 | `./Experiments/3dshape/3dshape_08_15_20_19/latest.pt` | 88.9% | 89.8% | 97.5% | 89.8% | 94.7% | 95.2% | 1000 |
| Fshape3d_env1 | `./Experiments/3dshape/3dshape_08_16_11_35/latest.pt` | 92.5% | 92.3% | 98.7% | 95.5% | 97.9% | 98.3% | 1000 |
| Ashape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_49/latest.pt` | 78.8% | 77.9% | 92.6% | 82.9% | 89.1% | 91.0% | 1000 |
| Vshape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_50/latest.pt` | 81.5% | 80.1% | 93.7% | 85.1% | 91.0% | 90.3% | 1000 |
| 4shape3d_env1 | `./Experiments/3dshape/3dshape_09_01_19_54/latest.pt` | 89.6% | 89.3% | 96.9% | 90.8% | 94.1% | 95.2% | 1000 |
| rectangle_env2 | `./Experiments/3dshape/3dshape_08_29_15_41/latest.pt` | 95.8% | 93.9% | 99.3% | 96.6% | 98.0% | 98.0% | 1000 |
| Lshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_04/latest.pt` | 79.8% | 80.6% | 93.5% | 83.0% | 87.5% | 86.3% | 998 |
| Fshape3d_env2 | `./Experiments/3dshape/3dshape_08_30_11_05/latest.pt` | 80.8% | 80.7% | 92.4% | 82.9% | 87.5% | 86.0% | 1000 |
| Ashape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_09/latest.pt` | 70.7% | 72.0% | 87.4% | 72.8% | 78.9% | 78.0% | 1000 |
| Vshape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_11/latest.pt` | 64.1% | 63.5% | 79.3% | 66.1% | 70.1% | 71.1% | 999 |
| 4shape3d_env2 | `./Experiments/3dshape/3dshape_09_04_09_13/latest.pt` | 82.6% | 82.3% | 93.9% | 84.9% | 88.8% | 87.4% | 1000 |
| rectangle_env3 | `./Experiments/3dshape/3dshape_09_04_09_14/latest.pt` | 94.9% | 95.0% | 99.1% | 97.8% | 98.3% | 98.1% | 1000 |
| Lshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_15/latest.pt` | 95.8% | 96.6% | 99.3% | 97.4% | 97.6% | 98.2% | 999 |
| Fshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_16/latest.pt` | 94.7% | 95.9% | 99.4% | 96.5% | 98.4% | 98.4% | 999 |
| Ashape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_21/latest.pt` | 93.6% | 94.2% | 98.2% | 95.4% | 97.5% | 97.2% | 999 |
| Vshape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_22/latest.pt` | 92.0% | 92.5% | 97.6% | 95.6% | 97.8% | 97.8% | 999 |
| 4shape3d_env3 | `./Experiments/3dshape/3dshape_09_04_09_23/latest.pt` | 92.8% | 94.2% | 98.7% | 96.4% | 98.1% | 98.0% | 1000 |
| rectangle_env4 | `./Experiments/3dshape/3dshape_09_04_09_26/latest.pt` | 95.2% | 97.3% | 99.2% | 97.4% | 98.2% | 97.9% | 1000 |
| Lshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_29/latest.pt` | 92.7% | 91.8% | 98.4% | 94.8% | 95.3% | 96.4% | 999 |
| Fshape3d_env4 | `./Experiments/3dshape/3dshape_09_04_09_30/latest.pt` | 91.6% | 92.0% | 98.1% | 95.2% | 95.5% | 95.4% | 999 |
| Ashape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_49/latest.pt` | 87.6% | 86.5% | 95.8% | 91.6% | 93.2% | 92.4% | 999 |
| Vshape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_52/latest.pt` | 90.0% | 88.4% | 96.6% | 91.4% | 93.9% | 94.3% | 1000 |
| 4shape3d_env4 | `./Experiments/3dshape/3dshape_09_01_20_55/latest.pt` | 92.7% | 92.4% | 98.7% | 93.8% | 95.2% | 95.6% | 1000 |
| Tshape3d_env4 (2D) | `./Experiments/3dshape/3dshape_08_19_12_31/latest.pt` | 98.8% | 98.4% | 99.6% | 99.1% | 99.1% | 99.4% | 1000 |

## 2-D shape task (`--2d`)

Seven shapes across the two planar environments -- `2denv4_zup.obj` (8 bodies) and
the denser `2d_env1_zup.obj` (12 bodies) -- each trained for 5000 epochs and scored
on 1000 held-out start/goal pairs with `evaluate_training_3d_batched.py --2d`.
The commands are in `README.md`; the sweep drivers are `_run_2d_preprocess.sh`,
`_run_2d_train.sh` and `_run_2d_eval.sh`.

Both pipelines are scored on the SAME test sets (`testing_data/3dshape/<ds>`,
generated once by the current preprocessor at `--offset 0.02`), so the two tables
differ only in how the training data was generated and which network was fit.

### Current pipeline (`preprocess_obj.py --2d` + `models/metric`)

800k training pairs, `--margin 0.05 --offset 0.001`.

| Env | Model | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv4 | `./Experiments/3dshape_2d/rectangle_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Lshape3d_2denv4 | `./Experiments/3dshape_2d/Lshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Fshape3d_2denv4 | `./Experiments/3dshape_2d/Fshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Ashape3d_2denv4 | `./Experiments/3dshape_2d/Ashape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Vshape3d_2denv4 | `./Experiments/3dshape_2d/Vshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| 4shape3d_2denv4 | `./Experiments/3dshape_2d/4shape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Tshape3d_2denv4 | `./Experiments/3dshape_2d/Tshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| rectangle_2denv1 | `./Experiments/3dshape_2d/rectangle_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Lshape3d_2denv1 | `./Experiments/3dshape_2d/Lshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Fshape3d_2denv1 | `./Experiments/3dshape_2d/Fshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Ashape3d_2denv1 | `./Experiments/3dshape_2d/Ashape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Vshape3d_2denv1 | `./Experiments/3dshape_2d/Vshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| 4shape3d_2denv1 | `./Experiments/3dshape_2d/4shape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Tshape3d_2denv1 | `./Experiments/3dshape_2d/Tshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |

### Recovered June-3 pipeline (`preprocess_obj_june03.py --2d` + `models/metric_june03`)

400k training pairs at the June-3 era's `--margin 0.1 --offset 0.01` and 8000 env
points, fit with the frozen single-route network.

| Env | Model | SR Forward | SR Reverse | SR OR | SR Alternate | SR Alternate Bellman | SR Alternate Bellman Horizon | test_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv4 | `./Experiments/3dshape_2d_june03/rectangle_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Lshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Lshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Fshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Fshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Ashape3d_2denv4 | `./Experiments/3dshape_2d_june03/Ashape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Vshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Vshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| 4shape3d_2denv4 | `./Experiments/3dshape_2d_june03/4shape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Tshape3d_2denv4 | `./Experiments/3dshape_2d_june03/Tshape3d_2denv4/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| rectangle_2denv1 | `./Experiments/3dshape_2d_june03/rectangle_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Lshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Lshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Fshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Fshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Ashape3d_2denv1 | `./Experiments/3dshape_2d_june03/Ashape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Vshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Vshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| 4shape3d_2denv1 | `./Experiments/3dshape_2d_june03/4shape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |
| Tshape3d_2denv1 | `./Experiments/3dshape_2d_june03/Tshape3d_2denv1/latest.pt` | -- | -- | -- | -- | -- | -- | -- |

### Head to head (`SR Forward`)

| Env | current | June-3 | delta |
| --- | --- | --- | --- |
| rectangle_2denv4 | -- | -- | -- |
| Lshape3d_2denv4 | -- | -- | -- |
| Fshape3d_2denv4 | -- | -- | -- |
| Ashape3d_2denv4 | -- | -- | -- |
| Vshape3d_2denv4 | -- | -- | -- |
| 4shape3d_2denv4 | -- | -- | -- |
| Tshape3d_2denv4 | -- | -- | -- |
| rectangle_2denv1 | -- | -- | -- |
| Lshape3d_2denv1 | -- | -- | -- |
| Fshape3d_2denv1 | -- | -- | -- |
| Ashape3d_2denv1 | -- | -- | -- |
| Vshape3d_2denv1 | -- | -- | -- |
| 4shape3d_2denv1 | -- | -- | -- |
| Tshape3d_2denv1 | -- | -- | -- |
