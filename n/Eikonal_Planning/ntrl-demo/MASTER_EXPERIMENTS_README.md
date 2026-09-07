# Master Experiments README

Status board for every (method, environment, shape) cell of the shape-planning
benchmark. Three charts -- **Preprocessing**, **Training**, **Evaluation** --
each split into a 3-D section and a 2-D section, with one row per method and one
column per environment/shape pair.

## Legend

| Symbol | Meaning |
| --- | --- |
| ✅ | done, artefact on disk and numbers recorded in the per-method table |
| 🟡 | running right now |
| 🕒 | queued behind a running job in the same sweep |
| ⬜ | not started |
| — | not applicable (the method has no such stage) |

## Methods

| Row | What it is | Where it lives |
| --- | --- | --- |
| **Our method** | current anisotropic metric network (`models/metric`) + MPPI controller | `ntrl-demo/ntrl-demo` |
| **Metric NTFields** | `models/metric` at the June-3 hyperparameters (0.2 output scale, unscaled Fourier `B`, normal loss on) | `baselines/ntrl-demo` |
| **NTFields** | stock isotropic NTFields, no normal loss | `baselines/NTFields_minimal_working` |
| **MPNet** | sampling-based neural planner baseline | not yet in the tree |
| **RRT-Connect** | OMPL RRT-Connect in SE(3) | `baselines/baseline_ompl/rrt_connect_eval.py` |
| **Lazy PRM** | OMPL LazyPRM in SE(3) | `baselines/baseline_ompl/lazy_prm_eval.py` |

Shapes are `rect`(angle), `L`, `F`, `A`, `V`, `4` and -- planar only -- `T`;
on disk they are `rectangle`, `Lshape3d`, `Fshape3d`, `Ashape3d`, `Vshape3d`,
`4shape3d`, `Tshape3d`. A cell is the dataset/checkpoint/result triple keyed
`<shape>_<env>`.

**Path convention.** Unqualified paths in every table below &mdash; `datasets/3dshape/...`,
`testing_data/3dshape/...`, `Experiments/...`, `.preplogs*/`,
`_run_*.sh` &mdash; are relative to `ntrl-demo/ntrl-demo/`, the main package. Paths
beginning `baselines/` are relative to this
directory, the repository root.

## Preprocessing

### 3-D section &mdash; SE(3), `env1`&ndash;`env4` x 6 shapes

| Method | env1<br>rect | env1<br>L | env1<br>F | env1<br>A | env1<br>V | env1<br>4 | env2<br>rect | env2<br>L | env2<br>F | env2<br>A | env2<br>V | env2<br>4 | env3<br>rect | env3<br>L | env3<br>F | env3<br>A | env3<br>V | env3<br>4 | env4<br>rect | env4<br>L | env4<br>F | env4<br>A | env4<br>V | env4<br>4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Metric NTFields** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NTFields** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MPNet** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **RRT-Connect** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **Lazy PRM** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

#### Where the 3-D preprocessing results live

Times are wall clock for the whole sweep, not per cell. Only **Our method** and
**Metric NTFields** are compared: NTFields consumes Our method's datasets
unchanged, so it has no generation time of its own, and the two OMPL rows have
no preprocessing stage at all.

| Method | Generator | Training data | Test sets | Log | Total generation time |
| --- | --- | --- | --- | --- | --- |
| **Our method** | `dataprocessing/preprocess_obj.py`, 800k pairs, `--margin 0.05 --offset 0.001` | `datasets/3dshape/<shape>_<env>/` | `testing_data/3dshape/<shape>_<env>/` (1000 pairs, `--offset 0.02`) | test sets: `.preplogs/<shape>_<env>.log`; 800k runs: **not retained** | training data: **not recorded** for all 24 cells; test sets: **601.4 s** over 24 cells (25.1 s mean) |
| **Metric NTFields** | `dataprocessing/preprocess_obj_june03.py`, 400k pairs, `--margin 0.1 --offset 0.01`, 8000 env points | `datasets/3dshape/<shape>_<env>_june03/` &mdash; only `rectangle_env1_june03` exists | shares Our method's test sets | `june03_preprocess.log` | **1300.3 s** for `rectangle_env1` (1 of 24 cells) |
| **NTFields** | none &mdash; same pipeline, same files | `datasets/3dshape/<shape>_<env>/` (Our method's) | Our method's | &mdash; | &mdash; (no separate generation) |
| **MPNet** | not in the tree | &mdash; | &mdash; | &mdash; | &mdash; |

The 3-D head-to-head is **not yet measurable**: the 24 current-pipeline 800k runs
predate the log-keeping convention (their datasets were written 2026-08-27 through
2026-09-04, per-run wall clocks lost), and the June-3 pipeline has only ever been
run on `rectangle_env1`. The one directly logged June-3-era pair is
`rectangle_env1_june03` at 1300.3 s versus `rectangle_env1_recovered_baseline` at
1584.2 s (`recovered_baseline_preprocess.log`) &mdash; both 400k, both June-3-era, so
that is a within-pipeline check, not the cross-method comparison. The 2-D sweep
below is the run designed to produce the real one, since it drives both pipelines
over the same cells and times each phase separately.

### 2-D section &mdash; SE(2), `2denv1`&ndash;`2denv4` x 7 shapes

| Method | 2d e1<br>rect | 2d e1<br>L | 2d e1<br>F | 2d e1<br>A | 2d e1<br>V | 2d e1<br>4 | 2d e1<br>T | 2d e2<br>rect | 2d e2<br>L | 2d e2<br>F | 2d e2<br>A | 2d e2<br>V | 2d e2<br>4 | 2d e2<br>T | 2d e3<br>rect | 2d e3<br>L | 2d e3<br>F | 2d e3<br>A | 2d e3<br>V | 2d e3<br>4 | 2d e3<br>T | 2d e4<br>rect | 2d e4<br>L | 2d e4<br>F | 2d e4<br>A | 2d e4<br>V | 2d e4<br>4 | 2d e4<br>T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅* |
| **Metric NTFields** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **NTFields** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **MPNet** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **RRT-Connect** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **Lazy PRM** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

\* `2d e4 T` is `datasets/3dshape/Tshape3d_env4`, built 2026-08-18 under the
pre-env-tag name with the same `--2d` 800k settings; it is on disk and usable, but
predates the timed sweep so it contributes no generation time.

#### Where the 2-D preprocessing results live

| Method | Generator | Training data | Test sets | Log | Total generation time |
| --- | --- | --- | --- | --- | --- |
| **Our method** | `preprocess_obj.py --2d`, 800k pairs, `--batch_size 500` (phase A settings, driven per-env by `_run_2denv4_preprocess.sh` / `_run_2denv1_preprocess.sh`) | `datasets/3dshape/<shape>_<env>/` | `testing_data/3dshape/<shape>_<env>/` (1000 pairs, `--offset 0.02`) | `.preplogs_2denv4/<shape>_2denv4.log`, `.preplogs_2denv1/<shape>_2denv1.log` | `2denv4`: **12571 s** wall (28447 s summed over 6 jobs, 3 GPUs), 2026-09-07; `2denv1`: running. Per-cell times in `2d_gen_times.md` |
| **Metric NTFields** | `preprocess_obj_june03.py --2d`, 400k pairs, `--margin 0.1 --offset 0.01 --batch_size 256` (phase B) | `datasets/3dshape/<shape>_<env>_june03/` | shares Our method's test sets | `.preplogs_2d/<shape>_<env>_june03.log` | **pending** &mdash; phase B has not started (it runs only after phase A completes) |
| **NTFields** | none &mdash; same pipeline, same files | Our method's | Our method's | &mdash; | &mdash; (no separate generation) |
| **MPNet** | not in the tree | &mdash; | &mdash; | &mdash; | &mdash; |

Phase A runs one job per GPU across three GPUs and phase B two jobs per GPU, so
the two totals are throughput under different packing, not single-job cost; each
per-cell log still ends with its own `Sampling done in <n>s`.

**Two bugs in the shared driver `_run_2d_preprocess.sh`**, both avoided by the
per-env scripts actually used: it maps `2denv1` to
`datasets/3dshape/2d_env1_zup.obj`, which does not exist (the mesh on disk is
`datasets/3dshape/2denv1_zup.obj`), and it does `cd /workspace/ntrl-demo/ntrl-demo`,
one level below the repo root under the README's container mount, so it exits
immediately. That failed `cd` is why three empty `<shape>_2denv4` directories
existed before the 2026-09-07 run. `_run_2denv4_preprocess.sh` and
`_run_2denv1_preprocess.sh` use the correct path and mesh name.

## Training

### 3-D section &mdash; SE(3), `env1`&ndash;`env4` x 6 shapes

| Method | env1<br>rect | env1<br>L | env1<br>F | env1<br>A | env1<br>V | env1<br>4 | env2<br>rect | env2<br>L | env2<br>F | env2<br>A | env2<br>V | env2<br>4 | env3<br>rect | env3<br>L | env3<br>F | env3<br>A | env3<br>V | env3<br>4 | env4<br>rect | env4<br>L | env4<br>F | env4<br>A | env4<br>V | env4<br>4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Metric NTFields** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NTFields** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MPNet** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **RRT-Connect** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **Lazy PRM** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

### 2-D section &mdash; SE(2), `2denv1`&ndash;`2denv4` x 7 shapes

| Method | 2d e1<br>rect | 2d e1<br>L | 2d e1<br>F | 2d e1<br>A | 2d e1<br>V | 2d e1<br>4 | 2d e1<br>T | 2d e2<br>rect | 2d e2<br>L | 2d e2<br>F | 2d e2<br>A | 2d e2<br>V | 2d e2<br>4 | 2d e2<br>T | 2d e3<br>rect | 2d e3<br>L | 2d e3<br>F | 2d e3<br>A | 2d e3<br>V | 2d e3<br>4 | 2d e3<br>T | 2d e4<br>rect | 2d e4<br>L | 2d e4<br>F | 2d e4<br>A | 2d e4<br>V | 2d e4<br>4 | 2d e4<br>T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Metric NTFields** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **NTFields** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **MPNet** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **RRT-Connect** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| **Lazy PRM** | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Evaluation

### 3-D section &mdash; SE(3), `env1`&ndash;`env4` x 6 shapes

| Method | env1<br>rect | env1<br>L | env1<br>F | env1<br>A | env1<br>V | env1<br>4 | env2<br>rect | env2<br>L | env2<br>F | env2<br>A | env2<br>V | env2<br>4 | env3<br>rect | env3<br>L | env3<br>F | env3<br>A | env3<br>V | env3<br>4 | env4<br>rect | env4<br>L | env4<br>F | env4<br>A | env4<br>V | env4<br>4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Metric NTFields** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NTFields** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MPNet** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **RRT-Connect** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Lazy PRM** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 2-D section &mdash; SE(2), `2denv1`&ndash;`2denv4` x 7 shapes

| Method | 2d e1<br>rect | 2d e1<br>L | 2d e1<br>F | 2d e1<br>A | 2d e1<br>V | 2d e1<br>4 | 2d e1<br>T | 2d e2<br>rect | 2d e2<br>L | 2d e2<br>F | 2d e2<br>A | 2d e2<br>V | 2d e2<br>4 | 2d e2<br>T | 2d e3<br>rect | 2d e3<br>L | 2d e3<br>F | 2d e3<br>A | 2d e3<br>V | 2d e3<br>4 | 2d e3<br>T | 2d e4<br>rect | 2d e4<br>L | 2d e4<br>F | 2d e4<br>A | 2d e4<br>V | 2d e4<br>4 | 2d e4<br>T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Metric NTFields** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **NTFields** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **MPNet** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **RRT-Connect** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Lazy PRM** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

## Notes on the cells

**Preprocessing.** The three learned rows share one set of datasets: the
`datasets/3dshape/<shape>_<env>` trees generated by
`dataprocessing/preprocess_obj.py`, and the matching
`testing_data/3dshape/<shape>_<env>` test sets generated once at `--offset 0.02`
so every method is scored on identical start/goal queries. Metric NTFields and
NTFields are marked ✅ because they consume those trees, not because they
generate their own. RRT-Connect and Lazy PRM plan directly against the meshes
and have no preprocessing stage; they read the shared test sets only.

**2-D preprocessing is the live front.** `_run_2denv4_preprocess.sh` finished all
six non-`T` `2denv4` cells on 2026-09-07 (3 h 29 m 31 s wall, three GPUs); `T` was
skipped because `Tshape3d_env4` already held it. `_run_2denv1_preprocess.sh` is now
running all seven `2denv1` cells with the same settings. The recovered June-3
pipeline (`<shape>_<env>_june03`, the Metric NTFields row) has not been run on any
2-D cell. **No `2denv2` or `2denv3` mesh exists yet** -- `datasets/3dshape/` holds
only `2denv1_zup.obj` and `2denv4_zup.obj` -- so those two columns cannot start
until the meshes are authored. 2-D training and evaluation remain blocked on the
`2denv1` half finishing.

**Training.** 4000 epochs per cell for all three learned rows, same datasets and
same optimizer-step budget. Our-method checkpoints are
`Experiments/3dshape/3dshape_<MM_DD_HH_MM>/latest.pt`; Metric NTFields writes
`Experiments/3dshape_metric/<shape>_<env>_<stamp>/latest.pt`; NTFields writes
`outputs/3dshape/<shape>_<env>/latest.pt`.

**Evaluation.** 1000 held-out start/goal pairs per cell. Our method reports six
success-rate variants (forward, reverse, OR, alternate, alternate-Bellman,
alternate-Bellman-horizon) from `evaluate_training_3d_batched.py`; the two
neural baselines report a single success rate plus plan time and SE(3) path
length; the OMPL rows report success rate, path time and path length. Lazy PRM
env1 has a second, higher-budget re-run in `experiments_lazyprm_env1_rerun.md` that
supersedes the env1 rows of the original sweep.

**MPNet** has no code, datasets or results anywhere in the tree yet -- every
MPNet cell in all three charts is ⬜.

## Source-of-truth tables

Per-cell numbers live in the per-method tables, all collected here at the root
next to this file; this README only tracks whether a cell exists. Each carries a
header note naming the package its unqualified paths are relative to, and the
generator that rewrites it.

| Method | Stage | File |
| --- | --- | --- |
| Our method | training + evaluation | `experiments_ours.md` |
| Metric NTFields | training + evaluation | `experiments_metric_ntfields.md` |
| NTFields | training + evaluation | `experiments_ntfields.md` |
| RRT-Connect | evaluation | `experiments_rrt_connect.md` |
| Lazy PRM | evaluation | `experiments_lazyprm.md`, `experiments_lazyprm_env1_rerun.md` (env1 re-run) |
| Our method / Metric NTFields | 2-D dataset generation time | `2d_gen_times.md` |

`experiments_metric_arm.md` came along in the move: it is `baselines/ntrl-demo`'s
`models/metric_arm` table, not one of the six rows charted above, kept here so the
three files once all named `experiments.md` no longer collide.

## Sweep drivers

| Script (under `ntrl-demo/ntrl-demo/`) | What it does |
| --- | --- |
| `_run_2d_preprocess.sh` | both 2-D pipelines (current + June-3) over all 7 shapes x `2denv4`, `2denv1` |
| `_run_2denv4_preprocess.sh` | the six non-`T` `2denv4` cells, current pipeline only &mdash; **done** 2026-09-07 |
| `_run_2denv1_preprocess.sh` | all seven `2denv1` cells, current pipeline only &mdash; **running** |
| `_run_2d_train.sh` | 2-D training sweep, 5000 epochs per cell |
| `_run_2d_eval.sh` | 2-D evaluation sweep, 1000 pairs per cell |
| `_run_bdiff.sh`, `_run_bdiff_eval.sh` | Fourier-`B`-scale ablation (`models/metric_bdiff`) |

All GPU work runs inside the `pytorchserver` container against `/workspace/ntrl-demo`.
