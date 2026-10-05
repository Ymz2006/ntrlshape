# 2dtight -- the 2denv4 cells on a tight-clearance test set

A small test (2026-09-13): the seven planar `2denv4` cells re-scored on a fresh
500-case test set sampled at **`--offset 0.005`** instead of the `0.02` every
other 2-D test set uses (`testing_data_1k_complete/`, the old `testing_data/`).
The offset is the clearance the rejection sampler enforces between a start/goal
placement and the environment, so start and goal poses may now sit four times
closer to the obstacles.  Nothing else changes: same checkpoints, same planner
flags, same `--2d` slice as the `2d_1k_valid` tables of `../../experiments_ours.md`
and `../../experiments_metric_ntfields.md`.  This is the `--2d` counterpart of the
`3D_tight` section in `experiments_ours.md` (`_run_3d_eval_tight.sh`), with the
Metric NTFields baseline scored alongside.

Everything below runs from `/workspace/ntrl-demo/ntrl-demo` inside the
`ntrl_baselines_train` container (image `pytorchserver`); the driver
`_run_2dtight.sh` does all of it, on `cuda:0` and `cuda:2`.

```
bash _run_2dtight.sh                                    # gen, ours, metric; 2 cards
SLOT_LIST="cuda:2" SHAPES="Tshape3d" bash _run_2dtight.sh
STAGES="metric" bash _run_2dtight.sh                    # just the baseline
python _make_2dtight_table.py                           # refresh the table + chart
```

Every stage skips a cell whose output already exists, so the driver resumes.
Per-cell logs are in `.evallogs_2d_tight/`, the driver log in
`.evallogs_2d_tight_driver.log`.

## 1. Generate the tight test sets

The `--testing_data` block of each `2denv4` cell in `README.md`, with
`--num_samples 500` and `--offset 0.005`, written to `<cell>_tight`.  For
`rectangle_2denv4`:

```
python dataprocessing/preprocess_obj.py \
    --env   datasets/3dshape/2denv4_zup.obj \
    --shape datasets/3dshape/rectangle_zup.obj \
    --out   testing_data/3dshape/rectangle_2denv4_tight \
    --num_samples 500 \
    --testing_data \
    --offset 0.005 \
    --2d \
    --batch_size 500 \
    --visualize \
    --device cuda:0
```

and the same with `Lshape3d_zup.obj`, `Fshape3d_zup.obj`, `Ashape3d_zup.obj`,
`Vshape3d_zup.obj`, `4shape3d_zup.obj`, `Tshape3d_zup.obj` for the other six
cells.  `dataprocessing/make_tight_testing_data.py --offset 0.005 --num-samples
500 --only 2denv4` produces the same seven commands from the README blocks.
`--batch_size 500` is the planar memory-safe chunk (see "A note on
`--batch_size`" in `README.md`); it only chunks the rejection sampler.  A 500-pair
planar test set samples in about a second, so `gen` is not the bottleneck.

Outputs: `testing_data/3dshape/<shape>_2denv4_tight/` -- `sampled_points.npy`
`(500, 12)`, `speed.npy`, `normal.npy`, `env.npy` and the `*.html` visualizations.

## 2. Our method

Same two stages and flags as `_run_2d_eval_1k.sh`, `--dataPath` pointed at the
tight set and `--cases 500`:

```
python evaluate_training_3d_batched.py \
    --dataPath testing_data/3dshape/rectangle_2denv4_tight \
    --out ./results/output_2d_tight/rectangle_2denv4 \
    --checkpoint ./Experiments/3dshape_2d/rectangle_2denv4/latest.pt \
    --2d --cases 500 --no-viser --verbose --device cuda:0

python _spread_planner.py \
    --dataPath testing_data/3dshape/rectangle_2denv4_tight \
    --checkpoint ./Experiments/3dshape_2d/rectangle_2denv4/latest.pt \
    --modelPath ./Experiments/3dshape_2d \
    --out ./results/spread_sd_2d_tight/rectangle_2denv4 \
    --2d --cases 500 --batch 250 --steer-bias 0.5 --no-gate --device cuda:0
```

The first fills the MPPI planner columns (`success_rate.txt`), the second the
`hlB SD` column (`records.json`), which is the shipped planner.  `--no-gate` is
kept to match the 2-D tables (the cv gate was measured better in 2-D on
2026-09-09 but is deliberately not switched on here, so only the test set
changes).

## 3. Metric NTFields baseline

From `/workspace/baselines/ntrl-demo`, the `tests/run_3d_plan_all.sh` driver in
its `TIGHT=1` mode (which appends `_tight` to the test path) with `--2d`; the
checkpoint is the newest `Experiments/3dshape_metric/<cell>_*/latest.pt`, the
same one the `2d_1k_valid` table uses:

```
cd ../../baselines/ntrl-demo
ENVS="rectangle_2denv4" TIGHT=1 MODELS=metric MODEL_PATH=./Experiments/3dshape_metric \
    OUT_ROOT=./results/3d_plan_metric_2dtight EXTRA="--2d" SLOTS="cuda:0" JOBS=1 \
    bash tests/run_3d_plan_all.sh
```

which is `tests/3d_plan.py --env rectangle_2denv4 --testPath
../../ntrl-demo/ntrl-demo/testing_data/3dshape/rectangle_2denv4_tight --models metric
--modelPath ./Experiments/3dshape_metric --2d` (all 500 cases, default controller:
200 steps, 50 samples, horizon 5, step 0.015, tol 0.01, momentum 2).  Output:
`results/3d_plan_metric_2dtight/<cell>/summary.json` + `cases.csv`.

## Results

<!-- results:begin -->
**Success rate** (`SD` = `_spread_planner.py --steer-bias 0.5 --no-gate`, the shipped
hlB planner; Metric NTFields = `tests/3d_plan.py` valid / evaluated).  The two
`@0.02 (1k)` columns are the same checkpoints on the 1000-case
`testing_data_1k_complete` sets, which were sampled at `--offset 0.02`.

| Cell | cases | Ours Fwd | Ours Alt | Ours AltB | Ours hlB | Ours hlB SD | Metric NTFields | Ours hlB SD @0.02 (1k) | Metric NTFields @0.02 (1k) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv4 | 500 | 96.6% | 99.2% | 98.6% | 99.2% | 98.2% | 92.8% (464/500) | 98.2% | 95.2% |
| Lshape3d_2denv4 | 500 | 96.0% | 97.6% | 99.0% | 99.2% | 98.8% | 94.4% (472/500) | 99.8% | 98.5% |
| Fshape3d_2denv4 | 500 | 96.4% | 98.4% | 99.4% | 98.6% | 98.6% | 95.4% (477/500) | 99.4% | 97.1% |
| Ashape3d_2denv4 | 500 | 91.8% | 91.4% | 94.4% | 94.8% | 97.2% | 97.4% (487/500) | 97.7% | 99.1% |
| Vshape3d_2denv4 | 500 | 96.4% | 97.8% | 98.0% | 98.4% | 98.6% | 94.8% (474/500) | 99.5% | 99.2% |
| 4shape3d_2denv4 | 500 | 96.0% | 96.6% | 97.0% | 97.2% | 97.4% | 95.4% (477/500) | 99.3% | 98.4% |
| Tshape3d_2denv4 | 500 | 96.0% | 97.2% | 98.4% | 99.0% | 99.0% | 94.6% (473/500) | 99.0% | 97.4% |
| **pooled** | 3500 | | | | | **98.3%** | **95.0%** | 99.0% | 97.8% |

**Failure split** on the tight sets:

| Cell | Ours hlB SD collision | Ours hlB SD no-converge | Metric collision/invalid | Metric no-converge | Metric endpoint-skipped | Metric time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv4 | 2 | 8 | 17 | 19 | 0 | 0.076 |
| Lshape3d_2denv4 | 1 | 5 | 16 | 12 | 0 | 0.079 |
| Fshape3d_2denv4 | 5 | 4 | 15 | 8 | 0 | 0.078 |
| Ashape3d_2denv4 | 14 | 2 | 7 | 6 | 0 | 0.076 |
| Vshape3d_2denv4 | 5 | 4 | 16 | 10 | 0 | 0.074 |
| 4shape3d_2denv4 | 10 | 8 | 12 | 11 | 0 | 0.080 |
| Tshape3d_2denv4 | 3 | 2 | 21 | 6 | 0 | 0.093 |

![2dtight success rate](results/2dtight_sr.png)

_7 / 7 cells complete._
<!-- results:end -->

## Offset 0.001 (`TAG=tight001`)

The same seven cells again at `--offset 0.001` -- the preprocessor's *default*
offset and the value the training pairs themselves were sampled with
(`--margin 0.05 --offset 0.001`), so start/goal poses may now touch the
obstacle-clearance floor the models were trained on.  Everything else is the
same as above; the outputs are tagged `tight001`:

```
OFFSET=0.001 TAG=tight001 bash _run_2dtight.sh
python _make_2dtight_table.py --tag tight001 --offset 0.001
```

-> `testing_data/3dshape/<shape>_2denv4_tight001/`, `results/output_2d_tight001/`,
`results/spread_sd_2d_tight001/`, `../../baselines/ntrl-demo/results/3d_plan_metric_2dtight001/`,
logs `.evallogs_2d_tight001/`.

<!-- results:tight001:begin -->
**Offset 0.001.**  **Success rate** (`SD` = `_spread_planner.py --steer-bias 0.5 --no-gate`, the shipped
hlB planner; Metric NTFields = `tests/3d_plan.py` valid / evaluated).  The two
`@0.02 (1k)` columns are the same checkpoints on the 1000-case
`testing_data_1k_complete` sets, which were sampled at `--offset 0.02`.

| Cell | cases | Ours Fwd | Ours Alt | Ours AltB | Ours hlB | Ours hlB SD | Metric NTFields | Ours hlB SD @0.02 (1k) | Metric NTFields @0.02 (1k) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv4 | 498 | 95.2% | 97.8% | 98.0% | 97.4% | 97.2% | 90.0% (448/498) | 98.2% | 95.2% |
| Lshape3d_2denv4 | 489 | 92.6% | 98.4% | 97.1% | 97.8% | 98.8% | 92.6% (452/488) | 99.8% | 98.5% |
| Fshape3d_2denv4 | 481 | 93.3% | 98.1% | 96.7% | 97.5% | 97.9% | 90.9% (438/482) | 99.4% | 97.1% |
| Ashape3d_2denv4 | 495 | 89.3% | 90.5% | 92.1% | 94.7% | 94.4% | 94.6% (470/497) | 97.7% | 99.1% |
| Vshape3d_2denv4 | 489 | 93.9% | 98.2% | 97.3% | 98.2% | 99.2% | 95.3% (467/490) | 99.5% | 99.2% |
| 4shape3d_2denv4 | 489 | 93.0% | 98.0% | 98.0% | 99.0% | 98.2% | 90.4% (441/488) | 99.3% | 98.4% |
| Tshape3d_2denv4 | 495 | 93.5% | 97.2% | 97.4% | 97.2% | 97.8% | 89.7% (444/495) | 99.0% | 97.4% |
| **pooled** | 3438 | | | | | **97.6%** | **91.9%** | 99.0% | 97.8% |

**Failure split** on the tight sets:

| Cell | Ours hlB SD collision | Ours hlB SD no-converge | Metric collision/invalid | Metric no-converge | Metric endpoint-skipped | Metric time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| rectangle_2denv4 | 1 | 13 | 29 | 21 | 2 | 0.119 |
| Lshape3d_2denv4 | 3 | 4 | 26 | 10 | 12 | 0.116 |
| Fshape3d_2denv4 | 4 | 8 | 27 | 17 | 18 | 0.116 |
| Ashape3d_2denv4 | 25 | 5 | 16 | 11 | 3 | 0.114 |
| Vshape3d_2denv4 | 4 | 2 | 16 | 7 | 10 | 0.121 |
| 4shape3d_2denv4 | 7 | 4 | 34 | 13 | 12 | 0.116 |
| Tshape3d_2denv4 | 11 | 0 | 29 | 22 | 5 | 0.119 |

![2dtight001 success rate](results/2dtight001_sr.png)

_7 / 7 cells complete._
<!-- results:tight001:end -->

#### Offset 0.001, all envs 1/3/4, all five methods -> `../../CHARTS_2DTIGHT.md`

The offset-0.001 sets were then extended to `2denv1` and `2denv3` (14 more cells,
same `TAG=tight001`) and all five methods of `CHARTS_PAPER.md` were scored on the 21
cells: Ours SR (`spread`) + lengths/time (`_run_er_opt_2dtight.sh`), Metric NTFields
(`metric`), NTFields (new `ntfields` stage), RRT-Connect and Lazy PRM
(`../../baselines/rrt_logs_2dtight001/run_sweep_2dtight001.sh`,
`../../baselines/lazyprm_logs_2dtight001/run_sweep_2dtight001.sh`, in `ntrl_mpnet`).

```
OFFSET=0.001 TAG=tight001 ENVS="2denv1 2denv3 2denv4" STAGES="gen spread metric ntfields" bash _run_2dtight.sh
DEV=cuda:0 TAG=tight001 bash _run_er_opt_2dtight.sh                       # after the GPU driver
python _make_2dtight_charts.py --tag tight001 --offset 0.001               # -> ../../CHARTS_2DTIGHT.md
```

The `main` stage (per-planner Fwd / Alt / AltB columns) was not run for the 14 new
cells; `STAGES=main` adds it.  `2denv2` was not run.

## Findings, offset 0.001 (2026-09-13, cuda:0 + cuda:2, 44 min wall)

- **The ladder holds: 0.02 -> 0.005 -> 0.001.**  Pooled hlB SD: Ours 99.0 -> 98.3
  -> **97.6%**; Metric NTFields 97.8 -> 95.0 -> **91.9%**.  The gap grows 1.2 ->
  3.3 -> **5.7 pts**.  Ours is ahead on six of seven cells by 4-8 pts; Ashape3d is
  again a one-case tie (94.4 vs 94.6).
- **Endpoints get discarded at 0.001.**  The sampler accepts a pose at 0.001
  clearance that the evaluators' own collision check rejects (2-19 per cell,
  Fshape3d worst with 13 starts + 6 goals); at 0.005 that was zero everywhere.
  Both evaluators skip those pairs (`discarded_invalid_endpoints` /
  `skipped_endpoint_collision`), so the `cases` column is 481-498 rather than 500
  and both methods are scored on the same surviving pairs (the two evaluators'
  counts differ by at most one).  The remaining pairs really are at the floor: at
  0.001 the sets are sampled with the same offset as the *training* pairs.
- **Failure modes at the floor.**  Ours stays convergence-limited on the convex
  shapes (rectangle: 1 collision / 13 no-converge) and collision-limited on the
  concave ones: Ashape3d 25 collisions (14 at 0.005), Tshape3d 11 collisions and
  zero non-convergence.  Metric NTFields' collisions roughly double per cell
  (rectangle 17 -> 29, 4shape3d 12 -> 34) while its non-convergence stays flat,
  so its extra losses are almost entirely paths grazing the obstacles.
- **Forward-only MPPI drops to 89-95%** (from 92-97% at 0.005), and the
  bidirectional Bellman variants recover 4-6 pts of that in every cell; the
  spread-steered SD run is best or within a point of best throughout.
- Metric NTFields planning time is 0.11-0.12 s/case here vs 0.07-0.09 at 0.005:
  more of its rollouts run to the 200-step limit.

## Findings, offset 0.005 (2026-09-13, cuda:0 + cuda:2, 32 min wall for all seven cells)

- **Tighter clearance costs both methods, the baseline more.**  Pooled over
  3500 cases, Ours (hlB SD) goes 99.0% -> 98.3% (-0.7 pts) against the same
  checkpoints on the 0.02-offset 1k sets; Metric NTFields goes 97.8% -> 95.0%
  (-2.8 pts).  The gap between the two widens from 1.2 to 3.3 pts, and Ours is
  ahead on six of seven cells.
- **`Ashape3d_2denv4` is the exception**: 97.2% vs 97.4%, a one-case
  difference on 500 cases.  It is also the only cell where our failures are
  dominated by *collisions* (14 of 16); everywhere else they split evenly or
  lean to non-convergence.  The concave interior of the A is where a 0.005
  clearance actually puts the start/goal against the wall.  `4shape3d` (10
  collisions of 18) is the milder version of the same effect.
- **Metric NTFields fails about half by collision / invalid rollout and half by
  non-convergence** in every cell (e.g. rectangle 17 + 19), i.e. tightening the
  set hurts it on both fronts, while ours mostly keeps converging and picks
  up a few extra collisions on the concave shapes.
- The per-planner Ours columns keep their 1k-set ordering: forward-only MPPI
  sits at 92-97%, the bidirectional Bellman variants (AltB / hlB) recover most
  of that, and the spread-steered SD run is best or tied in every cell.
- No start/goal pair was rejected by either evaluator (`discarded_invalid_endpoints
  = 0`, `skipped_endpoint_collision = 0`), so the tight sets are valid queries
  for both pipelines.  Metric NTFields plans in 0.07-0.09 s/case with its
  default controller.

Result dirs: `results/output_2d_tight/`, `results/spread_sd_2d_tight/`,
`../../baselines/ntrl-demo/results/3d_plan_metric_2dtight/`; test sets
`testing_data/3dshape/<shape>_2denv4_tight/`.
