# NTFields minimal working package

This self-contained subset demonstrates NTFields training and trajectory testing,
plus a CPU Fast Marching Method (FMM) baseline, on one bundled Gibson/Aloha scene.
Run every command from this directory.

## Contents

- `train.py`: short, configurable NTFields training run on the bundled Aloha scene.
- `train_3dshape.py`: NTFields training on the ntrl-demo 3-D shape (SE(3)) datasets.
- `train_3dshape_all.sh`: that run over every 3-D shape environment.
- `3d_plan.py`: path inference with a trained 3-D shape model -- the ntrl-demo
  MPPI controller driven by NTFields' travel-time field, reporting per-case
  planning time and path length.
- `3d_plan_all.sh`: that inference over every trained environment.
- `make_experiments_md.py`: collects the 3-D shape training *and* path-inference
  runs into `experiments.md`.
- `test.py`: trajectory generation with the bundled pretrained model.
- `fmm.py`: CPU FMM trajectory generation on a mesh-derived speed grid.
- `models/`: NTFields network and data loader from the working repository, with
  the small generalizations listed under "3-D shape datasets" below.
- `data/Aloha/`: only the arrays, mesh, and queries required by these examples.
- `checkpoints/`: model-size-2 checkpoint used by `test.py`.

The 50 GB experiment history, unrelated scenes, cached files, plots, editor files,
and generated evaluation results are intentionally excluded.

## Installation

The reference setup follows the original project (Python 3.9, PyTorch 1.10,
CUDA 11.3):

```bash
conda env create -f environment.yml
conda activate ntfields-minimal
```

If you already have a suitable PyTorch environment, this lighter option is also
available:

```bash
python -m pip install -r requirements.txt
```

PyTorch/CUDA packages are platform-specific. If pip cannot select the right
build, install PyTorch for your system first, then install the other requirements.

## Quick start

Test one trajectory with the pretrained model (GPU is used when available):

```bash
python test.py --num-trajectories 1
```

Run a one-epoch small-model training smoke test:

```bash
python train.py --epochs 1 --model-size 0 --batch-size 256
```

Reproduce the bundled large-model training schedule (about 100 optimizer steps
per epoch on the bundled 200,000 pairs):

```bash
python train.py --device cuda:0 --epochs 2000 --model-size 2 \
  --batch-size 2000 --max-batches 0 --seed 0 --save-every 500 \
  --print-every 50 \
  --output outputs/reproduction
```

`--max-batches 6` is the fast/smoke mode. It performs only 12,000 optimizer
steps over 2,000 epochs and therefore cannot reproduce the bundled checkpoint,
whose optimizer records 199,300 steps. `--max-batches 0` consumes the complete
dataloader and reports the mean over the batches actually processed.

This exact command was validated in the `fbntfields:demo` Docker environment.
The reproduced seed-0 checkpoint reached loss `0.019659` at epoch 2000 versus
`0.018998` for the historical checkpoint. On 100 bundled Aloha start/goal pairs,
it converged 100/100 times and produced 75/100 collision-free paths at a 0.06 m
clearance threshold (the historical checkpoint produced 73/100).

Run one FMM query on a compact `41^3` CPU grid:

```bash
python fmm.py --num-trajectories 1 --resolution 41
```

Results are written under `outputs/`. Use `--help` on any entry point to see
paths, device selection, trajectory count, resolution, and other controls.

## 3-D shape datasets (SE(3))

The bundled Aloha example plans for a point in a 3-D workspace (dim = 3). The
ntrl-demo 3-D shape task plans a rigid body over SE(3) (dim = 6, configurations
stored as `(x, y, z, rx, ry, rz)` with the rotation vector normalized by 2*pi).
NTFields' loss is isotropic and dimension-agnostic and needs only
`sampled_points.npy` (N, 2*dim) and `speed.npy` (N, 2) -- the `normal.npy` that
the ntrl-demo preprocessor also writes is unused -- so the same network trains
on those datasets directly. The container must mount the repository root so
that both `baselines/` and the datasets under `ntrl-demo/` are visible.

```bash
python train_3dshape.py --env rectangle_env1 --device cuda:0        # one env
bash train_3dshape_all.sh                                           # all 27 envs
python make_experiments_md.py                                       # write experiments.md
```

Runs land in `outputs/3dshape/<env>/` (final checkpoint copied to `latest.pt`),
logs in `outputs/3dshape/logs/`. `train_3dshape_all.sh` spreads jobs round-robin
over the GPU slots in `$SLOTS` and skips any environment that already has an
epoch-`$EPOCHS` checkpoint, so it is re-runnable after an interruption.

The defaults are chosen to match the ntrl-demo baselines rather than the Aloha
recipe: 4000 epochs of `--max-batches 5` at `--batch-size 2000` is the same
20,000 optimizer steps at the same batch size that `models/metric` and
`models/metric_arm` get, so the methods are comparable per training budget.
Losses are *not* comparable across those tables -- the objectives differ.

Four changes to `models/` were needed. The defaults keep the Aloha recipe
behaving as before, though the sampling change preserves the *distribution*
rather than the RNG stream, so an Aloha rerun is statistically but not
bit-identical to the numbers quoted above:

- **Epoch-acceptance guard.** NTFields replays an epoch whose mean loss exceeds
  1.2x the previous epoch's, measured against a `prev_diff` initialized to the
  constant `1.0`. The 6-D shape data starts near 2.0, so epoch 1 was rejected
  forever (1618 replays without reaching epoch 2). The threshold is now
  `Repeat Ratio`, with `Max Repeats Per Epoch` as a cap that accepts an epoch
  after that many replays; `train_3dshape.py` defaults to 2.5 and 20. Aloha's
  epoch 1 is already below 1.2, so its behaviour is unchanged. Every 3-D shape
  run needed exactly one replay, all at epoch 1.
- **Configuration scale.** `models/database.py` hardcoded `points /= 10.0`,
  which maps Aloha's raw metres into [-0.5, 0.5]. It is now a `scale`
  parameter (`Data Scale`, default 10.0); the shape datasets are already stored
  in that range and pass `--data-scale 1.0`.
- **Resampling weights.** The distance driving the weighted sampler was taken
  from the hardcoded columns `[:,0:3]` and `[:,3:6]`, which at dim = 6 compares
  a translation against a rotation vector. It now uses `self.dim`.
- **Sampling speed.** `WeightedRandomSampler` + `DataLoader` gathered one row at
  a time in Python, which dominated the runtime on the 800k-row shape datasets.
  A single multinomial draw with replacement over the resident tensor is the
  same distribution and roughly 17x faster (150 epochs: 400 s to 33 s).

`train_3dshape.py --threads` caps torch's intra-op threads (default 4). An epoch
is five small batches, so the run is launch-bound rather than compute-bound, and
leaving torch's default of one thread per core makes concurrent runs thrash: six
jobs on a 36-core host went from 0.15 s/epoch solo to 0.48 s/epoch.

## Path inference on the 3-D shape models (SE(3))

`3d_plan.py` takes one of those checkpoints and plans the 1000 start/goal pairs
of the matching test set under
`../../ntrl-demo/ntrl-demo/testing_data/3dshape/<env>/sampled_points.npy`
(an `(N, 12)` array, columns 0-5 the start config and 6-11 the goal). It reports
per-case wall-clock planning time and path length, mean ± sd, and writes them
into `experiments.md` via `make_experiments_md.py`.

```bash
python 3d_plan.py --env rectangle_env1 --device cuda:0     # one env, all 1000 cases
python 3d_plan.py --env rectangle_env1 --cases 20          # smoke test
bash 3d_plan_all.sh                                        # all trained envs
python make_experiments_md.py                              # fold into experiments.md
```

Each run writes `outputs/3dplan/<env>/plan_summary.txt` (the aggregate) and
`plan_cases.csv` (one row per case: time, both path lengths, waypoint count,
convergence, collision, closest approach to the goal), with logs in
`outputs/3dplan/logs/`. `3d_plan_all.sh` spreads envs round-robin over `$SLOTS`
and skips any that already has a `plan_summary.txt`, so it is re-runnable
(`FORCE=1` re-plans anyway).

### The controller

The controller is the MPPI rollout of `ntrl-demo/evaluate_training_3d.py`, so
NTFields is driven exactly the way the ntrl-demo planner is. Three things
changed:

- **Cost-to-go source.** `models.model_3d.Model.TravelTimes` replaces
  `models.metric.model_train_metric.Model.function.TravelTimes`. Both map
  `(N, 2*dim) -> (N,)`, so nothing else about the sampler moves: 50 samples, a
  horizon of 5, a 0.015 per-sample displacement cap, momentum 2.0 on the last
  accepted step, cost `10*tau(first) + tau(last)`, a `softmax(-50*cost)`-weighted
  first step, and a 0.01 convergence ball, capped at 200 iterations.
- **Batched over episodes.** The rollout carries a leading batch axis, so
  `--batch B` runs `B` pairs in one set of GPU launches. Only `--batch 1` (the
  default) yields true per-case wall clock; a larger batch shares the chunk time
  out evenly and the summary says so.
- **Device.** A parameter rather than a hardcoded `.cuda()`.

NTFields plans directly in the frame the datasets are stored in -- translations
in units of `meta.env_scale`, rotation vector divided by 2*pi -- so unlike the
bundled Aloha example (`test.py`, which divides by 10 and multiplies back) there
is no rescaling. Test sets generated by `preprocess_obj.py --2d` (`Tshape3d_env4`)
are planned in the planar sub-space `(x, y, rz)`, read from `meta.json`.

### Metrics

A case **succeeds** when the rollout reaches the 0.01 goal ball *and* the placed
shape is collision-free at every recorded waypoint -- the same two-part test
`evaluate_training_3d.py` applies. Collision is `preprocess_obj`'s point-in-tet
test (an environment surface point inside the shape's tetrahedral decomposition)
against a 50,000-point sampling of the *whole* environment mesh, walls included,
with the KD-tree broad phase of `baseline_ompl/rrt_connect_eval.py`. Waypoints
are checked individually with no interpolation between them, which the 0.015
stride cap already keeps dense.

`Path Time` is the mean ± sd over **all** scored cases; `Path Length` is the
mean ± sd over the **successful** ones. That is what
`ntrl-demo/RRT_experiments.md` reports, and the length uses the same metric:
OMPL's SE(3) distance summed along the polyline,
`sum ||dt|| + acos(|q_i . q_i+1|)`, in the normalized frame. Comparing the two
tables, note that MPPI emits one waypoint per iteration -- tens to hundreds of
0.015-long steps whose noise the length integrates -- while RRT-Connect emits a
handful of long straight segments, so the learned planner's length is inflated
relative to a shortcut/simplified path. `plan_cases.csv` also records
`cfg_length`, the plain Euclidean length in the normalized 6-D configuration
space the network actually plans in.

The first CUDA launch in a process pays context creation and kernel autotuning,
which would otherwise land entirely on case 0 -- an order of magnitude over its
true cost -- so the run does a short warm-up rollout before timing anything.

### Results

All 27 checkpoints have been run over the full 1000 pairs of their test set;
`experiments.md` holds the per-environment table. Across the 27,000 cases:

- **86.8% success** (23,424 / 27,000), per-environment 60.0% -- 97.2%.
- **0.112 s** mean per-case planning time (per-env means 0.094 -- 0.166 s).
- **1.810** mean path length (per-env means 1.517 -- 1.981).

The whole sweep took 935 s wall on three RTX 3090s, six jobs at a time.

Failures are dominated by collision, not by the controller: 2,935 collisions
against 738 non-convergences (a case can be both). The split moves with the
environment, and it is the more informative signal:

| Env group | Mean success | Character of the failures |
| --- | --- | --- |
| `env3` (6) | 96.2% | almost purely non-convergence -- 5-25 collisions per 1000 |
| `env4` (7) | 92.2% | mixed |
| `env1` (7) | 83.4% | collision-dominated |
| `env2` (6) | 79.4% | collision-dominated -- up to 318 per 1000 |
| `Lcouch_Corozal` | 60.0% | both at once (152 non-converged, 297 collisions) |

Where the field is well shaped (`env3`) the residual failures are MPPI not
sticking the 0.01 landing rather than the field pointing into an obstacle. Where
it is not (`env2`, and the scanned `Corozal` scene), the rollout is steered
through geometry. Within an environment, difficulty tracks shape concavity
almost monotonically: rectangle > F ~ 4 > L > A ~ V.

For scale, `ntrl-demo/RRT_experiments.md` covers 24 of these test sets (all but
`Tshape3d_env4`, `Tshape3d_env1` and `Lcouch_Corozal`). Restricted to those 24,
so the comparison is like for like:

| | Success | Path time | Path length |
| --- | --- | --- | --- |
| NTFields + MPPI | 89.0% | 0.108 s | 1.814 |
| RRT-Connect (OMPL) | 99.9 -- 100% | 1.008 s | 2.743 |

NTFields plans about 9.4x faster and reports a shorter path, but gives up 11
points of success rate -- and the length advantage is partly an artifact of the
metric note above rather than a genuinely better path.

## Notes

- `test.py` can run on CPU, but CUDA is much faster.
- `train.py` defaults to a deliberately short six-batch smoke mode, not a
  converged model. Use the reproduction command above for real training.
- FMM grid construction uses `libigl`; FMM propagation uses `pykonal`.
  Increase `--resolution` for a finer baseline at greater memory/time cost.
- The bundled checkpoint is the independently reproduced seed-0, 2000-epoch
  model generated with the command above. The historical checkpoint remains in
  `Experiments/ntfields_for_fbntfields_baseline_used/Aloha/` in the source tree.
- This package snapshots the working tree based on Git commit
  `f231c01797a166cf063140768303da1a98e25779`; it intentionally includes the
  source tree's relevant uncommitted research changes.
- `README_ORIGINAL.md` contains the original repository documentation and citation.
