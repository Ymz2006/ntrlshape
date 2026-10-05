# `2denv4` datasets -- current ("my method") pipeline

Data generation run for the six `<shape>_2denv4` pairs, produced with the **current**
generator `dataprocessing/preprocess_obj.py --2d` (not the June-3 recovered generator,
which writes `<shape>_2denv4_june03` and is documented in `README.md`).

Generated 2026-09-07 by `_run_2denv4_preprocess.sh`.

## Scope

| shape | shape mesh | boundary triangles | included |
| --- | --- | --- | --- |
| `Ashape3d`  | `Ashape3d_zup.obj`  | 40 | yes |
| `4shape3d`  | `4shape3d_zup.obj`  | 36 | yes |
| `Fshape3d`  | `Fshape3d_zup.obj`  | 36 | yes |
| `Vshape3d`  | `Vshape3d_zup.obj`  | 20 | yes |
| `Lshape3d`  | `Lshape3d_zup.obj`  | 20 | yes |
| `rectangle` | `rectangle_zup.obj` | 12 | yes |
| `Tshape3d`  | `Tshape3d_zup.obj`  | -- | **skipped** -- already generated as `datasets/3dshape/Tshape3d_env4` (legacy name, same settings) |

Environment: `datasets/3dshape/2denv4_zup.obj` -- the 8-body 350 x 350 maze, z-up.

## Commands

Per shape, the two README calls (training set, then the held-out test set):

```
python dataprocessing/preprocess_obj.py \
    --env   datasets/3dshape/2denv4_zup.obj \
    --shape datasets/3dshape/<shape>_zup.obj \
    --out   datasets/3dshape/<shape>_2denv4 \
    --num_samples 800000 \
    --2d --visualize \
    --batch_size 500 \
    --device <cuda:N>

python dataprocessing/preprocess_obj.py \
    --env   datasets/3dshape/2denv4_zup.obj \
    --shape datasets/3dshape/<shape>_zup.obj \
    --out   testing_data/3dshape/<shape>_2denv4 \
    --num_samples 1000 \
    --testing_data --offset 0.02 \
    --2d --visualize \
    --batch_size 500 \
    --device <cuda:N>
```

Driver (all six, round-robin over cuda:0/1/2, one job per GPU):

```
docker run -d --name ntrl_prep_2denv4 \
  --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo/ntrl-demo:/workspace" \
  --volume="/usr/lib/x86_64-linux-gnu/:/glu" \
  --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/.local:/.local" \
  --gpus all pytorchserver \
  bash /workspace/ntrl-demo/_run_2denv4_preprocess.sh
```

Per-shape logs land in `.preplogs_2denv4/<shape>_2denv4.log`.

## Settings actually used (from `meta.json`)

| key | training set | test set |
| --- | --- | --- |
| `num_samples` | 800000 | 1000 |
| `margin` | 0.05 | 0.05 |
| `offset` | 0.001 | 0.02 |
| `two_d` | true | true |
| `yrot` | false | false |
| `env_scale` | 350.0 | 350.0 |
| `shape_z_thickness` | 0.01 | 0.01 |
| `rot_norm` | 2*pi | 2*pi |
| `--batch_size` | 500 | 500 |

`--batch_size 500` rather than the 3-D default 2000: flattening the environment onto
`z = 0` defeats the broad-phase env cull in `evaluate_placements`, so the clearance
tensor stays dense and 2000 OOMs a 24 GiB card. Batch size only chunks the rejection
sampler; it does not change the sampled distribution.

## Generation times

Wall clock, measured from the driver's `[start]` / `[ok]` lines (`docker logs -t`).
Each figure covers the 800k training set **and** the 1k test set for that shape; the
test set is only 5--13 s of it (visible as the gap between the `meta.json` mtimes in
`datasets/` and `testing_data/`).

| shape | GPU | round | start (UTC) | end (UTC) | wall | mean samples/s |
| --- | --- | --- | --- | --- | --- | --- |
| `Ashape3d`  | cuda:0 | 1 | 03:23:24 | 05:31:38 | **2 h 08 m 14 s** (7694 s) | 104 |
| `Fshape3d`  | cuda:2 | 1 | 03:23:24 | 04:49:55 | **1 h 26 m 31 s** (5191 s) | 154 |
| `4shape3d`  | cuda:1 | 1 | 03:23:24 | 04:49:45 | **1 h 26 m 21 s** (5181 s) | 154 |
| `Vshape3d`  | cuda:0 | 2 | 05:31:38 | 06:52:55 | **1 h 21 m 17 s** (4877 s) | 164 |
| `Lshape3d`  | cuda:1 | 2 | 04:49:45 | 05:45:39 | **0 h 55 m 55 s** (3355 s) | 238 |
| `rectangle` | cuda:2 | 2 | 04:49:55 | 05:25:44 | **0 h 35 m 49 s** (2149 s) | 372 |

- **Total wall time: 3 h 29 m 31 s** (12571 s) for all six, three GPUs in parallel.
- **Total GPU-job time: 7 h 54 m 07 s** (28447 s) summed over the six jobs.

Cost tracks boundary-triangle count (the per-pair clearance query is
`(B, kept env points, F)`) but not exactly -- `Vshape3d` and `Lshape3d` have the same
20 triangles yet differ by 25 min, because the rejection sampler's acceptance rate also
depends on how much free space the shape leaves in the maze. cuda:0 and cuda:2 were
also sharing the card with unrelated jobs during this run, so the per-shape numbers are
an upper bound rather than a clean benchmark.

## Output

```
datasets/3dshape/<shape>_2denv4/          ~196 MB   training set, 800k pairs
    sampled_points.npy   (800000, 12)     env.npy          (10000, 3)
    speed.npy            (800000, 2)      normal.npy       (800000, 12)
    speed_dists.npy      (800000, 2)      speed_angles.npy (800000, 2)
    trans_n.npy / rot_n.npy  (800000, 12)  meta.json
    sampled_placements.html + speed*_distribution.html   (--visualize)

testing_data/3dshape/<shape>_2denv4/      ~600 KB   1k held-out start/goal pairs
    same layout at N = 1000
```

Each row is a correlated pair `(x0, x1)` in the 6-D SE(3) layout, so `sampled_points`
is `(N, 12)`. Verified after the run: all six train sets are `(800000, 12)`, all six test
sets `(1000, 12)`, no NaNs, and `z`/`rx`/`ry` are exactly 0 everywhere (the `--2d` slice
held). No tracebacks in `.preplogs_2denv4/`. Files are written by the container as **root** -- `sudo chown` them
if you need host-side writes.

## Next steps

```
python train/train_3dshape.py --dataPath datasets/3dshape/<shape>_2denv4 \
   --modelPath ./Experiments/3dshape_2d --name <shape>_2denv4

python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/<shape>_2denv4 \
   --out ./results/output_3d/<shape>_2denv4 \
   --checkpoint ./Experiments/3dshape_2d/<shape>_2denv4/latest.pt \
   --2d
```

The test sets here are the shared ones: if the June-3 pipeline is run on `2denv4`, it
reuses `testing_data/3dshape/<shape>_2denv4` so both pipelines are scored on exactly the
same start/goal queries.

## Note on the repo-wide driver

`_run_2d_preprocess.sh` does `cd /workspace/ntrl-demo/ntrl-demo`, but with the README's
`docker run` mount the repo root inside the container is `/workspace/ntrl-demo`. That
`cd` fails, and the earlier aborted run (`_2d_preprocess_driver.log`) is why three empty
`<shape>_2denv4` directories existed before this run. `_run_2denv4_preprocess.sh` uses
the correct path.
