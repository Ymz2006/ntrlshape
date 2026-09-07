# 2-D generation times

Wall-clock dataset-generation time per (method, environment, shape) cell of the
2-D section of `MASTER_EXPERIMENTS_README.md`. One row per generating method, one
column per cell, same column order as that file's 2-D charts.

Each cell is the time to produce **both** artefacts for that pair -- the 800k-pair
training set and the 1000-pair test set (`--offset 0.02`), the latter being only
5-13 s of it. Times come from the sweep driver's `[start]`/`[ok]` lines
(`docker logs -t`). `⬜` = not generated, `🟡` = running, `—` = no separate
generation stage.

| Method | 2d e1<br>rect | 2d e1<br>L | 2d e1<br>F | 2d e1<br>A | 2d e1<br>V | 2d e1<br>4 | 2d e1<br>T | 2d e2<br>rect | 2d e2<br>L | 2d e2<br>F | 2d e2<br>A | 2d e2<br>V | 2d e2<br>4 | 2d e2<br>T | 2d e3<br>rect | 2d e3<br>L | 2d e3<br>F | 2d e3<br>A | 2d e3<br>V | 2d e3<br>4 | 2d e3<br>T | 2d e4<br>rect | 2d e4<br>L | 2d e4<br>F | 2d e4<br>A | 2d e4<br>V | 2d e4<br>4 | 2d e4<br>T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 2149 s | 3355 s | 5191 s | 7694 s | 4877 s | 5181 s | ⬜* |
| **Metric NTFields** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

\* `2d e4 T` exists on disk as `datasets/3dshape/Tshape3d_env4` (built 2026-08-18
under the pre-env-tag name, same `--2d`, 800k, `--margin 0.05 --offset 0.001`
settings), but it predates the timed sweep, so its generation time was never
recorded.

## How these were measured

- **Our method**, `2denv4`: `_run_2denv4_preprocess.sh`, six shapes, one job per
  GPU round-robin over cuda:0/1/2, 2026-09-07 03:23-06:53 UTC. Total wall time
  **12571 s** (3 h 29 m 31 s); summed job time **28447 s** (7 h 54 m 07 s).
  Per-shape breakdown, settings and output layout: `ntrl-demo/ntrl-demo/README_2denv4.md`.
- **Our method**, `2denv1`: `_run_2denv1_preprocess.sh`, all seven shapes
  (`Tshape3d` included -- it has no `2denv1` dataset yet), same settings, started
  2026-09-07 07:0x UTC. Running.
- **Metric NTFields**: phase B of `_run_2d_preprocess.sh`
  (`preprocess_obj_june03.py --2d`, 400k pairs, `--margin 0.1 --offset 0.01
  --batch_size 256`) has not been run on any 2-D cell.
- `2denv2` / `2denv3`: no mesh exists in `datasets/3dshape/`, so those 14 columns
  cannot start.

Cells are not a clean single-job benchmark: three jobs share the machine and
cuda:0/cuda:2 also carried unrelated load during the `2denv4` sweep, so each
figure is an upper bound on isolated cost. Within a sweep the ordering is still
meaningful -- cost rises with the shape's boundary-triangle count (rectangle 12,
L/V 20, T 28, F/4 36, A 40) and with how little free space the shape leaves for
the rejection sampler.
