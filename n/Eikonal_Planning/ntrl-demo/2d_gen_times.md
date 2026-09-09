# 2-D generation times

Wall-clock dataset-generation time per (method, environment, shape) cell of the
2-D section of `MASTER_EXPERIMENTS_README.md`. One row per generating method, one
column per cell, same column order as that file's 2-D charts.

Each cell is the time to produce **both** artefacts for that pair -- the 800k-pair
training set and the 1000-pair test set (`--offset 0.02`), the latter being only
5-13 s of it. Times come from the sweep driver's `[start]`/`[ok]` lines
(`docker logs -t`). `⬜` = not generated, `—` = no separate generation stage.

**All 28 cells of the Our-method row are generated** as of 2026-09-08 21:43 UTC;
only `2d e4 T` carries no time, for the reason footnoted below.

Test sets exist for every 2-D cell in **both** variants: the standard 1000-pair set
at `--offset 0.02` (generated alongside each training set) and the `_tight` set at
`--offset 0.005`, built afterwards by
`dataprocessing/make_tight_testing_data.py --batch-size 500`, which reads the
`--testing_data` blocks straight out of `README.md`. Both are verified
`(1000, 12)`, no NaNs, `z`/`rx`/`ry` = 0. The tight sets carry no generation times
of their own -- at 1000 pairs each they are seconds to minutes, not hours.

| Method | 2d e1<br>rect | 2d e1<br>L | 2d e1<br>F | 2d e1<br>A | 2d e1<br>V | 2d e1<br>4 | 2d e1<br>T | 2d e2<br>rect | 2d e2<br>L | 2d e2<br>F | 2d e2<br>A | 2d e2<br>V | 2d e2<br>4 | 2d e2<br>T | 2d e3<br>rect | 2d e3<br>L | 2d e3<br>F | 2d e3<br>A | 2d e3<br>V | 2d e3<br>4 | 2d e3<br>T | 2d e4<br>rect | 2d e4<br>L | 2d e4<br>F | 2d e4<br>A | 2d e4<br>V | 2d e4<br>4 | 2d e4<br>T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Our method** | 3278 s | 5094 s | 8263 s | 11900 s | 7574 s | 8382 s | 16486 s | 3082 s | 4138 s | 7504 s | 18298 s | 8500 s | 7866 s | 15271 s | 1287 s | 2142 s | 3012 s | 3910 s | 2669 s | 3087 s | 4400 s | 2149 s | 3355 s | 5191 s | 7694 s | 4877 s | 5181 s | ⬜* |
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
  (`Tshape3d` included -- it had no `2denv1` dataset), same settings. Ran in two
  parts: 2026-09-07 13:23-17:49 UTC produced `F`, `4`, `A`, `L`, `V`, then the host
  rebooted at 20:34 UTC and killed the container (exit 255, not OOM) with
  `Tshape3d` at 581475/800000 and `rectangle` not yet started; those two were
  re-run from scratch 2026-09-08 06:07-10:42 UTC
  (`SHAPES="Tshape3d rectangle"`). Summed job time over the seven
  cells: **60977 s** (16 h 56 m 17 s).
- **Our method**, `2denv2`: `_run_2denv2_preprocess.sh`, 2026-09-08 06:24-20:51 UTC.
  The mesh (`2denv2.obj`, 13 bodies, 350 x 30 x 350) was rotated to
  `2denv2_zup.obj` first. Ran in two phases: `A`, `4`, `F` strictly serial on
  cuda:2 (`GPU_LIST="cuda:2" NWORKERS=1`) while the `2denv1` re-run held the other
  two cards, then `T`, `V`, `L`, `rect` across all three once those freed up.
  Summed job time **64659 s** (17 h 57 m 39 s).
- **Our method**, `2denv3`: `_run_2denv3_preprocess.sh` with
  `GPU_LIST="cuda:1 cuda:2" NWORKERS=2`, started 2026-09-08 18:42 UTC, and the
  only sweep run at **`--batch_size 1000`** rather than 500 (`BATCH` env var).
  `2denv3.obj` (6 bodies) was authored 2026-09-08 and rotated to
  `2denv3_zup.obj`. Peak memory at the larger batch measured 16.1 GiB on
  `Ashape3d` and 12.5 GiB on `4shape3d`, both inside a 24 GiB card, and no cell
  OOMed. Finished 21:43 UTC: **10879 s** wall on two cards, **20507 s**
  (5 h 41 m 47 s) summed over the seven cells -- the cheapest environment of the
  four by a factor of three.
- **Metric NTFields**: phase B of `_run_2d_preprocess.sh`
  (`preprocess_obj_june03.py --2d`, 400k pairs, `--margin 0.1 --offset 0.01
  --batch_size 256`) has not been run on any 2-D cell.

Cells are not a clean single-job benchmark: up to three jobs share the machine,
`2denv3` ran at a different batch size, and the cards carried unrelated load
throughout, so each figure is an upper bound on isolated cost. Within a sweep the ordering is still
meaningful -- cost rises with the shape's boundary-triangle count (rectangle 12,
L/V 20, T 28, F/4 36, A 40) and with how little free space the shape leaves for
the rejection sampler.

Environment density dominates shape cost, and it dominates it far more than the
shape's own triangle count does. Every `2denv1` cell (12-body maze) is 1.5-2.2x its
`2denv4` twin (8-body), `2denv2` (13-body) is dearer still, and `2denv3` (6-body) is
the cheapest of the four by a wide margin. `Tshape3d` shows the effect most sharply:
by triangle count it sits mid-pack, yet it is the most expensive cell in both dense
envs (16486 s on `2denv1`, 15271 s on `2denv2`) and an unremarkable 4400 s on the
sparse `2denv3` -- 3.5x cheaper. What is being measured is the rejection sampler's
acceptance rate, i.e. how much collision-free room the shape has, not the cost of a
single clearance query.

Totals per environment, summed over the seven cells: `2denv3` **20507 s**,
`2denv4` **28447 s** (six cells), `2denv1` **60977 s**, `2denv2` **64659 s**.
