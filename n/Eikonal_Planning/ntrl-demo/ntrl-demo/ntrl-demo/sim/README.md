# Push task -- pushing a planar shape along the planned path

A closed-loop pushing simulation built on the planar (`2denv*`) planners.  The
Eikonal planner produces an SE(2) path for the shape itself; this sim closes the
gap to a robot by driving a single cylindrical pusher (pymunk, top-down physics)
so that the shape follows that path, replanning from the shape's measured pose
after every push.  All commands run from the nested `ntrl-demo` root (the one
containing `models/`, `datasets/`, `sim/`) inside the `pytorchserver` container.

Files:

| File | What |
| --- | --- |
| `push_t_demo.py` | One case, interactive (viser on `:8080`) or `--headless`.  Planning, push primitives, controller, replanner. |
| `push_batch.py` | Many cases per (shape, env) in one process; writes per-case records for the table below. |
| `pymunk_viser_push.py` | The bare physics (`Sim`), mesh -> footprint helpers, and a mouse-driven demo. |
| `../_make_charts_pushtask.py` | Turns `results/push_task/*.json` into `Eikonal_Planning/ntrl-demo/CHARTS_PUSHTASK.md`. |
| `viz_normals.py` | Debug view of the training normals. |

## Running one case

```
python sim/push_t_demo.py --shape Ashape3d --env 2denv4 --case 3 --teleport --spacing 1 --push-len 5
python sim/push_t_demo.py --shape Lshape3d --env 2denv2 --case 7 --headless      # no viewer, prints a report
python sim/push_t_demo.py --solid-env                                              # walls/blocks collide again
```

`--shape` and `--env` are **names**, and everything else is derived from the pair
(`resolve_paths` in `push_t_demo.py`):

- meshes: `datasets/3dshape/<name>.obj` (y-up, for the sim) and `<shape>_zup.obj`
  (its bounding box recovers the origin the planner poses the shape about).  A
  path ending in `.obj` is accepted for either.
- test set: `testing_data_1k_complete/3dshape/<shape>_<env>` (`--case k` is the
  k-th start/goal pair; override the set with `--dataPath`, the root with
  `--testing-root`).
- checkpoint (`find_checkpoint`): `Experiments/3dshape_2d/<shape>_<env>/latest.pt`
  for the planar cells, then `Experiments/3dshape/<shape>_<env>/`, then the
  timestamped SE(3) mapping in `_make_experiments_3d.py`.  Override with `--ckpt`.

Shapes: `rectangle Lshape3d Fshape3d Ashape3d Vshape3d 4shape3d Tshape3d`; envs:
`2denv1 2denv2 2denv3 2denv4`.  The SE(3) envs (`env1..4`) resolve to a checkpoint
but have no `_zup` twin and non-planar test sets, so the demo stops at the mesh
check for them.

**The environment is scenery by default.**  Walls and blocks are drawn and used by
the pusher's transit planner, but nothing collides with them in the physics
(`Sim.set_env_solid(False)` -- a pymunk `ShapeFilter` with empty categories/mask).
The planned path is what keeps the shape clear; a solid wall only added the pinch
that ejects a shape squeezed between it and the infinite-mass pusher.  `--solid-env`
restores physical contact.

Other knobs that matter for the benchmark: `--spacing` (reference resample step,
world units), `--push-len` (units per push primitive), `--goal-dist 6 --goal-deg 6`
(goal test), `--max-seconds` (simulated-time cap, 120 s in headless mode),
`--plan-device`.

### What one run does

1. **Plan** -- planar MPPI on the travel-time field from the test pair, lifted from the
   planner's normalized frame to world units (`planner_to_world`), resampled at
   `--spacing`.
2. **Primitives** -- an action is (contact point, push direction, push length).  The
   10 000-action library (100 contacts x 100 directions) is measured once per shape and
   push length by rolling each one out in an empty arena, and cached next to the test
   sets as `.push_primitives_<hash>.npy` (~30 s per shape).
3. **Loop** -- per action: replan from the shape's measured pose, pick the action whose
   measured outcome best matches the reference `--action-steps` ahead (after screening
   out entry points the pusher cannot occupy), teleport the pusher to its entry point
   (`--teleport`; `--teleport-interp` flies it instead), push, repeat until the goal
   test passes.

## Running the benchmark

```
python sim/push_batch.py --envs 2denv1 2denv3 2denv4 --shapes rectangle Lshape3d Ashape3d \
    --cases 100 --out results/push_task -- --teleport --spacing 1 --push-len 5 --plan-device cuda:1
python3 _make_charts_pushtask.py          # host python is fine; writes ../../CHARTS_PUSHTASK.md
```

Everything after `--` is passed to `push_t_demo.build_args`, so a batch cell is exactly
the single-case command above for `--case 0 .. N-1`.  The driver loads the checkpoint
and the primitive library once per cell, runs the physics headless as fast as it
computes, skips cells whose json already exists (`--redo` to rerun), and writes
`results/push_task/<shape>_<env>.json` with one record per case.  The 9-cell x 100-case
sweep took 132 min on one card.

### Metrics (per case)

- **Compute time** -- wall-clock from the first plan to the end of the run: the MPPI
  solve, every replan, every action selection and the physics.  Excludes the per-cell
  one-offs shared by all cases (checkpoint load ~3 s, primitive calibration ~30 s per
  shape, cached).
- **Deviation** -- on every physics frame, the distance from the shape's position to the
  nearest point of the **original** planned path (before any replan), in world units
  (the 2-D envs are ~350 units across, the shapes ~40-80).  Averaged over the frames of
  a case; the table reports mean +- SD over cases, and the per-case maximum averaged.
- **Collision** -- the shape's footprint intersected a wall or a block on at least one
  frame.  A geometric check (shapely) every frame, since the environment is scenery in
  the physics.  The pusher is not checked (it is teleported to each entry point).
- **Goal reached** -- the controller's goal test (`--goal-dist` / `--goal-deg`) passed
  within `--max-seconds` of simulated time.
- **Success** -- goal reached **and** never collided.

## Results (2026-09-13)

`--teleport --spacing 1 --push-len 5`, first 100 cases of each
`testing_data_1k_complete/3dshape/<shape>_<env>` set, `Experiments/3dshape_2d/<cell>`
checkpoints.  Group rows pool the cases above them.  Full table with extra columns:
`Eikonal_Planning/ntrl-demo/CHARTS_PUSHTASK.md`.

| Env | Shape | Cases | Compute total (s) | Compute / case (s) | Deviation mean ± SD (units) | Max dev. mean (units) | Success (%) | Collision (%) | Goal reached (%) | Actions / case |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2denv1 | rectangle | 100 | 986 | 9.9 | 4.39 ± 7.20 | 12.4 | 85.0 | 15.0 | 100.0 | 27.1 |
| 2denv1 | Lshape3d | 100 | 927 | 9.3 | 4.25 ± 5.46 | 11.9 | 68.0 | 32.0 | 100.0 | 24.9 |
| 2denv1 | Ashape3d | 100 | 1101 | 11.0 | 3.17 ± 2.05 | 8.9 | 81.0 | 19.0 | 99.0 | 25.0 |
| **2denv1** | **mean** | 300 | 3015 | 10.0 | 3.94 ± 5.36 | 11.1 | 78.0 | 22.0 | 99.7 | 25.7 |
| 2denv3 | rectangle | 100 | 859 | 8.6 | 3.94 ± 3.86 | 11.7 | 93.0 | 7.0 | 100.0 | 26.2 |
| 2denv3 | Lshape3d | 100 | 675 | 6.8 | 4.05 ± 3.23 | 10.9 | 93.0 | 7.0 | 100.0 | 25.6 |
| 2denv3 | Ashape3d | 100 | 801 | 8.0 | 4.55 ± 4.31 | 11.3 | 94.0 | 6.0 | 100.0 | 25.6 |
| **2denv3** | **mean** | 300 | 2335 | 7.8 | 4.18 ± 3.82 | 11.3 | 93.3 | 6.7 | 100.0 | 25.8 |
| 2denv4 | rectangle | 100 | 853 | 8.5 | 3.46 ± 3.27 | 10.0 | 95.0 | 5.0 | 100.0 | 24.9 |
| 2denv4 | Lshape3d | 100 | 776 | 7.8 | 4.52 ± 3.26 | 11.7 | 82.0 | 18.0 | 100.0 | 27.5 |
| 2denv4 | Ashape3d | 100 | 945 | 9.4 | 4.19 ± 3.12 | 11.7 | 91.0 | 9.0 | 100.0 | 27.3 |
| **2denv4** | **mean** | 300 | 2574 | 8.6 | 4.06 ± 3.24 | 11.1 | 89.3 | 10.7 | 100.0 | 26.5 |
| **all** | **mean** | 900 | 7923 | 8.8 | 4.06 ± 4.23 | 11.2 | 86.9 | 13.1 | 99.9 | 26.0 |

Reading it:

- The controller reaches the goal in 899/900 cases; what fails is brushing an obstacle
  on the way.  The test pairs are planned with only `offset 0.02 x 350 = 7` units of
  clearance, and the pusher's tracking error (mean ~4 units, per-case max ~11) eats into
  that, so the collision rate tracks clutter: 2denv1 (11 bodies) is worst, 2denv3 (6)
  best, and the L-shape -- whose concave corner limits where the pusher can stand -- is
  the worst shape.
- Collisions are mostly brief grazes (tens of frames out of ~1000).  A few cases show a
  large max deviation (>80 units, two rectangle/2denv1 runs): a replan chose a
  different route around an obstacle and the executed path left the original one
  entirely, even though the goal was still reached.
- Compute is ~9 s per case for ~26 actions, i.e. ~0.35 s per action, of which the
  replan is ~20 ms; the rest is the 10 000-primitive screen and selection plus physics.
  The clearance screen is vectorised (`PushController._free_many`); before that it was
  ~1.2 s per action.

Per-case records (all of the above plus final pose error, replans, first-collision
time, sim time, path length) are in `results/push_task/<shape>_<env>.json`; the driver
log is `results/push_task/driver.log`.
