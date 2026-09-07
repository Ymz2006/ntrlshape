# Experiments -- arm baseline (`models/metric_arm`) on the 3-D shape task

> Collected at the repository root. Every unqualified path below (`datasets/`, `Experiments/`, `outputs/`, `results/`, `tests/`, `train/`, ...)
> is relative to `baselines/ntrl-demo/`, where this table's runs live. Regenerate with `python train/make_experiments_md.py` from that directory.

Trained with `train/train_3dshape_arm_all.sh` (4000 epochs per environment, dim = 6, source = origin) on the same datasets as the main repo's `experiments_ours.md`, whose numbers come from `models/metric`.
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
`results/3d_plan_tight/<env>/summary.json`.

| Env | Model | Epochs | Final Loss | Train Time (s) | Valid | Plan Time (s) | Path Length | Valid (tight) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./Experiments/3dshape_arm/rectangle_env1_09_04_21_51/latest.pt` | 4000 | 2.8525e-03 | 748 | 99.7% (997/1000) | 0.082 ± 0.029 | 0.771 ± 0.260 | 95.7% (921/962) |
| Lshape3d_env1 | `./Experiments/3dshape_arm/Lshape3d_env1_09_04_22_12/latest.pt` | 4000 | 3.2245e-03 | 1962 | 98.4% (984/1000) | 0.081 ± 0.028 | 0.765 ± 0.251 | 92.1% (891/967) |
| Fshape3d_env1 | `./Experiments/3dshape_arm/Fshape3d_env1_09_04_22_12/latest.pt` | 4000 | 3.3625e-03 | 1976 | 99.6% (996/1000) | 0.083 ± 0.029 | 0.775 ± 0.265 | 93.5% (902/965) |
| Ashape3d_env1 | `./Experiments/3dshape_arm/Ashape3d_env1_09_04_22_12/latest.pt` | 4000 | 3.5588e-03 | 1991 | 97.3% (973/1000) | 0.085 ± 0.036 | 0.774 ± 0.276 | 89.2% (839/941) |
| Vshape3d_env1 | `./Experiments/3dshape_arm/Vshape3d_env1_09_04_22_12/latest.pt` | 4000 | 3.6330e-03 | 1995 | 95.9% (959/1000) | 0.084 ± 0.035 | 0.774 ± 0.268 | 87.9% (831/945) |
| 4shape3d_env1 | `./Experiments/3dshape_arm/4shape3d_env1_09_04_22_12/latest.pt` | 4000 | 3.4493e-03 | 2008 | 96.5% (965/1000) | 0.087 ± 0.037 | 0.771 ± 0.260 | 90.4% (865/957) |
| rectangle_env2 | `./Experiments/3dshape_arm/rectangle_env2_09_04_22_12/latest.pt` | 4000 | 3.0512e-03 | 1942 | 98.8% (988/1000) | 0.089 ± 0.045 | 0.742 ± 0.257 | 94.1% (910/967) |
| Lshape3d_env2 | `./Experiments/3dshape_arm/Lshape3d_env2_09_04_22_44/latest.pt` | 4000 | 3.5410e-03 | 1938 | 96.0% (958/998) | 0.085 ± 0.031 | 0.749 ± 0.259 | 87.9% (846/963) |
| Fshape3d_env2 | `./Experiments/3dshape_arm/Fshape3d_env2_09_04_22_45/latest.pt` | 4000 | 3.5663e-03 | 1952 | 96.4% (964/1000) | 0.087 ± 0.033 | 0.766 ± 0.271 | 89.2% (850/953) |
| Ashape3d_env2 | `./Experiments/3dshape_arm/Ashape3d_env2_09_04_22_45/latest.pt` | 4000 | 3.9185e-03 | 1966 | 89.2% (892/1000) | 0.083 ± 0.030 | 0.737 ± 0.248 | 78.2% (739/945) |
| Vshape3d_env2 | `./Experiments/3dshape_arm/Vshape3d_env2_09_04_22_45/latest.pt` | 4000 | 3.9969e-03 | 1963 | 89.9% (898/999) | 0.078 ± 0.027 | 0.752 ± 0.249 | 79.6% (747/939) |
| 4shape3d_env2 | `./Experiments/3dshape_arm/4shape3d_env2_09_04_22_45/latest.pt` | 4000 | 3.8330e-03 | 1983 | 95.7% (957/1000) | 0.086 ± 0.032 | 0.755 ± 0.265 | 84.8% (797/940) |
| rectangle_env3 | `./Experiments/3dshape_arm/rectangle_env3_09_04_22_46/latest.pt` | 4000 | 2.8474e-03 | 1934 | 99.8% (998/1000) | 0.081 ± 0.030 | 0.682 ± 0.240 | 94.7% (904/955) |
| Lshape3d_env3 | `./Experiments/3dshape_arm/Lshape3d_env3_09_04_23_17/latest.pt` | 4000 | 3.2206e-03 | 1812 | 98.3% (982/999) | 0.077 ± 0.031 | 0.664 ± 0.242 | 91.2% (863/946) |
| Fshape3d_env3 | `./Experiments/3dshape_arm/Fshape3d_env3_09_04_23_17/latest.pt` | 4000 | 3.3295e-03 | 1869 | 98.7% (986/999) | 0.072 ± 0.027 | 0.657 ± 0.243 | 91.3% (860/942) |
| Ashape3d_env3 | `./Experiments/3dshape_arm/Ashape3d_env3_09_04_23_18/latest.pt` | 4000 | 3.4302e-03 | 1871 | 98.4% (983/999) | 0.069 ± 0.025 | 0.652 ± 0.236 | 92.1% (852/925) |
| Vshape3d_env3 | `./Experiments/3dshape_arm/Vshape3d_env3_09_04_23_18/latest.pt` | 4000 | 3.5674e-03 | 1868 | 98.3% (982/999) | 0.070 ± 0.026 | 0.659 ± 0.237 | 91.4% (844/923) |
| 4shape3d_env3 | `./Experiments/3dshape_arm/4shape3d_env3_09_04_23_18/latest.pt` | 4000 | 3.3159e-03 | 1861 | 97.9% (978/999) | 0.070 ± 0.025 | 0.654 ± 0.233 | 90.1% (847/940) |
| rectangle_env4 | `./Experiments/3dshape_arm/rectangle_env4_09_04_23_18/latest.pt` | 4000 | 2.9718e-03 | 1828 | 99.5% (995/1000) | 0.075 ± 0.026 | 0.680 ± 0.229 | 94.7% (903/954) |
| Lshape3d_env4 | `./Experiments/3dshape_arm/Lshape3d_env4_09_04_23_47/latest.pt` | 4000 | 3.5186e-03 | 2017 | 98.2% (981/999) | 0.071 ± 0.024 | 0.663 ± 0.222 | 92.4% (868/939) |
| Fshape3d_env4 | `./Experiments/3dshape_arm/Fshape3d_env4_09_04_23_49/latest.pt` | 4000 | 3.5719e-03 | 2031 | 98.3% (982/999) | 0.073 ± 0.024 | 0.676 ± 0.222 | 91.6% (857/936) |
| Ashape3d_env4 | `./Experiments/3dshape_arm/Ashape3d_env4_09_04_23_49/latest.pt` | 4000 | 3.8480e-03 | 2009 | 96.5% (964/999) | 0.070 ± 0.025 | 0.655 ± 0.231 | 87.9% (807/918) |
| Vshape3d_env4 | `./Experiments/3dshape_arm/Vshape3d_env4_09_04_23_49/latest.pt` | 4000 | 3.8269e-03 | 1992 | 97.4% (974/1000) | 0.069 ± 0.024 | 0.657 ± 0.230 | 88.6% (821/927) |
| 4shape3d_env4 | `./Experiments/3dshape_arm/4shape3d_env4_09_04_23_49/latest.pt` | 4000 | 3.6465e-03 | 2002 | 97.1% (971/1000) | 0.072 ± 0.025 | 0.668 ± 0.224 | 88.8% (835/940) |
| Tshape3d_env4 | `./Experiments/3dshape_arm/Tshape3d_env4_09_04_23_49/latest.pt` | 4000 | 2.3631e-03 | 2009 | 82.6% (826/1000) | 0.121 ± 0.065 | 0.974 ± 0.482 | 67.5% (675/1000) |
| Tshape3d_env1 | `./Experiments/3dshape_arm/Tshape3d_env1_09_05_00_33/latest.pt` | 4000 | 3.6193e-03 | 2159 | 91.6% (909/992) | 0.088 ± 0.036 | 0.827 ± 0.307 | 85.0% (819/963) |
| Lcouch_Corozal | `./Experiments/3dshape_arm/Lcouch_Corozal_09_05_00_33/latest.pt` | 4000 | 3.0696e-03 | 2197 | 76.9% (665/865) | 0.103 ± 0.050 | 0.916 ± 0.444 | 59.9% (392/654) |
