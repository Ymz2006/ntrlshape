# Experiments -- NTFields on the 3-D shape task

Trained with `train_3dshape_all.sh` (4000 epochs x 5 batches of 2000 per environment -- the same 20,000 optimizer steps the ntrl-demo baselines get) on the same datasets as the main repo's `ntrl-demo/experiments.md`.

The last three columns come from `3d_plan.py`, which plans the 1000 start/goal
pairs of `testing_data/3dshape/<env>` with the MPPI controller of
`ntrl-demo/evaluate_training_3d.py` driven by this checkpoint. A case succeeds
when the rollout reaches the 0.01 goal ball AND the placed shape is
collision-free at every waypoint (`preprocess_obj`'s point-in-tet test against a
50k-point sampling of the whole environment mesh, walls included). Time is the
per-case wall clock of the rollout over all scored cases; path length is OMPL's
SE(3) metric (`sum ||dt|| + acos(|q.q'|)`) over the successful ones -- the same
two quantities `RRT_experiments.md` reports, measured the same way.

NTFields replays an epoch whose mean loss exceeds `--repeat-ratio` times the
previous one, measured against a `prev_diff` that starts at 1.0. The 6-D shape
data starts near 2.0, so the shipped 1.2 rejects epoch 1 forever; these runs use
a relaxed 2.5 with a 20-replay cap. Losses are therefore not comparable in scale
to the ntrl-demo tables -- the loss functions differ (NTFields is isotropic and
ignores `normal.npy`).

| Env | Model | Epochs | Final Loss | Train Time (s) | Success Rate | Path Time mean ± sd (s) | Path Length mean ± sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | `./outputs/3dshape/rectangle_env1/latest.pt` | 4000 | 7.2579e-01 | 2313 | 96.8% | 0.094 ± 0.035 | 1.879 ± 0.634 |
| Lshape3d_env1 | `./outputs/3dshape/Lshape3d_env1/latest.pt` | 4000 | 8.2165e-01 | 2333 | 89.0% | 0.110 ± 0.044 | 1.903 ± 0.651 |
| Fshape3d_env1 | `./outputs/3dshape/Fshape3d_env1/latest.pt` | 4000 | 8.0156e-01 | 2315 | 91.8% | 0.117 ± 0.054 | 1.912 ± 0.661 |
| Ashape3d_env1 | `./outputs/3dshape/Ashape3d_env1/latest.pt` | 4000 | 9.1472e-01 | 2348 | 76.5% | 0.126 ± 0.063 | 1.962 ± 0.764 |
| Vshape3d_env1 | `./outputs/3dshape/Vshape3d_env1/latest.pt` | 4000 | 9.3451e-01 | 2322 | 78.1% | 0.109 ± 0.050 | 1.976 ± 0.728 |
| 4shape3d_env1 | `./outputs/3dshape/4shape3d_env1/latest.pt` | 4000 | 8.3883e-01 | 2240 | 88.2% | 0.106 ± 0.042 | 1.939 ± 0.668 |
| rectangle_env2 | `./outputs/3dshape/rectangle_env2/latest.pt` | 4000 | 9.5538e-01 | 1103 | 87.7% | 0.118 ± 0.059 | 1.861 ± 0.668 |
| Lshape3d_env2 | `./outputs/3dshape/Lshape3d_env2/latest.pt` | 4000 | 8.9852e-01 | 1108 | 80.7% | 0.103 ± 0.047 | 1.917 ± 0.720 |
| Fshape3d_env2 | `./outputs/3dshape/Fshape3d_env2/latest.pt` | 4000 | 8.8708e-01 | 1094 | 87.4% | 0.127 ± 0.062 | 1.952 ± 0.734 |
| Ashape3d_env2 | `./outputs/3dshape/Ashape3d_env2/latest.pt` | 4000 | 9.5673e-01 | 1099 | 70.3% | 0.124 ± 0.063 | 1.923 ± 0.719 |
| Vshape3d_env2 | `./outputs/3dshape/Vshape3d_env2/latest.pt` | 4000 | 9.6110e-01 | 1091 | 66.6% | 0.113 ± 0.057 | 1.914 ± 0.697 |
| 4shape3d_env2 | `./outputs/3dshape/4shape3d_env2/latest.pt` | 4000 | 8.9985e-01 | 1080 | 83.6% | 0.114 ± 0.055 | 1.907 ± 0.707 |
| rectangle_env3 | `./outputs/3dshape/rectangle_env3/latest.pt` | 4000 | 6.5910e-01 | 1095 | 93.7% | 0.105 ± 0.069 | 1.662 ± 0.578 |
| Lshape3d_env3 | `./outputs/3dshape/Lshape3d_env3/latest.pt` | 4000 | 6.9644e-01 | 1108 | 96.6% | 0.097 ± 0.054 | 1.673 ± 0.627 |
| Fshape3d_env3 | `./outputs/3dshape/Fshape3d_env3/latest.pt` | 4000 | 6.9397e-01 | 1079 | 96.6% | 0.101 ± 0.056 | 1.639 ± 0.623 |
| Ashape3d_env3 | `./outputs/3dshape/Ashape3d_env3/latest.pt` | 4000 | 7.4669e-01 | 1079 | 97.2% | 0.104 ± 0.053 | 1.685 ± 0.626 |
| Vshape3d_env3 | `./outputs/3dshape/Vshape3d_env3/latest.pt` | 4000 | 7.4460e-01 | 1116 | 96.0% | 0.100 ± 0.053 | 1.685 ± 0.621 |
| 4shape3d_env3 | `./outputs/3dshape/4shape3d_env3/latest.pt` | 4000 | 6.9736e-01 | 1097 | 97.0% | 0.099 ± 0.050 | 1.665 ± 0.607 |
| rectangle_env4 | `./outputs/3dshape/rectangle_env4/latest.pt` | 4000 | 8.5260e-01 | 1103 | 95.4% | 0.108 ± 0.054 | 1.762 ± 0.622 |
| Lshape3d_env4 | `./outputs/3dshape/Lshape3d_env4/latest.pt` | 4000 | 8.4169e-01 | 1093 | 94.6% | 0.101 ± 0.045 | 1.734 ± 0.628 |
| Fshape3d_env4 | `./outputs/3dshape/Fshape3d_env4/latest.pt` | 4000 | 8.3270e-01 | 1110 | 94.0% | 0.103 ± 0.049 | 1.767 ± 0.619 |
| Ashape3d_env4 | `./outputs/3dshape/Ashape3d_env4/latest.pt` | 4000 | 8.9163e-01 | 1097 | 91.5% | 0.103 ± 0.050 | 1.737 ± 0.632 |
| Vshape3d_env4 | `./outputs/3dshape/Vshape3d_env4/latest.pt` | 4000 | 8.8901e-01 | 1098 | 91.6% | 0.101 ± 0.054 | 1.718 ± 0.638 |
| 4shape3d_env4 | `./outputs/3dshape/4shape3d_env4/latest.pt` | 4000 | 8.4678e-01 | 1111 | 94.1% | 0.105 ± 0.052 | 1.767 ± 0.628 |
| Tshape3d_env4 | `./outputs/3dshape/Tshape3d_env4/latest.pt` | 4000 | 7.1293e-01 | 1111 | 84.3% | 0.126 ± 0.123 | 1.517 ± 0.790 |
| Tshape3d_env1 | `./outputs/3dshape/Tshape3d_env1/latest.pt` | 4000 | 9.1801e-01 | 1097 | 63.1% | 0.166 ± 0.119 | 1.981 ± 0.731 |
| Lcouch_Corozal | `./outputs/3dshape/Lcouch_Corozal/latest.pt` | 4000 | 6.9529e-01 | 1034 | 60.0% | 0.151 ± 0.099 | 1.828 ± 0.678 |
