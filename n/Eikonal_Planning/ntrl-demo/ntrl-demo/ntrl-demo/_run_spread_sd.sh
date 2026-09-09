#!/bin/bash
# hlB + sampled-speed-spread steering ("SD" column of ../../experiments_ours.md).
# One _spread_planner.py run per 3-D table row, 1000 cases, --steer-bias 0.5 --no-gate.
set -u
DEV=${DEV:-cuda:2}
OUT=results/spread_sd
mkdir -p $OUT

run () {  # run <dataset> <checkpoint>
  local ds=$1 ck=$2
  if [ -f "$OUT/$ds/records.json" ]; then echo "skip $ds (done)"; return; fi
  echo "=== $ds  $(date +%H:%M:%S)"
  python _spread_planner.py \
      --dataPath testing_data/3dshape/$ds \
      --checkpoint $ck \
      --device $DEV --cases 1000 --batch 250 \
      --steer-bias 0.5 --no-gate \
      --out $OUT/$ds > $OUT/$ds.log 2>&1
  tail -1 $OUT/$ds.log
}

run rectangle_env1 ./Experiments/3dshape/3dshape_08_06_17_06/latest.pt
run Lshape3d_env1  ./Experiments/3dshape/3dshape_08_15_20_19/latest.pt
run Fshape3d_env1  ./Experiments/3dshape/3dshape_08_16_11_35/latest.pt
run Ashape3d_env1  ./Experiments/3dshape/3dshape_09_01_19_49/latest.pt
run Vshape3d_env1  ./Experiments/3dshape/3dshape_09_01_19_50/latest.pt
run 4shape3d_env1  ./Experiments/3dshape/3dshape_09_01_19_54/latest.pt
run rectangle_env2 ./Experiments/3dshape/3dshape_08_29_15_41/latest.pt
run Lshape3d_env2  ./Experiments/3dshape/3dshape_08_30_11_04/latest.pt
run Fshape3d_env2  ./Experiments/3dshape/3dshape_08_30_11_05/latest.pt
run Ashape3d_env2  ./Experiments/3dshape/3dshape_09_04_09_09/latest.pt
run Vshape3d_env2  ./Experiments/3dshape/3dshape_09_04_09_11/latest.pt
run 4shape3d_env2  ./Experiments/3dshape/3dshape_09_04_09_13/latest.pt
run rectangle_env3 ./Experiments/3dshape/3dshape_09_04_09_14/latest.pt
run Lshape3d_env3  ./Experiments/3dshape/3dshape_09_04_09_15/latest.pt
run Fshape3d_env3  ./Experiments/3dshape/3dshape_09_04_09_16/latest.pt
run Ashape3d_env3  ./Experiments/3dshape/3dshape_09_04_09_21/latest.pt
run Vshape3d_env3  ./Experiments/3dshape/3dshape_09_04_09_22/latest.pt
run 4shape3d_env3  ./Experiments/3dshape/3dshape_09_04_09_23/latest.pt
run rectangle_env4 ./Experiments/3dshape/3dshape_09_04_09_26/latest.pt
run Lshape3d_env4  ./Experiments/3dshape/3dshape_09_04_09_29/latest.pt
run Fshape3d_env4  ./Experiments/3dshape/3dshape_09_04_09_30/latest.pt
run Ashape3d_env4  ./Experiments/3dshape/3dshape_09_01_20_49/latest.pt
run Vshape3d_env4  ./Experiments/3dshape/3dshape_09_01_20_52/latest.pt
run 4shape3d_env4  ./Experiments/3dshape/3dshape_09_01_20_55/latest.pt
run Tshape3d_env4  ./Experiments/3dshape/3dshape_08_19_12_31/latest.pt
echo "ALL DONE $(date)"
