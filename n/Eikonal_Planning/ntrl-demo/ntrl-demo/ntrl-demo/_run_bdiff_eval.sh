#!/usr/bin/env bash
set -u
cd /workspace/ntrl-demo
EXP=./Experiments/3dshape_bdiff
OUT=./results/output_3d/bdiff_eval
mkdir -p $OUT/logs
pids=()
run () {  # run_dir dataset device
    local run=$1 ds=$2 dev=$3
    python -u evaluate_training_3d_batched.py \
        --dataPath ./testing_data/3dshape/$ds \
        --out $OUT/$run \
        --checkpoint $EXP/$run/latest.pt \
        --cases 500 --batch 250 --device $dev --no-viser --verbose \
        > $OUT/logs/${run}.log 2>&1 &
    pids+=($!)
    echo "[launch] $run ($ds) on $dev pid ${pids[-1]}"
}
run Ashape3d_env2_metric_09_05_21_18        Ashape3d_env2 cuda:0
run Ashape3d_env2_metric_bdiff_09_05_21_18  Ashape3d_env2 cuda:0
run Vshape3d_env2_metric_09_05_21_18        Vshape3d_env2 cuda:1
run Vshape3d_env2_metric_bdiff_09_05_21_18  Vshape3d_env2 cuda:1
run 4shape3d_env2_metric_09_05_21_18        4shape3d_env2 cuda:2
run 4shape3d_env2_metric_bdiff_09_05_21_18  4shape3d_env2 cuda:2
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail+1)); done
echo "[done] failures=$fail"
