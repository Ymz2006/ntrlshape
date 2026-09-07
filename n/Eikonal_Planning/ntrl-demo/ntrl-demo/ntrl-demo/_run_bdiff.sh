#!/usr/bin/env bash
set -u
cd /workspace/ntrl-demo/ntrl-demo
OUT=./Experiments/3dshape_bdiff
mkdir -p $OUT/logs
pids=()
launch () {  # dataset package device
    local ds=$1 pkg=$2 dev=$3
    python -u train/train_3dshape_bdiff.py \
        --dataPath ./datasets/3dshape/$ds \
        --models $pkg --seed 1 --device $dev --modelPath $OUT \
        > $OUT/logs/${ds}_${pkg}.log 2>&1 &
    pids+=($!)
    echo "[launch] $ds $pkg on $dev (pid ${pids[-1]})"
}
launch Ashape3d_env2 metric_bdiff cuda:0
launch Ashape3d_env2 metric       cuda:0
launch Vshape3d_env2 metric_bdiff cuda:1
launch Vshape3d_env2 metric       cuda:1
launch 4shape3d_env2 metric_bdiff cuda:2
launch 4shape3d_env2 metric       cuda:2
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail+1)); done
echo "[done] failures=$fail"
