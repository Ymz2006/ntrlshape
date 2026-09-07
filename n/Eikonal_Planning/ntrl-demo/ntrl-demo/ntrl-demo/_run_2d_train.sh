#!/usr/bin/env bash
# Train every (shape, 2-D env) dataset with BOTH pipelines, 5000 epochs each.
#
#   current  : train/train_3dshape.py         (models/metric)         -> Experiments/3dshape_2d/<ds>
#   recovered: train/train_3dshape_june03.py  (models/metric_june03)  -> Experiments/3dshape_2d_june03/<ds>
#
# --name pins the run-folder name; without it the folder is <dataset parent>_<timestamp>,
# which every run launched in the same minute would collide on.
set -u
cd /workspace/ntrl-demo/ntrl-demo

SHAPES="rectangle Lshape3d Fshape3d Ashape3d Vshape3d 4shape3d Tshape3d"
ENVS="2denv4 2denv1"
EPOCHS=${EPOCHS:-5000}
EXP_CUR=./Experiments/3dshape_2d
EXP_J03=./Experiments/3dshape_2d_june03
mkdir -p $EXP_CUR/logs $EXP_J03/logs

current () {  # ds device
    python -u train/train_3dshape.py \
        --dataPath ./datasets/3dshape/$1 \
        --modelPath $EXP_CUR \
        --name $1 \
        --epochs $EPOCHS \
        --no-wandb \
        --device $2 > $EXP_CUR/logs/$1.log 2>&1
}

recovered () {  # ds device
    python -u train/train_3dshape_june03.py \
        --dataPath ./datasets/3dshape/${1}_june03 \
        --modelPath $EXP_J03 \
        --name $1 \
        --epochs $EPOCHS \
        --device $2 > $EXP_J03/logs/$1.log 2>&1
}

work=()
for env in $ENVS; do
    for shape in $SHAPES; do work+=("${shape}_${env}"); done
done

# Two runs per GPU, six slots -- the same packing the 3-D sweeps use.
SLOTS=(cuda:0 cuda:0 cuda:1 cuda:1 cuda:2 cuda:2)
NSLOT=${#SLOTS[@]}

worker () {  # index stage
    local w=$1 stage=$2 k=$1 rc
    while [ $k -lt ${#work[@]} ]; do
        echo "[$stage $w] start ${work[$k]} on ${SLOTS[$w]}"
        $stage ${work[$k]} ${SLOTS[$w]}
        rc=$?
        if [ $rc -eq 0 ]; then echo "[ok]   $stage ${work[$k]}"
        else echo "[FAIL] $stage ${work[$k]} rc=$rc"; fi
        k=$((k+NSLOT))
    done
}

t0=$(date +%s)
for stage in current recovered; do
    tS=$(date +%s)
    pids=()
    for w in $(seq 0 $((NSLOT-1))); do worker $w $stage & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    echo "[$stage done] $(( $(date +%s) - tS ))s"
done
echo "[done] total wall time: $(( $(date +%s) - t0 ))s"
echo "--- checkpoints ---"
ls $EXP_CUR/*/latest.pt $EXP_J03/*/latest.pt 2>/dev/null | wc -l
