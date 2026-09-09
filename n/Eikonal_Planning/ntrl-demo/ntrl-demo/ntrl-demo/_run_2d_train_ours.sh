#!/usr/bin/env bash
# Our method (models/metric, train/train_3dshape.py) on every 2-D cell:
# 7 shapes x 4 planar envs = 28 runs -> Experiments/3dshape_2d/<cell>/latest.pt
#
# Unlike _run_2d_train.sh this covers all four envs (that one predates 2denv2 /
# 2denv3) and runs the current pipeline only, not the June-3 one.
#
# --name pins the run-folder name; without it the folder is <dataset parent>_<timestamp>
# and runs launched in the same minute collide.
#
# The (Tshape3d, 2denv4) cell's dataset is datasets/3dshape/Tshape3d_env4 -- built
# before the env-tag naming -- but its run is still named Tshape3d_2denv4 so the
# checkpoint matches the cell name used in 2d_gen_times.md and the master README.
set -u
cd /workspace/ntrl-demo

SHAPES=${SHAPES:-"rectangle Lshape3d Fshape3d Ashape3d Vshape3d 4shape3d Tshape3d"}
ENVS=${ENVS:-"2denv1 2denv2 2denv3 2denv4"}
EPOCHS=${EPOCHS:-5000}
EXP=./Experiments/3dshape_2d
mkdir -p $EXP/logs

dataset_of () {  # cell -> dataset dir name
    case $1 in
        Tshape3d_2denv4) echo Tshape3d_env4 ;;
        *)               echo $1 ;;
    esac
}

current () {  # cell device
    local cell=$1 dev=$2 ds
    ds=$(dataset_of $cell)
    python -u train/train_3dshape.py \
        --dataPath ./datasets/3dshape/$ds \
        --modelPath $EXP \
        --name $cell \
        --epochs $EPOCHS \
        --no-wandb \
        --device $dev > $EXP/logs/$cell.log 2>&1
}

work=()
for env in $ENVS; do
    for shape in $SHAPES; do work+=("${shape}_${env}"); done
done

# One slot per GPU plus a second on the two quieter cards; override with SLOTS.
SLOTS=(${SLOT_LIST:-cuda:0 cuda:1 cuda:1 cuda:2 cuda:2})
NSLOT=${#SLOTS[@]}

worker () {  # index
    local w=$1 k=$1 rc cell
    while [ $k -lt ${#work[@]} ]; do
        cell=${work[$k]}
        if [ -f "$EXP/$cell/latest.pt" ]; then
            echo "[skip] $cell -- already trained"
        elif [ ! -f "./datasets/3dshape/$(dataset_of $cell)/sampled_points.npy" ]; then
            echo "[miss] $cell -- no dataset $(dataset_of $cell)"
        else
            echo "[run ] $cell on ${SLOTS[$w]}"
            current $cell ${SLOTS[$w]}
            rc=$?
            if [ $rc -eq 0 ] && [ -f "$EXP/$cell/latest.pt" ]; then echo "[ok]   $cell"
            else echo "[FAIL] $cell rc=$rc -- see $EXP/logs/$cell.log"; fi
        fi
        k=$((k+NSLOT))
    done
}

t0=$(date +%s)
pids=()
for w in $(seq 0 $((NSLOT-1))); do worker $w & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
echo "[done] total wall time: $(( $(date +%s) - t0 ))s"
echo "--- checkpoints: $(ls $EXP/*/latest.pt 2>/dev/null | wc -l) ---"
