#!/usr/bin/env bash
#
# Train NTFields on every 3-D shape dataset, 4000 epochs of 5 x 2000 samples
# each -- the same 20,000 optimizer steps at the same batch size that the
# ntrl-demo baselines (models/metric, models/metric_arm) get, so the three are
# comparable per training budget rather than per wall clock.
#
# Jobs are spread round-robin over the GPU slots in $SLOTS and run
# concurrently.  A dataset that already has an epoch-$EPOCHS checkpoint is
# skipped, so the script is re-runnable after an interruption.
#
#   bash train_3dshape_all.sh
#   DATASETS="rectangle_env1" SLOTS="cuda:1" bash train_3dshape_all.sh
#
set -u

cd "$(dirname "$0")"

DATA_ROOT=${DATA_ROOT:-../../ntrl-demo/ntrl-demo/datasets/3dshape}
OUT_ROOT=${OUT_ROOT:-./outputs/3dshape}
EPOCHS=${EPOCHS:-4000}
BATCH_SIZE=${BATCH_SIZE:-2000}
MAX_BATCHES=${MAX_BATCHES:-5}
SLOTS=${SLOTS:-"cuda:0 cuda:0 cuda:1 cuda:1 cuda:2 cuda:2"}

# The env list of ntrl-demo/experiments.md, followed by the envs that exist as
# datasets but are not tabulated there.
DATASETS=${DATASETS:-"\
rectangle_env1 Lshape3d_env1 Fshape3d_env1 Ashape3d_env1 Vshape3d_env1 4shape3d_env1 \
rectangle_env2 Lshape3d_env2 Fshape3d_env2 Ashape3d_env2 Vshape3d_env2 4shape3d_env2 \
rectangle_env3 Lshape3d_env3 Fshape3d_env3 Ashape3d_env3 Vshape3d_env3 4shape3d_env3 \
rectangle_env4 Lshape3d_env4 Fshape3d_env4 Ashape3d_env4 Vshape3d_env4 4shape3d_env4 \
Tshape3d_env4 \
Tshape3d_env1 Lcouch_Corozal"}

LOG_DIR=${LOG_DIR:-$OUT_ROOT/logs}
mkdir -p "$LOG_DIR"

ckpt_of() {
    ls -t "$OUT_ROOT"/"$1"/Model_Epoch_$(printf '%05d' "$EPOCHS")_*.pt 2>/dev/null | head -1
}

run_one() {
    local dataset=$1 device=$2
    local log=$LOG_DIR/$dataset.log

    if [ -n "$(ckpt_of "$dataset")" ]; then
        echo "[skip] $dataset -- already trained to $EPOCHS epochs"
        return 0
    fi
    if [ ! -f "$DATA_ROOT/$dataset/sampled_points.npy" ]; then
        echo "[miss] $dataset -- no sampled_points.npy under $DATA_ROOT"
        return 1
    fi

    echo "[run ] $dataset on $device -> $log"
    python -u train_3dshape.py \
        --env "$dataset" \
        --data-root "$DATA_ROOT" \
        --output "$OUT_ROOT/$dataset" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-batches "$MAX_BATCHES" \
        --device "$device" > "$log" 2>&1
    local status=$?

    local ckpt; ckpt=$(ckpt_of "$dataset")
    if [ $status -ne 0 ] || [ -z "$ckpt" ]; then
        echo "[FAIL] $dataset (exit $status) -- see $log"
        return 1
    fi
    cp "$ckpt" "$(dirname "$ckpt")/latest.pt"
    echo "[done] $dataset -- $(dirname "$ckpt")/latest.pt"
}

slots=($SLOTS)
n_slots=${#slots[@]}
i=0
joblist=$(mktemp)
for dataset in $DATASETS; do
    echo "$dataset ${slots[$((i % n_slots))]}"
    i=$((i + 1))
done > "$joblist"

export -f run_one ckpt_of
export DATA_ROOT OUT_ROOT EPOCHS BATCH_SIZE MAX_BATCHES LOG_DIR

start=$(date +%s)
xargs -a "$joblist" -P "$n_slots" -n 2 bash -c 'run_one "$0" "$1"'
rm -f "$joblist"
echo "Total wall time: $(( $(date +%s) - start ))s"
