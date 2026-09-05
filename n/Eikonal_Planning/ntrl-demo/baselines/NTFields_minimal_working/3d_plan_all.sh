#!/usr/bin/env bash
#
# Infer paths with every trained NTFields 3-D shape model, over the full 1000-case
# test set of the matching environment, and write the per-env timing / path-length
# summaries that make_experiments_md.py turns into experiments.md columns.
#
# Jobs are spread round-robin over the GPU slots in $SLOTS and run concurrently.
# An environment that already has a plan_summary.txt is skipped, so the script is
# re-runnable after an interruption (pass FORCE=1 to re-plan anyway).
#
#   bash 3d_plan_all.sh
#   DATASETS="rectangle_env1" SLOTS="cuda:1" bash 3d_plan_all.sh
#
set -u

cd "$(dirname "$0")"

MODEL_ROOT=${MODEL_ROOT:-./outputs/3dshape}
TEST_ROOT=${TEST_ROOT:-../../ntrl-demo/ntrl-demo/testing_data/3dshape}
OUT_ROOT=${OUT_ROOT:-./outputs/3dplan}
CASES=${CASES:-0}                       # 0 = all 1000 pairs
SLOTS=${SLOTS:-"cuda:0 cuda:1 cuda:2"}
FORCE=${FORCE:-0}

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

plan_one() {
    local dataset=$1 device=$2
    local log=$LOG_DIR/$dataset.log
    local summary=$OUT_ROOT/$dataset/plan_summary.txt

    if [ "$FORCE" != "1" ] && [ -f "$summary" ]; then
        echo "[skip] $dataset -- $summary already exists"
        return 0
    fi
    if [ ! -f "$MODEL_ROOT/$dataset/latest.pt" ]; then
        echo "[miss] $dataset -- no latest.pt under $MODEL_ROOT/$dataset"
        return 1
    fi
    if [ ! -f "$TEST_ROOT/$dataset/sampled_points.npy" ]; then
        echo "[miss] $dataset -- no test set under $TEST_ROOT/$dataset"
        return 1
    fi

    echo "[run ] $dataset on $device -> $log"
    python -u 3d_plan.py \
        --env "$dataset" \
        --model-root "$MODEL_ROOT" \
        --test-root "$TEST_ROOT" \
        --out "$OUT_ROOT/$dataset" \
        --cases "$CASES" \
        --device "$device" > "$log" 2>&1
    local status=$?

    if [ $status -ne 0 ] || [ ! -f "$summary" ]; then
        echo "[FAIL] $dataset (exit $status) -- see $log"
        return 1
    fi
    echo "[done] $dataset -- $(grep -m1 success_rate "$summary")"
}

slots=($SLOTS)
n_slots=${#slots[@]}
i=0
joblist=$(mktemp)
for dataset in $DATASETS; do
    echo "$dataset ${slots[$((i % n_slots))]}"
    i=$((i + 1))
done > "$joblist"

export -f plan_one
export MODEL_ROOT TEST_ROOT OUT_ROOT CASES LOG_DIR FORCE

start=$(date +%s)
xargs -a "$joblist" -P "$n_slots" -n 2 bash -c 'plan_one "$0" "$1"'
rm -f "$joblist"
echo "Total wall time: $(( $(date +%s) - start ))s"
