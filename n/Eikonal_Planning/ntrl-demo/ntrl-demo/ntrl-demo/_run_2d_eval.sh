#!/usr/bin/env bash
# Evaluate every (shape, 2-D env) checkpoint from BOTH pipelines on the SHARED
# test sets, with --2d so MPPI samples in the (x, y, rz) slice the field was fit on.
#
#   current  : Experiments/3dshape_2d/<ds>/latest.pt        -> results/output_3d/<ds>
#   recovered: Experiments/3dshape_2d_june03/<ds>/latest.pt -> results/output_3d/<ds>_june03
#              (--models metric_june03: the June-3 network is a different
#               architecture, so its checkpoint only loads under that package)
set -u
cd /workspace/ntrl-demo/ntrl-demo

SHAPES="rectangle Lshape3d Fshape3d Ashape3d Vshape3d 4shape3d Tshape3d"
ENVS="2denv4 2denv1"
CASES=${CASES:-1000}
EXP_CUR=./Experiments/3dshape_2d
EXP_J03=./Experiments/3dshape_2d_june03
LOGS=./.evallogs_2d
mkdir -p $LOGS

current () {  # ds device
    python -u evaluate_training_3d_batched.py \
        --dataPath ./testing_data/3dshape/$1 \
        --out ./results/output_3d/$1 \
        --checkpoint $EXP_CUR/$1/latest.pt \
        --2d --cases $CASES --no-viser --verbose \
        --device $2 > $LOGS/$1.log 2>&1
}

recovered () {  # ds device
    python -u evaluate_training_3d_batched.py \
        --dataPath ./testing_data/3dshape/$1 \
        --out ./results/output_3d/${1}_june03 \
        --models metric_june03 \
        --checkpoint $EXP_J03/$1/latest.pt \
        --2d --cases $CASES --no-viser --verbose \
        --device $2 > $LOGS/${1}_june03.log 2>&1
}

work=()
for env in $ENVS; do
    for shape in $SHAPES; do work+=("${shape}_${env}"); done
done

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
grep -l "Traceback" $LOGS/*.log 2>/dev/null || echo "no tracebacks in $LOGS"
