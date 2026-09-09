#!/usr/bin/env bash
# 2denv3 only, current ("my method") pipeline: training set + shared test set.
# All seven shapes -- 2denv3 is a fresh environment, nothing generated yet.
# BATCH defaults to 1000 here rather than the 500 the other 2-D sweeps use: 2denv3
# is the sparsest env (6 bodies vs 8/12/13), so its clearance tensor is the smallest
# of the four.  Batch size only chunks the rejection sampler, never the sampled
# distribution -- drop it back to 500 if a larger shape OOMs.
# Same settings as _run_2d_preprocess.sh phase A.
set -u
cd /workspace/ntrl-demo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ordered by boundary-triangle count (descending) -- longest job first.
SHAPES="${SHAPES:-Ashape3d 4shape3d Fshape3d Tshape3d Vshape3d Lshape3d rectangle}"
MESH=datasets/3dshape/2denv3_zup.obj
BATCH=${BATCH:-1000}
LOGS=./.preplogs_2denv3
mkdir -p $LOGS

current () {  # shape device
    local shape=$1 dev=$2 ds=${1}_2denv3
    (
        set -e
        python -u dataprocessing/preprocess_obj.py \
            --env   $MESH \
            --shape datasets/3dshape/${shape}_zup.obj \
            --out   datasets/3dshape/$ds \
            --num_samples 800000 \
            --2d \
            --visualize \
            --batch_size $BATCH \
            --device $dev
        python -u dataprocessing/preprocess_obj.py \
            --env   $MESH \
            --shape datasets/3dshape/${shape}_zup.obj \
            --out   testing_data/3dshape/$ds \
            --num_samples 1000 \
            --testing_data \
            --offset 0.02 \
            --2d \
            --batch_size $BATCH \
            --visualize \
            --device $dev
    ) > $LOGS/${ds}.log 2>&1
}

work=($SHAPES)
# GPU_LIST / NWORKERS let one run be pinned to a subset of the cards; the default
# is the three-GPU round robin, NWORKERS=1 runs the shapes strictly one at a time.
GPUS=(${GPU_LIST:-cuda:0 cuda:1 cuda:2})
NG=${#GPUS[@]}
NWORKERS=${NWORKERS:-3}

worker () {  # index stride
    local w=$1 stride=$2 k=$1 rc dev=${GPUS[$(( $1 % NG ))]}
    while [ $k -lt ${#work[@]} ]; do
        echo "[$w] start ${work[$k]}_2denv3 on $dev"
        current ${work[$k]} $dev
        rc=$?
        if [ $rc -eq 0 ]; then echo "[ok]   ${work[$k]}_2denv3"
        else echo "[FAIL] ${work[$k]}_2denv3 rc=$rc"; fi
        k=$((k+stride))
    done
}

t0=$(date +%s)
pids=()
for w in $(seq 0 $((NWORKERS-1))); do worker $w $NWORKERS & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
echo "[done] total wall time: $(( $(date +%s) - t0 ))s"
echo "--- datasets written ---"
ls -d datasets/3dshape/*_2denv3 testing_data/3dshape/*_2denv3 2>/dev/null
grep -l "Traceback" $LOGS/*.log 2>/dev/null || echo "no tracebacks in $LOGS"
