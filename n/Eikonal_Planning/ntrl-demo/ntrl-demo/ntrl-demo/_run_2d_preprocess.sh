#!/usr/bin/env bash
# Preprocess every (shape, 2-D env) pair through BOTH pipelines.
#
#   current  : dataprocessing/preprocess_obj.py --2d        -> datasets/3dshape/<shape>_<env>
#   recovered: dataprocessing/preprocess_obj_june03.py --2d -> datasets/3dshape/<shape>_<env>_june03
#   test set : dataprocessing/preprocess_obj.py --2d --testing_data --offset 0.02
#              -> testing_data/3dshape/<shape>_<env>   (shared by both pipelines so the
#                 two are scored on exactly the same start/goal queries)
#
# Both meshes must be z-up; dataprocessing/obj_yup_to_zup.py produced the *_zup.obj files.
#
# --batch_size: on a planar env the z-flattened environment defeats the
# broad-phase env cull in evaluate_placements (every env point ends up within
# min_center_dist + 2*R_shape of the placement), so the (B, F, E) clearance
# tensor stays dense and peak memory scales with the shape's triangle count.
# The README's 3-D --batch_size 2000 needs >23 GiB here and OOMs a 24 GiB card
# on its own for the larger shapes; 500 keeps the worst of them near 6 GiB, which
# is what lets two jobs share a GPU.  Batch size only changes how the rejection
# sampler is chunked, never the sampled distribution.
#
# The two generators run as separate phases (June-3 at its era --batch_size 256
# still needs ~5 GiB) so a phase's per-GPU footprint stays predictable.
set -u
cd /workspace/ntrl-demo/ntrl-demo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ordered by boundary-triangle count (descending): the per-pair clearance query
# is (B, kept env points, F), so this is longest-job-first for the round robin.
SHAPES="Ashape3d 4shape3d Fshape3d Vshape3d Tshape3d Lshape3d rectangle"
ENVS="2denv4 2denv1"
LOGS=./.preplogs_2d
mkdir -p $LOGS

env_mesh () {  # env-tag -> env OBJ
    case $1 in
        2denv1) echo datasets/3dshape/2d_env1_zup.obj ;;
        2denv4) echo datasets/3dshape/2denv4_zup.obj ;;
    esac
}

current () {  # shape env device -- current pipeline: training set + test set
    local shape=$1 env=$2 dev=$3
    local mesh=$(env_mesh $env) ds=${shape}_${env}
    (
        set -e
        python -u dataprocessing/preprocess_obj.py \
            --env   $mesh \
            --shape datasets/3dshape/${shape}_zup.obj \
            --out   datasets/3dshape/$ds \
            --num_samples 800000 \
            --2d \
            --visualize \
            --batch_size 500 \
            --device $dev
        python -u dataprocessing/preprocess_obj.py \
            --env   $mesh \
            --shape datasets/3dshape/${shape}_zup.obj \
            --out   testing_data/3dshape/$ds \
            --num_samples 1000 \
            --testing_data \
            --offset 0.02 \
            --2d \
            --batch_size 500 \
            --visualize \
            --device $dev
    ) > $LOGS/${ds}.log 2>&1
}

recovered () {  # shape env device -- June-3 pipeline, era defaults
    local shape=$1 env=$2 dev=$3
    local mesh=$(env_mesh $env) ds=${shape}_${env}
    python -u dataprocessing/preprocess_obj_june03.py \
        --env   $mesh \
        --shape datasets/3dshape/${shape}_zup.obj \
        --out   datasets/3dshape/${ds}_june03 \
        --num_samples 400000 \
        --margin 0.1 \
        --offset 0.01 \
        --2d \
        --visualize \
        --batch_size 256 \
        --device $dev > $LOGS/${ds}_june03.log 2>&1
}

work=()
for env in $ENVS; do
    for shape in $SHAPES; do work+=("$shape $env"); done
done

GPUS=(cuda:0 cuda:1 cuda:2)
NG=${#GPUS[@]}

worker () {  # index stage stride  -- walk every `stride`-th dataset
    local w=$1 stage=$2 stride=$3 k=$1 rc dev=${GPUS[$(( $1 % NG ))]}
    while [ $k -lt ${#work[@]} ]; do
        set -- ${work[$k]}
        echo "[$stage $w] start $1_$2 on $dev"
        $stage $1 $2 $dev
        rc=$?
        if [ $rc -eq 0 ]; then echo "[ok]   $stage $1_$2"
        else echo "[FAIL] $stage $1_$2 rc=$rc"; fi
        k=$((k+stride))
    done
}

t0=$(date +%s)

# ── Phase A: current pipeline (training + test sets), ONE job per GPU.
# At --batch_size 500 a single job peaks near 9.5 GiB and already holds the GPU
# at ~90% utilization, so a second one buys throughput only in OOM risk. ──
pids=()
for w in 0 1 2; do worker $w current 3 & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
echo "[phase A done] $(( $(date +%s) - t0 ))s"

# ── Phase B: June-3 pipeline, two jobs per GPU ──
tB=$(date +%s)
pids=()
for w in 0 1 2 3 4 5; do worker $w recovered 6 & pids+=($!); done
for p in "${pids[@]}"; do wait $p; done
echo "[phase B done] $(( $(date +%s) - tB ))s"
echo "[done] total wall time: $(( $(date +%s) - t0 ))s"
echo "--- datasets written ---"
ls -d datasets/3dshape/*_2denv[14] datasets/3dshape/*_2denv[14]_june03 testing_data/3dshape/*_2denv[14] 2>/dev/null | wc -l
grep -l "Traceback" $LOGS/*.log 2>/dev/null || echo "no tracebacks in $LOGS"
