#!/usr/bin/env bash
# 2-D test sets, both variants.
#
#   1. the one missing standard set (offset 0.02): Tshape3d_2denv4 -- every other
#      2-D cell got its test set alongside its training set in the env sweeps.
#   2. the tight variant (offset 0.005) for all 28 2-D cells, via
#      dataprocessing/make_tight_testing_data.py, which reads the --testing_data
#      blocks out of README.md so env/shape/flags stay in sync with the docs.
#      Existing 3-D *_tight sets are skipped, not regenerated.
#
# --batch-size 500 rather than the script's 1000 default: the cards carry
# unrelated load, and batch size only chunks the rejection sampler.
set -u
cd /workspace/ntrl-demo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGS=./.preplogs_2dtest
mkdir -p $LOGS

echo "[1/2] standard test set: Tshape3d_2denv4"
python -u dataprocessing/preprocess_obj.py \
    --env   datasets/3dshape/2denv4_zup.obj \
    --shape datasets/3dshape/Tshape3d_zup.obj \
    --out   testing_data/3dshape/Tshape3d_2denv4 \
    --num_samples 1000 \
    --testing_data \
    --offset 0.02 \
    --2d \
    --batch_size 500 \
    --visualize \
    --device cuda:0 > $LOGS/Tshape3d_2denv4.log 2>&1 \
  && echo "[ok]   Tshape3d_2denv4 (standard)" || echo "[FAIL] Tshape3d_2denv4 (standard)"

echo "[2/2] tight sets (offset 0.005) across cuda:0,1,2"
python -u dataprocessing/make_tight_testing_data.py \
    --offset 0.005 \
    --num-samples 1000 \
    --batch-size 500 \
    --jobs 3 \
    --devices cuda:0,cuda:1,cuda:2
echo "[done] 2-D test sets"
