#!/bin/bash
# Sweep RRT-Connect (OMPL) over every 3-D shape/env pair in testing_data/3dshape.
# Run inside the pytorchserver container with cwd = /workspace/ntrl-demo.
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

LOGDIR=/workspace/rrt_logs
mkdir -p "$LOGDIR"

run_one() {
    name="$1"; shape="$2"; env="$3"
    if [ -f "$LOGDIR/$name.done" ]; then
        echo "skip $name (already done)"; return
    fi
    echo "start $name"
    python ../baseline_ompl/rrt_connect_eval.py \
        --obj "datasets/3dshape/$shape.obj" \
        --env "datasets/3dshape/$env.obj" \
        --dataPath "testing_data/3dshape/$name" \
        --n 0 --time 30 \
        --out "results/ompl_rrtconnect/$name" \
        > "$LOGDIR/$name.log" 2>&1 && touch "$LOGDIR/$name.done"
    echo "finish $name"
}
export -f run_one
export LOGDIR

for env in env1 env2 env3 env4; do
    for shape in rectangle Lshape3d Fshape3d Ashape3d Vshape3d 4shape3d; do
        echo "${shape}_${env} ${shape} ${env}"
    done
done | xargs -P 24 -L 1 bash -c 'run_one "$@"' _
echo "SWEEP COMPLETE"
