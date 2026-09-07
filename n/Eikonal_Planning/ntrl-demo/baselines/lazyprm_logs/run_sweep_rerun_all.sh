#!/bin/bash
# Full re-run of the LazyPRM sweep -- all 24 3-D shape/env pairs, settings
# identical to the original sweep behind experiments_lazyprm.md (30 s budget,
# roadmap cleared per case, seed 1, no path simplification).  Results go to
# results/ompl_lazyprm_rerun_all/ so neither the original sweep
# (results/ompl_lazyprm/) nor the earlier env1-only re-run
# (results/ompl_lazyprm_rerun/) is touched.
#
# Lives in baselines/lazyprm_logs/, next to baselines/baseline_ompl/.  Run inside
# a pytorchserver container on a mount that covers the repository root, with
# cwd = the main package (ntrl-demo/ntrl-demo) -- that is what datasets/,
# testing_data/ and results/ below are relative to.
# Needs the 1.7.0 bindings venv: /opt/ompl17venv (the 2.0.1 wheel has no LazyPRM).
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=/opt/ompl17venv/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL=$HERE/../baseline_ompl/lazy_prm_eval.py
LOGDIR=$HERE/rerun_all
mkdir -p "$LOGDIR"

run_one() {
    name="$1"; shape="$2"; env="$3"
    if [ -f "$LOGDIR/$name.done" ]; then
        echo "skip $name (already done)"; return
    fi
    echo "start $name"
    $PY "$EVAL" \
        --obj "datasets/3dshape/$shape.obj" \
        --env "datasets/3dshape/$env.obj" \
        --dataPath "testing_data/3dshape/$name" \
        --n 0 --time 30 \
        --out "results/ompl_lazyprm_rerun_all/$name" \
        > "$LOGDIR/$name.log" 2>&1 && touch "$LOGDIR/$name.done"
    echo "finish $name"
}
export -f run_one
export LOGDIR PY EVAL

for env in env1 env2 env3 env4; do
    for shape in rectangle Lshape3d Fshape3d Ashape3d Vshape3d 4shape3d; do
        echo "${shape}_${env} ${shape} ${env}"
    done
done | xargs -P 24 -L 1 bash -c 'run_one "$@"' _
echo "SWEEP COMPLETE"
