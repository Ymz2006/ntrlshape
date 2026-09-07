#!/bin/bash
# Re-run of the LazyPRM sweep, env1 only -- identical settings to the first
# sweep (30 s budget, roadmap cleared per case, seed 1), separate output dir so
# the original results/ompl_lazyprm numbers stay intact.
# Lives in baselines/lazyprm_logs/; same run context as run_sweep.sh there
# (cwd = the main package, ntrl-demo/ntrl-demo).
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PY=/opt/ompl17venv/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL=$HERE/../baseline_ompl/lazy_prm_eval.py
LOGDIR=$HERE/rerun_env1
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
        --out "results/ompl_lazyprm_rerun/$name" \
        > "$LOGDIR/$name.log" 2>&1 && touch "$LOGDIR/$name.done"
    echo "finish $name"
}
export -f run_one
export LOGDIR PY EVAL

for shape in rectangle Lshape3d Fshape3d Ashape3d Vshape3d 4shape3d; do
    echo "${shape}_env1 ${shape} env1"
done | xargs -P 6 -L 1 bash -c 'run_one "$@"' _
echo "SWEEP COMPLETE"
