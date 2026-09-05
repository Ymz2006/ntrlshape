"""Collect the NTFields 3-D shape runs into ``experiments.md``.

Scans the output root written by ``train_3dshape_all.sh`` and emits one row per
environment -- checkpoint path, epochs and final training loss -- in the same
env order as the main repo's ``ntrl-demo/experiments.md``.  Re-run it at any
time; environments that have not finished training yet are listed as pending.

Where ``3d_plan.py`` has also inferred paths for an environment (its
``plan_summary.txt`` under ``--planPath``), the row gains that model's success
rate, per-case planning time and path length -- mean +/- sd -- in the same form
and the same metrics as ``ntrl-demo/RRT_experiments.md``, so the two tables can
be read side by side.  Environments with no planning run yet show ``--``.

    python make_experiments_md.py
"""

import argparse
import os
import re
import glob

ENVS = [
    'rectangle_env1', 'Lshape3d_env1', 'Fshape3d_env1', 'Ashape3d_env1', 'Vshape3d_env1', '4shape3d_env1',
    'rectangle_env2', 'Lshape3d_env2', 'Fshape3d_env2', 'Ashape3d_env2', 'Vshape3d_env2', '4shape3d_env2',
    'rectangle_env3', 'Lshape3d_env3', 'Fshape3d_env3', 'Ashape3d_env3', 'Vshape3d_env3', '4shape3d_env3',
    'rectangle_env4', 'Lshape3d_env4', 'Fshape3d_env4', 'Ashape3d_env4', 'Vshape3d_env4', '4shape3d_env4',
    'Tshape3d_env4',
    # Present as datasets but absent from the main repo's experiments.md.
    'Tshape3d_env1', 'Lcouch_Corozal',
]

parser = argparse.ArgumentParser(description='Write the NTFields experiments table.')
parser.add_argument('--modelPath', default='./outputs/3dshape')
parser.add_argument('--planPath', default='./outputs/3dplan',
                    help='Root of the 3d_plan.py runs; <planPath>/<env>/'
                         'plan_summary.txt supplies the path-inference columns.')
parser.add_argument('--epochs', type=int, default=4000)
parser.add_argument('--out', default='./experiments.md')
args = parser.parse_args()

CKPT = 'Model_Epoch_{:05d}_ValLoss_*.pt'.format(args.epochs)


def run_dir(env):
    """Newest run folder for ``env`` that reached the target epoch count."""
    hits = sorted(glob.glob(os.path.join(args.modelPath, env, CKPT)),
                  key=os.path.getmtime)
    return os.path.dirname(hits[-1]) if hits else None


def log_stats(env):
    """(final loss, train seconds) parsed from the per-env training log."""
    path = os.path.join(args.modelPath, 'logs', env + '.log')
    loss = seconds = None
    if os.path.exists(path):
        with open(path, errors='ignore') as fh:
            for line in fh:
                m = re.match(r'Epoch = (\d+) -- Loss = ([\d.e+-]+)', line)
                if m and int(m.group(1)) == args.epochs:
                    loss = float(m.group(2))
                m = re.search(r'Training time: ([\d.]+)s', line)
                if m:
                    seconds = float(m.group(1))
    return loss, seconds


def plan_stats(env):
    """Path-inference numbers for ``env`` from its 3d_plan.py summary.

    Returns (success_rate, time_mean, time_std, len_mean, len_std, n_cases) with
    ``None`` in every slot when the environment has not been planned yet.
    """
    path = os.path.join(args.planPath, env, 'plan_summary.txt')
    if not os.path.exists(path):
        return None
    keys = ('success_rate', 'time_mean', 'time_std',
            'path_length_mean', 'path_length_std', 'test_cases_scored')
    got = {}
    with open(path, errors='ignore') as fh:
        for line in fh:
            m = re.match(r'(\w+)\s*:\s*([\d.eE+-]+)', line)
            if m and m.group(1) in keys and m.group(1) not in got:
                got[m.group(1)] = float(m.group(2))
    if not all(k in got for k in keys):
        return None
    return (got['success_rate'], got['time_mean'], got['time_std'],
            got['path_length_mean'], got['path_length_std'],
            int(got['test_cases_scored']))


rows, pending, planned = [], [], []
for env in ENVS:
    folder = run_dir(env)
    if folder is None:
        pending.append(env)
        continue
    loss, seconds = log_stats(env)
    plan = plan_stats(env)
    if plan is None:
        rate = ptime = plen = '--'
    else:
        rate = '{:.1%}'.format(plan[0])
        ptime = '{:.3f} ± {:.3f}'.format(plan[1], plan[2])
        plen = '{:.3f} ± {:.3f}'.format(plan[3], plan[4])
        planned.append(env)
    rows.append((env,
                 os.path.join(folder, 'latest.pt'),
                 '{:.4e}'.format(loss) if loss is not None else '--',
                 '{:.0f}'.format(seconds) if seconds is not None else '--',
                 rate, ptime, plen))

lines = [
    '# Experiments -- NTFields on the 3-D shape task',
    '',
    'Trained with `train_3dshape_all.sh` ({} epochs x 5 batches of 2000 per '
    'environment -- the same 20,000 optimizer steps the ntrl-demo baselines get) '
    'on the same datasets as the main repo\'s `ntrl-demo/experiments.md`.'.format(args.epochs),
    '',
    'The last three columns come from `3d_plan.py`, which plans the 1000 start/goal',
    'pairs of `testing_data/3dshape/<env>` with the MPPI controller of',
    '`ntrl-demo/evaluate_training_3d.py` driven by this checkpoint. A case succeeds',
    'when the rollout reaches the 0.01 goal ball AND the placed shape is',
    'collision-free at every waypoint (`preprocess_obj`\'s point-in-tet test against a',
    '50k-point sampling of the whole environment mesh, walls included). Time is the',
    'per-case wall clock of the rollout over all scored cases; path length is OMPL\'s',
    'SE(3) metric (`sum ||dt|| + acos(|q.q\'|)`) over the successful ones -- the same',
    'two quantities `RRT_experiments.md` reports, measured the same way.',
    '',
    'NTFields replays an epoch whose mean loss exceeds `--repeat-ratio` times the',
    'previous one, measured against a `prev_diff` that starts at 1.0. The 6-D shape',
    'data starts near 2.0, so the shipped 1.2 rejects epoch 1 forever; these runs use',
    'a relaxed 2.5 with a 20-replay cap. Losses are therefore not comparable in scale',
    'to the ntrl-demo tables -- the loss functions differ (NTFields is isotropic and',
    'ignores `normal.npy`).',
    '',
    '| Env | Model | Epochs | Final Loss | Train Time (s) | Success Rate | '
    'Path Time mean ± sd (s) | Path Length mean ± sd |',
    '| --- | --- | --- | --- | --- | --- | --- | --- |',
]
for env, ckpt, loss, seconds, rate, ptime, plen in rows:
    lines.append('| {} | `{}` | {} | {} | {} | {} | {} | {} |'.format(
        env, ckpt, args.epochs, loss, seconds, rate, ptime, plen))
if pending:
    lines += ['', 'Pending (no epoch-{} checkpoint yet): {}'.format(
        args.epochs, ', '.join('`{}`'.format(e) for e in pending))]
not_planned = [env for env, *_ in rows if env not in planned]
if not_planned:
    lines += ['', 'Not planned yet (no `{}/<env>/plan_summary.txt`): {}'.format(
        args.planPath.rstrip('/'),
        ', '.join('`{}`'.format(e) for e in not_planned))]
lines.append('')

with open(args.out, 'w') as fh:
    fh.write('\n'.join(lines))
print('wrote {} ({} trained, {} pending, {} planned)'.format(
    args.out, len(rows), len(pending), len(planned)))
