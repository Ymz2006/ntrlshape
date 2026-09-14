"""Append the 2-D sweep results to ../../experiments_ours.md.

Reads the ``success_rate.txt`` each ``evaluate_training_3d_batched.py`` run wrote
under ``results/output_3d/`` and emits one table per pipeline plus a head-to-head
column, in the same shape as the 3-D table already in the file.  Everything above
the ``## 2-D shape task`` heading is left untouched, so re-running it just
refreshes the 2-D half.

    python _make_experiments_2d.py
"""

import os
import re
import argparse

SHAPES = ['rectangle', 'Lshape3d', 'Fshape3d', 'Ashape3d', 'Vshape3d', '4shape3d', 'Tshape3d']
ENVS = ['2denv1', '2denv2', '2denv3', '2denv4']

# The (Tshape3d, 2denv4) dataset was built 2026-08-18 under the pre-env-tag name
# Tshape3d_env4 and renamed to Tshape3d_2denv4 on 2026-09-10, when the 3-D
# Tshape3d_env4 cell took that name.  No alias is needed any more.
DATASET_ALIAS = {}

# success_rate.txt line prefix -> column header
PLANNERS = [
    ('regular/forward', 'SR Forward'),
    ('flipped/reverse', 'SR Reverse'),
    ('either/or', 'SR OR'),
    ('alt/alternate', 'SR Alternate'),
    ('locB/Alternative Bellman', 'SR Alternate Bellman'),
    ('hlB/Alternative Bellman Horizon', 'SR Alternate Bellman Horizon'),
]

# Filled in separately by _make_spread_sd.py (it reads results/spread_sd/), so this
# script only leaves the column in place.
SD = 'SR Alternate Bellman Horizon SD'

parser = argparse.ArgumentParser(description='Write the 2-D half of ../../experiments_ours.md.')
parser.add_argument('--resultPath', default='./results/output_3d')
parser.add_argument('--expCur', default='./Experiments/3dshape_2d')
parser.add_argument('--expJ03', default='./Experiments/3dshape_2d_june03')
parser.add_argument('--out', default='../../experiments_ours.md')
parser.add_argument('--trainTimes', default='./_train_times_2d.tsv',
                    help='Fallback TSV of "<cell>\\t<seconds>" from '
                         '_collect_train_times_2d.sh, used only for a cell whose log '
                         'has no closing "Training time:" line.')
args = parser.parse_args()

MARKER = '## 2-D shape task (`--2d`)'


def read_train_times(datasets, log_root, tsv):
    """{cell: seconds} for finished runs.

    Primary source is the trainer's own closing ``Training time: <n>s`` line in
    <log_root>/logs/<cell>.log -- the same string NTFields writes, so both tables
    measure the same quantity.  A cell still training has no such line; it then
    falls back to the sweep driver's [run ]/[ok] wall clock in ``tsv`` (which runs
    a few seconds long, counting process start-up), and otherwise renders as --.
    """
    out = {}
    for ds in datasets:
        path = os.path.join(log_root, 'logs', ds + '.log')
        if not os.path.exists(path):
            continue
        m = None
        with open(path, errors='ignore') as fh:
            for line in fh:
                hit = re.search(r'Training time: ([\d.]+)s', line)
                if hit:
                    m = hit
        if m:
            out[ds] = float(m.group(1))
    if os.path.exists(tsv):
        for line in open(tsv):
            parts = line.split()
            if len(parts) == 2 and parts[0] not in out:
                out[parts[0]] = float(parts[1])
    return out


def hms(seconds):
    """'1h 02m 03s' from a float count of seconds."""
    total = int(round(seconds))
    return '{}h {:02d}m {:02d}s'.format(total // 3600, total % 3600 // 60, total % 60)


def read_result(ds, suffix=''):
    """Parse one run's success_rate.txt into {header: rate, 'test_cases': n}."""
    path = os.path.join(args.resultPath, ds + suffix, 'success_rate.txt')
    if not os.path.exists(path):
        return None
    text = open(path).read()
    out = {}
    for key, header in PLANNERS:
        m = re.search(re.escape(key) + r'\s*:\s*([\d.]+)', text)
        if m:
            out[header] = float(m.group(1))
    m = re.search(r'test_cases\s*:\s*(\d+)', text)
    out['test_cases'] = int(m.group(1)) if m else None
    m = re.search(r'checkpoint\s*:\s*(\S+)', text)
    out['checkpoint'] = m.group(1) if m else None
    return out if len(out) > 2 else None


def pct(v):
    return '--' if v is None else '{:.1f}%'.format(100 * v)


def table(rows, ckpt_root, times=None):
    times = times or {}
    lines = ['| Env | Model | Train Time (s) | Training Time | '
             + ' | '.join(h for _, h in PLANNERS) + ' | ' + SD + ' | test_cases |',
             '| --- | --- | --- | --- | ' + ' | '.join('---' for _ in PLANNERS) + ' | --- | --- |']
    for ds, res in rows:
        secs = times.get(ds)
        tsec = '--' if secs is None else '{:.0f}'.format(secs)
        thms = '--' if secs is None else hms(secs)
        if res is None:
            lines.append('| {} | `{}` | {} | {} | {} |'.format(
                ds, os.path.join(ckpt_root, ds, 'latest.pt'), tsec, thms,
                ' | '.join(['--'] * (len(PLANNERS) + 2))))
            continue
        lines.append('| {} | `{}` | {} | {} | {} | -- | {} |'.format(
            ds,
            res['checkpoint'] or os.path.join(ckpt_root, ds, 'latest.pt'),
            tsec, thms,
            ' | '.join(pct(res.get(h)) for _, h in PLANNERS),
            res['test_cases'] if res['test_cases'] is not None else '--'))
    return lines


datasets = ['{}_{}'.format(s, e) for e in ENVS for s in SHAPES]
cur = [(ds, read_result(ds)) for ds in datasets]
j03 = [(ds, read_result(ds, '_june03')) for ds in datasets]
train_times = read_train_times(datasets, args.expCur, args.trainTimes)
j03_times = read_train_times(datasets, args.expJ03, os.devnull)

lines = [
    MARKER,
    '',
    'Seven shapes across all four planar environments -- `2denv1_zup.obj` (12 bodies),',
    '`2denv2_zup.obj` (13), `2denv3_zup.obj` (6) and `2denv4_zup.obj` (8), each a',
    '350 x 350 footprint -- 28 cells, trained for 5000 epochs and scored on 1000',
    'held-out start/goal pairs with `evaluate_training_3d_batched.py --2d`.',
    'The commands are in `README.md`; preprocessing ran per env',
    '(`_run_2denv{1,2,3,4}_preprocess.sh`, times in `2d_gen_times.md`), training via',
    '`_run_2d_train_ours.sh`, evaluation via `_run_2d_eval.sh`.',
    '',
    '`Train Time (s)` is the wall clock of that cell\'s 5000-epoch run, taken from',
    "the trainer's closing `Training time:` line -- the same quantity",
    '`experiments_ntfields.md` reports, so the two are directly comparable.',
    '`Training Time` is that figure in h/m/s. A cell still training shows `--`.',
    'The June-3 table has no such column -- that pipeline has not been run on any',
    '2-D cell.',

    '',
    'Both pipelines are scored on the SAME test sets (`testing_data/3dshape/<ds>`,',
    'generated once by the current preprocessor at `--offset 0.02`), so the two tables',
    'differ only in how the training data was generated and which network was fit.',
    '',
    '### Current pipeline (`preprocess_obj.py --2d` + `models/metric`)',
    '',
    '800k training pairs, `--margin 0.05 --offset 0.001`. `Tshape3d_2denv4` trains',
    'from `datasets/3dshape/Tshape3d_2denv4` (built 2026-08-18 as `Tshape3d_env4`,',
    'renamed 2026-09-10 when the 3-D `Tshape3d_env4` cell took that name).',
    '',
]
lines += table(cur, args.expCur, train_times)
lines += [
    '',
    '### Recovered June-3 pipeline (`models/metric_june03`)',
    '',
    'The frozen single-route June-3 network, 5000 epochs, trained by',
    '`_run_2d_june03_train.sh` on the **shared current-pipeline datasets** --',
    'the same `datasets/3dshape/<shape>_<env>` dirs the current rows above use,',
    'not regenerated `_june03` data.  `models/metric_june03/data_mlp.py` reads only',
    '`sampled_points` / `speed` / `normal`, which those dirs already carry, so the',
    'generator is held fixed and a difference here is a statement about the network',
    '(single embedding route, Fourier `B` at std 0.86 vs 0.2) rather than the data.',
    '',
    '`preprocess_obj_june03.py` (400k pairs, `--margin 0.1 --offset 0.01`, 8000 env',
    'points) was NOT used for these rows; it remains available if the generator',
    'ablation is wanted separately.',
    '',
]
lines += table(j03, args.expJ03, j03_times)

# Head-to-head on the forward planner, the strictest of the six.
lines += ['', '### Head to head (`SR Forward`)', '',
          '| Env | current | June-3 | delta |', '| --- | --- | --- | --- |']
deltas = []
for (ds, a), (_, b) in zip(cur, j03):
    va = a.get('SR Forward') if a else None
    vb = b.get('SR Forward') if b else None
    if va is not None and vb is not None:
        d = 100 * (va - vb)
        deltas.append(d)
        ds_delta = '{:+.1f}'.format(d)
    else:
        ds_delta = '--'
    lines.append('| {} | {} | {} | {} |'.format(ds, pct(va), pct(vb), ds_delta))
if deltas:
    ma = [a['SR Forward'] for _, a in cur if a and a.get('SR Forward') is not None]
    mb = [b['SR Forward'] for _, b in j03 if b and b.get('SR Forward') is not None]
    lines.append('| **MEAN** | **{}** | **{}** | **{:+.1f}** |'.format(
        pct(sum(ma) / len(ma)), pct(sum(mb) / len(mb)), sum(deltas) / len(deltas)))
lines.append('')

body = '\n'.join(lines)
text = open(args.out).read() if os.path.exists(args.out) else ''
if MARKER in text:
    text = text[:text.index(MARKER)]
if not text.endswith('\n\n'):
    text = text.rstrip('\n') + '\n\n'
with open(args.out, 'w') as fh:
    fh.write(text + body)

done_cur = sum(1 for _, r in cur if r)
done_j03 = sum(1 for _, r in j03 if r)
print('wrote {}  (current {}/{}, june03 {}/{})'.format(
    args.out, done_cur, len(cur), done_j03, len(j03)))
