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
ENVS = ['2denv4', '2denv1']

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
args = parser.parse_args()

MARKER = '## 2-D shape task (`--2d`)'


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


def table(rows, ckpt_root):
    lines = ['| Env | Model | ' + ' | '.join(h for _, h in PLANNERS) + ' | ' + SD + ' | test_cases |',
             '| --- | --- | ' + ' | '.join('---' for _ in PLANNERS) + ' | --- | --- |']
    for ds, res in rows:
        if res is None:
            lines.append('| {} | `{}` | {} |'.format(
                ds, os.path.join(ckpt_root, ds, 'latest.pt'),
                ' | '.join(['--'] * (len(PLANNERS) + 2))))
            continue
        lines.append('| {} | `{}` | {} | -- | {} |'.format(
            ds,
            res['checkpoint'] or os.path.join(ckpt_root, ds, 'latest.pt'),
            ' | '.join(pct(res.get(h)) for _, h in PLANNERS),
            res['test_cases'] if res['test_cases'] is not None else '--'))
    return lines


datasets = ['{}_{}'.format(s, e) for e in ENVS for s in SHAPES]
cur = [(ds, read_result(ds)) for ds in datasets]
j03 = [(ds, read_result(ds, '_june03')) for ds in datasets]

lines = [
    MARKER,
    '',
    'Seven shapes across the two planar environments -- `2denv4_zup.obj` (8 bodies) and',
    'the denser `2d_env1_zup.obj` (12 bodies) -- each trained for 5000 epochs and scored',
    'on 1000 held-out start/goal pairs with `evaluate_training_3d_batched.py --2d`.',
    'The commands are in `README.md`; the sweep drivers are `_run_2d_preprocess.sh`,',
    '`_run_2d_train.sh` and `_run_2d_eval.sh`.',
    '',
    'Both pipelines are scored on the SAME test sets (`testing_data/3dshape/<ds>`,',
    'generated once by the current preprocessor at `--offset 0.02`), so the two tables',
    'differ only in how the training data was generated and which network was fit.',
    '',
    '### Current pipeline (`preprocess_obj.py --2d` + `models/metric`)',
    '',
    '800k training pairs, `--margin 0.05 --offset 0.001`.',
    '',
]
lines += table(cur, args.expCur)
lines += [
    '',
    '### Recovered June-3 pipeline (`preprocess_obj_june03.py --2d` + `models/metric_june03`)',
    '',
    "400k training pairs at the June-3 era's `--margin 0.1 --offset 0.01` and 8000 env",
    'points, fit with the frozen single-route network.',
    '',
]
lines += table(j03, args.expJ03)

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
