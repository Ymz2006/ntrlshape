"""Add/refresh the `SR Alternate Bellman Horizon SD` column of ../../experiments_ours.md.

SD = hlB (Alternate Bellman Horizon) with the executed MPPI step biased along the
least-squares gradient of the SAMPLED speeds, i.e. `_spread_planner.py
--steer-bias 0.5 --no-gate` (see `_run_spread_sd.sh`).  The column is inserted right
after `SR Alternate Bellman Horizon` in every planner table; rows with no run yet
get `--`.  Re-running just refreshes the values.

    python _make_spread_sd.py
"""

import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resultPath', default='./results/spread_sd')
parser.add_argument('--out', default='../../experiments_ours.md')
args = parser.parse_args()

BASE = 'SR Alternate Bellman Horizon'
NEW = BASE + ' SD'


def read_result(ds):
    path = os.path.join(args.resultPath, ds, 'records.json')
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    return '{:.1f}%'.format(100 * r['n_ok'] / r['n']) if r['n'] else None


def cells(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


lines = open(args.out).read().split('\n')
out = []
cols = None                                    # header cells of the table being walked
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('|') and BASE in line and NEW not in line:
        cols = cells(line)
        k = cols.index(BASE) + 1
        cols.insert(k, NEW)
        out.append('| ' + ' | '.join(cols) + ' |')
        sep = cells(lines[i + 1])
        sep.insert(k, '---')
        out.append('| ' + ' | '.join(sep) + ' |')
        i += 2
        continue
    if line.startswith('|') and NEW in line:
        cols = cells(line)
        out.append(line)
        i += 1
        continue
    if cols is not None and line.startswith('|'):
        row = cells(line)
        k = cols.index(NEW)
        val = read_result(row[0].split()[0]) or '--'
        if len(row) == len(cols) - 1:                  # column not there yet
            row.insert(k, val)
        elif len(row) == len(cols) and row[k] != '---':
            row[k] = val                               # refresh in place
        out.append('| ' + ' | '.join(row) + ' |')
        i += 1
        continue
    if not line.startswith('|'):
        cols = None
    out.append(line)
    i += 1

open(args.out, 'w').write('\n'.join(out))
print('wrote', args.out)
