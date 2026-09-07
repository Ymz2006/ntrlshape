"""Rebuild ``experiments_lazyprm_rerun.md`` from the LazyPRM re-run results.

Same table and sections as ``experiments_lazyprm.md``, filled from whatever
``results/ompl_lazyprm_rerun_all/<name>/`` directories are complete right now --
rows for configs still running are left blank -- so this can be re-run
repeatedly while the sweep is in flight.  Also refreshes the Lazy PRM (re-run)
row of the evaluation status board in ``MASTER_EXPERIMENTS_README.md``.

    python baselines/baseline_ompl/make_lazyprm_md.py
"""

import argparse
import csv
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_PKG = os.path.join(_ROOT, 'ntrl-demo', 'ntrl-demo')

SHAPES = ['rectangle', 'Lshape3d', 'Fshape3d', 'Ashape3d', 'Vshape3d', '4shape3d']
ENVS = ['env1', 'env2', 'env3', 'env4']
NAMES = ['%s_%s' % (s, e) for e in ENVS for s in SHAPES]


def read_summary(d):
    """Parse ``lazy_prm_success_rate.txt`` into a ``key -> first token`` dict."""
    path = os.path.join(d, 'lazy_prm_success_rate.txt')
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path):
        if ':' not in line:
            continue
        key, _, rest = line.partition(':')
        key = key.strip()
        if not key or key.startswith('#'):
            continue
        out[key] = rest.strip()
    # A dir written by a still-running process may be truncated.
    return out if 'path_length_std' in out else None


def num(summary, key):
    return float(summary[key].split()[0])


def failure_times(d):
    """Wall-clock times of the failed cases in ``lazy_prm_cases.csv``."""
    path = os.path.join(d, 'lazy_prm_cases.csv')
    if not os.path.exists(path):
        return []
    times = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row['endpoints_valid'] == '1' and row['success'] == '0':
                times.append(float(row['time_s']))
    return times


def collect(results_dir):
    rows = {}
    for name in NAMES:
        s = read_summary(os.path.join(results_dir, name))
        if s is not None:
            rows[name] = s
    return rows


def table(rows):
    out = ['| Env | Success Rate | Path Time mean ± sd (s) | Path Length mean ± sd |',
           '| --- | --- | --- | --- |']
    for name in NAMES:
        s = rows.get(name)
        if s is None:
            out.append('| %s |  |  |  |' % name)
            continue
        rate = num(s, 'successes') / num(s, 'test_cases_scored') * 100.0
        out.append('| %s | %.1f%% | %.3f ± %.3f | %.3f ± %.3f |' % (
            name, rate,
            num(s, 'time_mean_successful'), num(s, 'time_std_successful'),
            num(s, 'path_length_mean'), num(s, 'path_length_std')))
    return '\n'.join(out)


def failure_section(rows, results_dir):
    if not rows:
        return ('No configs have finished yet, so there is nothing to say about '
                'the failures.\n')
    counts = {k: 0 for k in ('no_solution_in_time', 'over_time_limit',
                             'path_in_collision')}
    for s in rows.values():
        for k in counts:
            counts[k] += int(num(s, k))
    times = []
    for name in rows:
        times.extend(failure_times(os.path.join(results_dir, name)))

    n = len(rows)
    lead = ('Every failure so far is the 30 s budget running out.' if
            counts['path_in_collision'] == 0 else
            '%d returned paths were in collision.' % counts['path_in_collision'])
    text = [
        '%s  `path_in_collision` is %d across the %d config%s finished so far.'
        % (lead, counts['path_in_collision'], n, '' if n == 1 else 's'),
        '',
        '| failure mode | count across the %d finished config%s | what it means |'
        % (n, '' if n == 1 else 's'),
        '| --- | --- | --- |',
        '| `no_solution_in_time` | %d | `solve()` used the full 30 s and returned no exact solution |'
        % counts['no_solution_in_time'],
        '| `over_time_limit` | %d | an exact path came back, but wall clock passed 30 s getting there |'
        % counts['over_time_limit'],
        '| `path_in_collision` | %d | %s |'
        % (counts['path_in_collision'],
           'never happened' if counts['path_in_collision'] == 0 else 'a returned path hit an obstacle'),
    ]
    if times:
        text += ['',
                 'Failed cases cluster around the cap: min %.2f s, median %.2f s, '
                 'max %.2f s.  Nothing gives up early; it only runs out of clock.'
                 % (min(times), statistics.median(times), max(times))]
    return '\n'.join(text) + '\n'


DOC = """# LazyPRM (OMPL) Baseline Experiments -- Full Re-run

> Collected at the repository root. Every unqualified path below (`datasets/`, `Experiments/`, `outputs/`, `results/`, `tests/`, `train/`, ...)
> is relative to `ntrl-demo/ntrl-demo/`, where this table's runs live.

LazyPRM in SE(3) run on the same 3-D test sets the learned planner is evaluated
on (`testing_data/3dshape/<shape>_<env>`), so the numbers line up with
`experiments_ours.md` and `experiments_rrt_connect.md` case for case.  Produced by
`baselines/baseline_ompl/lazy_prm_eval.py`.

This is a **full re-run of all 24 configs** with settings identical to
`experiments_lazyprm.md` (30 s budget, roadmap cleared per case, seed 1, no path
simplification).  Results land in `results/ompl_lazyprm_rerun_all/` so the
original sweep and the earlier env1-only re-run (`experiments_lazyprm_env1_rerun.md`)
are both untouched.

**Progress: {done}/24 configs finished.**{running}  This file is regenerated by
`baselines/baseline_ompl/make_lazyprm_md.py` each time a config lands, so blank
rows are still running.

{table}

## How these were produced

```
python ../../baselines/baseline_ompl/lazy_prm_eval.py \\
    --obj      datasets/3dshape/<shape>.obj \\
    --env      datasets/3dshape/<env>.obj \\
    --dataPath testing_data/3dshape/<shape>_<env> \\
    --n 0 --time 30 \\
    --out      results/ompl_lazyprm_rerun_all/<shape>_<env>
```

Sweep driver: `baselines/lazyprm_logs/run_sweep_rerun_all.sh` (all 24 configs in
parallel, one process each, `OMP_NUM_THREADS=1`).  Per-config logs land in
`baselines/lazyprm_logs/rerun_all/<name>.log` and the full summaries in
`results/ompl_lazyprm_rerun_all/<name>/`.

**OMPL version**: this needs the **1.7.0** python bindings, in a venv at
`/opt/ompl17venv` (`pip install ompl==1.7.0`).  The 2.0.1 wheel used for
`experiments_rrt_connect.md` exposes only 14 planners and has no lazy variants.  The
1.7.0 Boost.Python bindings also differ from 2.0.1 in two places the script
handles: the validity checker must be wrapped in `ob.StateValidityCheckerFn`,
and `setStartAndGoalStates` takes a `ScopedState` rather than a raw `State*`.
They additionally corrupt the heap on interpreter teardown, so the script
`os._exit(0)`s once its output files are flushed.

## Notes on the numbers

- **Test cases**: all 1000 start/goal pairs per config, the same
  `sampled_points.npy` the learned planner and RRT-Connect are scored on.
- **Success rate**: fraction of *scored* pairs solved.  Pairs whose start or
  goal is already in collision are unplannable by construction and are excluded
  from the denominator (0-2 per config, identical to the RRT-Connect run).
- **Time**: wall-clock `ss.solve()` time over the successful cases only, so it
  is directly comparable with the path-length column.  Measured with 24
  concurrent single-threaded processes on a 36-core host, so treat the timings
  as relative rather than best-case single-run latency.
- **Path length**: OMPL's SE(3) metric (weighted translation + rotation) in the
  normalized frame, measured on the raw planner output -- no path
  simplification (`--simplify` off), matching the RRT-Connect run.  Raw roadmap
  paths are jagged, so these lengths are roughly 2-3x the RRT-Connect ones and
  should not be read as a path-quality verdict on LazyPRM.
- **Single-query protocol**: the roadmap is cleared between test cases
  (`SimpleSetup.clear()`), so each case is planned from scratch exactly like the
  RRT-Connect baseline.  This deliberately gives up LazyPRM's multi-query
  advantage in a fixed environment; `lazy_prm_eval.py --reuse` keeps the roadmap
  across cases if the amortized number is wanted instead.

## How the failures fail

{failures}
"""


def render(results_dir, logdir):
    rows = collect(results_dir)
    running = sorted(n for n in NAMES
                     if n not in rows
                     and os.path.exists(os.path.join(logdir, n + '.log')))
    note = ''
    if len(running) > 6:
        note = '  %d still running.' % len(running)
    elif running:
        note = ('  Running: %s.' % ', '.join('`%s`' % n for n in running))
    return DOC.format(done=len(rows), running=note, table=table(rows),
                      failures=failure_section(rows, results_dir))


def update_master(master, rows, running):
    """Refresh the Lazy PRM (re-run) row of the 3-D evaluation status board."""
    if not os.path.exists(master):
        return False
    lines = open(master).read().split('\n')

    def cell(name):
        if name in rows:
            return '✅'
        return '🟡' if name in running else '⬜'

    row = '| **Lazy PRM (re-run)** | ' + ' | '.join(cell(n) for n in NAMES) + ' |'

    # The 3-D evaluation table is the first "### 3-D section" after "## Evaluation".
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == '## Evaluation')
        start = next(i for i in range(start, len(lines))
                     if lines[i].startswith('### 3-D section'))
        anchor = next(i for i in range(start, len(lines))
                      if lines[i].startswith('| **Lazy PRM** |'))
    except StopIteration:
        return False

    if anchor + 1 < len(lines) and lines[anchor + 1].startswith('| **Lazy PRM (re-run)** |'):
        lines[anchor + 1] = row
    else:
        lines.insert(anchor + 1, row)

    # Point the report-file table at the new document.
    old = ('| Lazy PRM | evaluation | `experiments_lazyprm.md`, '
           '`experiments_lazyprm_env1_rerun.md` (env1 re-run) |')
    new = ('| Lazy PRM | evaluation | `experiments_lazyprm.md`, '
           '`experiments_lazyprm_env1_rerun.md` (env1 re-run), '
           '`experiments_lazyprm_rerun.md` (full re-run) |')
    lines = [new if l == old else l for l in lines]

    open(master, 'w').write('\n'.join(lines))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',
                   default=os.path.join(_PKG, 'results', 'ompl_lazyprm_rerun_all'))
    p.add_argument('--logdir',
                   default=os.path.join(_ROOT, 'baselines', 'lazyprm_logs', 'rerun_all'))
    p.add_argument('--out',
                   default=os.path.join(_ROOT, 'experiments_lazyprm_rerun.md'))
    p.add_argument('--master',
                   default=os.path.join(_ROOT, 'MASTER_EXPERIMENTS_README.md'))
    p.add_argument('--no-master', action='store_true',
                   help='skip the MASTER_EXPERIMENTS_README.md status row')
    args = p.parse_args()

    rows = collect(args.results)
    running = sorted(n for n in NAMES
                     if n not in rows
                     and os.path.exists(os.path.join(args.logdir, n + '.log')))

    open(args.out, 'w').write(render(args.results, args.logdir))
    touched = args.out
    if not args.no_master and update_master(args.master, rows, running):
        touched += ', ' + args.master
    print('%d/24 done, %d running -> %s' % (len(rows), len(running), touched))


if __name__ == '__main__':
    main()
