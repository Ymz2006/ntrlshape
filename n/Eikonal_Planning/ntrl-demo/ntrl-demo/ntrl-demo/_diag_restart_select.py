"""Simulate a K-restart hlB planner from independent-seed runs of _diag_hlB_collisions.py.

For each test case we have one rollout per run (seed).  Reports the success rate of
  * each run alone,
  * 'OR' (any rollout clean & converged -- the mesh-checked upper bound),
  * field-only selection: keep the rollout with the lowest network travel time
    sum_j tau(wp_j -> wp_j+1) (no mesh queries at all),
  * field-only selection by highest min local step speed.

    python _diag_restart_select.py --runs results/diag_hlB/Ashape3d_env2 \
        results/diag_hlB/variants_Ashape3d_env2/seed1 \
        --dataPath testing_data/3dshape/Ashape3d_env2 \
        --checkpoint ./Experiments/3dshape/3dshape_09_04_09_09/latest.pt --device cuda:1
"""
import sys
sys.path.append('.')
import os, json, argparse
import numpy as np
import torch
from models.metric import model_train_metric as md

DIM = 6
p = argparse.ArgumentParser()
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--dataPath', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--modelPath', default='./Experiments/3dshape')
p.add_argument('--device', default='cuda')
a = p.parse_args()

womodel = md.Model(a.modelPath, a.dataPath, DIM, [0.0] * DIM, device=a.device)
womodel.load(a.checkpoint)
womodel.network.eval()


def score(paths):
    segs, owner = [], []
    for i, P in enumerate(paths):
        P = np.asarray(P, dtype=np.float32)
        segs.append(np.concatenate([P[:-1], P[1:]], 1)); owner.append(np.full(P.shape[0] - 1, i))
    segs = np.concatenate(segs); owner = np.concatenate(owner)
    with torch.no_grad():
        X = torch.tensor(segs, device=a.device)
        tau = torch.cat([womodel.function.TravelTimes(X[s:s + 50000]) for s in range(0, len(X), 50000)]).cpu().numpy()
    ln = np.linalg.norm(segs[:, DIM:] - segs[:, :DIM], axis=1)
    n = len(paths); T = np.zeros(n); mn = np.full(n, np.inf)
    for o, t, l in zip(owner, tau, ln):
        if l < 1e-6:
            continue
        T[o] += t; mn[o] = min(mn[o], l / max(t, 1e-9))
    return T, mn


runs = []
for r in a.runs:
    R = json.load(open(os.path.join(r, 'records.json')))
    P = np.load(os.path.join(r, 'paths.npy'), allow_pickle=True)
    ok = np.array([x['ok'] for x in R['records']])
    case = np.array([x['case'] for x in R['records']])
    T, mn = score(P)
    runs.append(dict(name=r, ok=ok, case=case, T=T, mn=mn))
    print(f'{r}: SR {ok.mean():.3f}')

cases = runs[0]['case']
for r in runs[1:]:
    assert np.array_equal(r['case'], cases)
OK = np.stack([r['ok'] for r in runs])           # (K, N)
T = np.stack([r['T'] for r in runs])
MN = np.stack([r['mn'] for r in runs])
K = len(runs)
print(f'\nK={K} restarts')
print(f'  OR (mesh-checked upper bound)         : {OK.any(0).mean():.3f}')
sel = np.argmin(T, 0)
print(f'  pick lowest field travel time          : {OK[sel, np.arange(OK.shape[1])].mean():.3f}')
sel = np.argmax(MN, 0)
print(f'  pick highest min local step speed      : {OK[sel, np.arange(OK.shape[1])].mean():.3f}')
# hybrid: travel time, but only among rollouts that converged (min_dis known to the planner)
