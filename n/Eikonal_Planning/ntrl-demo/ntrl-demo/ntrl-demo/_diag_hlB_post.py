"""Post-hoc analysis of a _diag_hlB_collisions.py run.

1. Junction analysis: are collisions concentrated where the two bidirectional
   chains meet (base rate = fraction of all waypoints within +-k of the junction)?
2. Field-only path selector: can the network alone tell a colliding rollout
   from a clean one (path travel time, min / mean local step speed)?  If so, a
   K-restart planner that keeps the best-scoring rollout is a legitimate,
   mesh-free planner-side fix; the mesh-checked 'OR' number is its upper bound.

    python _diag_hlB_post.py --run results/diag_hlB/Ashape3d_env2 \
        --dataPath testing_data/3dshape/Ashape3d_env2 \
        --checkpoint ./Experiments/3dshape/3dshape_09_04_09_09/latest.pt --device cuda:0
"""
import sys
sys.path.append('.')
import os, json, argparse
import numpy as np
import torch
from models.metric import model_train_metric as md

DIM = 6
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--dataPath', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--modelPath', default='./Experiments/3dshape')
p.add_argument('--device', default='cuda')
a = p.parse_args()

R = json.load(open(os.path.join(a.run, 'records.json')))
recs = R['records']
paths = np.load(os.path.join(a.run, 'paths.npy'), allow_pickle=True)
print(f"cases {R['n_cases']}  ok {R['n_ok']}  coll {R['n_coll']}  noconv {R['n_conv_fail']}")

# ── 1. junction ─────────────────────────────────────────────────────────────
for k in (1, 2, 5):
    tot = near = 0
    coll_tot = coll_near = 0
    for r in recs:
        j = r['n_a'] - 1
        idx = np.arange(r['L'])
        m = np.abs(idx - j) <= k
        tot += r['L']; near += int(m.sum())
        for c in r['coll_idx']:
            coll_tot += 1
            coll_near += int(abs(c - j) <= k)
    print(f'junction +-{k}: base rate of waypoints {near / tot:.3f}   colliding-waypoint rate {coll_near / max(coll_tot, 1):.3f}')
fails = [r for r in recs if r['collided']]
print(f'failed paths whose FIRST colliding wp is within +-2 of junction: '
      f'{np.mean([abs(r["coll_idx"][0] - (r["n_a"] - 1)) <= 2 for r in fails]):.3f}')
print(f'failed paths with ANY colliding wp within +-2 of junction: '
      f'{np.mean([any(abs(c - (r["n_a"] - 1)) <= 2 for c in r["coll_idx"]) for r in fails]):.3f}')
print(f'failed paths where ALL colliding wps are within +-3 of junction: '
      f'{np.mean([all(abs(c - (r["n_a"] - 1)) <= 3 for c in r["coll_idx"]) for r in fails]):.3f}')
Ls = np.array([r['L'] for r in recs])
print(f'path length (waypoints) median {np.median(Ls):.0f}')

# ── 2. field-only path score ────────────────────────────────────────────────
womodel = md.Model(a.modelPath, a.dataPath, DIM, [0.0] * DIM, device=a.device)
womodel.load(a.checkpoint)
womodel.network.eval()
segs, owner = [], []
for i, P in enumerate(paths):
    P = np.asarray(P, dtype=np.float32)
    if P.shape[0] < 2:
        continue
    segs.append(np.concatenate([P[:-1], P[1:]], 1)); owner.append(np.full(P.shape[0] - 1, i))
segs = np.concatenate(segs); owner = np.concatenate(owner)
with torch.no_grad():
    X = torch.tensor(segs, device=a.device)
    tau = torch.cat([womodel.function.TravelTimes(X[s:s + 50000]) for s in range(0, len(X), 50000)]).cpu().numpy()
ln = np.linalg.norm(segs[:, DIM:] - segs[:, :DIM], axis=1)
spd = ln / np.maximum(tau, 1e-9)
n = len(paths)
T = np.zeros(n); mn = np.full(n, np.inf); mean_s = np.zeros(n); cnt = np.zeros(n)
for o, t, s, l in zip(owner, tau, spd, ln):
    if l < 1e-6:
        continue
    T[o] += t; mn[o] = min(mn[o], s); mean_s[o] += s; cnt[o] += 1
mean_s /= np.maximum(cnt, 1)
ok = np.array([r['ok'] for r in recs]); coll = np.array([r['collided'] for r in recs])


def auc(score, pos):
    from scipy.stats import rankdata
    r = rankdata(score); npos = pos.sum(); nneg = (~pos).sum()
    return (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


print('\nfield-only separability of colliding vs clean rollouts (AUC; 0.5 = none):')
print(f'  min local step speed (higher=clean)   AUC {auc(mn, ~coll):.3f}   median clean {np.median(mn[~coll]):.3f} coll {np.median(mn[coll]):.3f}')
print(f'  mean local step speed                  AUC {auc(mean_s, ~coll):.3f}   median clean {np.median(mean_s[~coll]):.3f} coll {np.median(mean_s[coll]):.3f}')
print(f'  path travel time (lower=clean)         AUC {auc(-T, ~coll):.3f}   median clean {np.median(T[~coll]):.3f} coll {np.median(T[coll]):.3f}')
