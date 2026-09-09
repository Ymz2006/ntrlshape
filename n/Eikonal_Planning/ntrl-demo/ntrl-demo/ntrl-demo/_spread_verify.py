"""Verify: is the SPREAD of sample-estimated speeds a near-obstacle signal?

At each waypoint of a _diag_hlB_collisions.py run (true clearance known from
waypoints.npz), draw the planner's own displacement samples dP_i (|dP| <= step),
estimate the speed of each as  s_i = |dP_i| / tau(cur -> cur + dP_i)  -- exactly
what the Bellman local term already computes -- and derive per-waypoint features:

    s_mean, s_std, cv = s_std / s_mean, s_min, s_max / s_min,
    g = least-squares slope of s_i on dP_i (local 'speed gradient', points AWAY
        from the obstacle if the hypothesis holds), |g| and the fit R^2.

Done for three samplers: planner-like 6-D, translation-only, rotation-only.
Reports each feature binned by true clearance, the AUC of each feature for
'clearance < 0.005' and 'penetrating', whether cv adds to the gradient-based
predicted speed (logistic fit), and -- for near-obstacle waypoints -- the cosine
between the translational part of g and the numerical clearance gradient.

    python _spread_verify.py --run results/diag_hlB/Ashape3d_env2 \
        --dataPath testing_data/3dshape/Ashape3d_env2 \
        --checkpoint ./Experiments/3dshape/3dshape_09_04_09_09/latest.pt --device cuda:1
"""
import sys
sys.path.append('.')
import os, json, argparse
import numpy as np
import torch
import igl
from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import load_obj, _rotvec_to_matrix_np

DIM = 6
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--dataPath', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--modelPath', default='./Experiments/3dshape')
p.add_argument('--device', default='cuda')
p.add_argument('--step', type=float, default=0.015)
p.add_argument('--samples', type=int, default=50)
p.add_argument('--max-wp', type=int, default=30000)
p.add_argument('--normal-check', type=int, default=1500)
a = p.parse_args()
torch.manual_seed(0); rs = np.random.RandomState(0)

womodel = md.Model(a.modelPath, a.dataPath, DIM, [0.0] * DIM, device=a.device)
womodel.load(a.checkpoint); womodel.network.eval()

W = np.load(os.path.join(a.run, 'waypoints.npz'))
paths = np.load(os.path.join(a.run, 'paths.npy'), allow_pickle=True)
clear, case, pos, sp_full = W['clear'], W['case'], W['pos'], W['sp_full']
cfg = np.concatenate([np.asarray(P, dtype=np.float32) for P in paths], 0)
assert len(cfg) == len(clear)
# stratified subsample: keep every near-obstacle waypoint, thin the free ones
near = clear < 0.01
keep = np.nonzero(near)[0]
free = np.nonzero(~near)[0]
n_free = max(a.max_wp - len(keep), 5000)
keep = np.sort(np.concatenate([keep, rs.choice(free, min(n_free, len(free)), replace=False)]))
cfg, clear, sp_full = cfg[keep], clear[keep], sp_full[keep]
N = len(cfg)
print(f'waypoints analysed: {N}  (near<0.01: {near[keep].sum()})')


def sample_dP(N, S, mode):
    dP = torch.randn(N, S, DIM, device=a.device)
    if mode == 'trans':
        dP[..., 3:] = 0
    elif mode == 'rot':
        dP[..., :3] = 0
    dP = dP * a.step  # matches planner: step*N(0,1) then clamp |dP| <= step
    n = torch.norm(dP, dim=2, keepdim=True)
    dP = dP / (torch.clamp(n, min=a.step) / a.step)
    return dP


def features(mode):
    X = torch.tensor(cfg, device=a.device)
    out = {k: [] for k in ['mean', 'std', 'cv', 'min', 'ratio', 'gnorm', 'r2']}
    G = []
    with torch.no_grad():
        for s in range(0, N, 4000):
            x = X[s:s + 4000]; B = x.shape[0]
            dP = sample_dP(B, a.samples, mode)
            src = x[:, None, :].expand(B, a.samples, DIM)
            pairs = torch.cat([src, src + dP], 2).reshape(-1, 2 * DIM)
            tau = womodel.function.TravelTimes(pairs).reshape(B, a.samples)
            ln = torch.norm(dP, dim=2)
            sp = ln / tau.clamp(min=1e-9)
            m = sp.mean(1); sd = sp.std(1)
            out['mean'].append(m); out['std'].append(sd); out['cv'].append(sd / m)
            out['min'].append(sp.min(1).values); out['ratio'].append(sp.max(1).values / sp.min(1).values)
            # least squares  sp_i - mean = g . dP_i
            A = dP; y = (sp - m[:, None])[..., None]
            AtA = A.transpose(1, 2) @ A + 1e-8 * torch.eye(DIM, device=a.device)
            g = torch.linalg.solve(AtA, A.transpose(1, 2) @ y)[..., 0]     # (B, DIM)
            pred = (A @ g[..., None])[..., 0]
            r2 = 1 - ((y[..., 0] - pred) ** 2).sum(1) / ((y[..., 0]) ** 2).sum(1).clamp(min=1e-12)
            out['gnorm'].append(torch.norm(g, dim=1)); out['r2'].append(r2); G.append(g)
    out = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    return out, torch.cat(G).cpu().numpy()


def auc(score, posm):
    from scipy.stats import rankdata
    r = rankdata(score); npos = posm.sum(); nneg = (~posm).sum()
    return (r[posm].sum() - npos * (npos + 1) / 2) / (npos * nneg)


edges = [-1, -0.005, 0, 0.005, 0.01, 0.02, 0.03, 0.05, 1]
results = {}
for mode in ['6d', 'trans', 'rot']:
    f, G = features(mode)
    results[mode] = (f, G)
    print(f'\n===== sampler: {mode}  (S={a.samples}, |dP|<={a.step}) =====')
    print('  clearance bin        n     s_mean   s_std    cv      s_min   max/min  |g|     R2     pred_speed(grad)')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (clear >= lo) & (clear < hi)
        if m.sum() < 10:
            continue
        print(f'  [{lo:+.3f},{hi:+.3f})  {m.sum():6d}   {np.median(f["mean"][m]):.3f}    {np.median(f["std"][m]):.4f}   {np.median(f["cv"][m]):.3f}   {np.median(f["min"][m]):.3f}   {np.median(f["ratio"][m]):.2f}     {np.median(f["gnorm"][m]):.2f}    {np.median(f["r2"][m]):.2f}   {np.median(sp_full[m]):.3f}')
    for name, posm in [('clearance<0.005', clear < 0.005), ('penetrating', clear < 0)]:
        print(f'  AUC for {name}:  cv {auc(f["cv"], posm):.3f}  s_std {auc(f["std"], posm):.3f}  max/min {auc(f["ratio"], posm):.3f}  |g| {auc(f["gnorm"], posm):.3f}  '
              f'-s_mean {auc(-f["mean"], posm):.3f}  -s_min {auc(-f["min"], posm):.3f}  -pred_speed {auc(-sp_full, posm):.3f}')
    # does cv add to predicted speed?  logistic regression, 5-fold
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
    except ImportError:
        LogisticRegression = None
    posm = clear < 0.005
    for feats, label in [((-sp_full,), 'pred_speed only'), ((-sp_full, f['cv']), 'pred_speed + cv'),
                         ((-f['mean'], f['cv']), 's_mean + cv'), ((-sp_full, -f['mean'], f['cv'], f['gnorm']), 'all')]:
        if LogisticRegression is None:
            print('    (sklearn missing: logistic combo skipped)'); break
        Xf = np.stack(feats, 1); Xf = (Xf - Xf.mean(0)) / Xf.std(0)
        pr = cross_val_predict(LogisticRegression(max_iter=500), Xf, posm, cv=5, method='predict_proba')[:, 1]
        print(f'    logistic {label:18s}: AUC {auc(pr, posm):.3f}')

# ── direction check: does g (trans part) point away from the obstacle? ────────
with open(os.path.join(a.dataPath, 'meta.json')) as fh:
    meta = json.load(fh)
env_scale = float(meta['env_scale']); env_center = np.asarray(meta['env_center'])
_res = lambda q: q if os.path.exists(q) else os.path.join('./datasets/3dshape', os.path.basename(q))
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    V_sh, F_sh, _ = load_obj(_res(meta['shape_obj'])); V_env, F_env, _ = load_obj(_res(meta['env_obj']))
shape_V = np.ascontiguousarray((V_sh - 0.5 * (V_sh.min(0) + V_sh.max(0))) / env_scale)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())
V_env_n = (V_env - env_center) / env_scale
from dataprocessing.preprocess_obj import sample_surface_points
env_pts = np.ascontiguousarray(sample_surface_points(V_env_n, F_env, 50000))


def clearance(c):
    t = c[:3]; m = np.linalg.norm(env_pts - t, axis=1) <= shape_radius + 0.06
    if not m.any():
        return 0.06
    R = _rotvec_to_matrix_np(c[3:6] * 2 * np.pi)
    S = igl.signed_distance(np.ascontiguousarray((env_pts[m] - t) @ R), shape_V, shape_F, _SDF_SIGN)[0]
    return float(S.min())


sel = np.nonzero((clear > 0) & (clear < 0.01))[0]
sel = rs.choice(sel, min(a.normal_check, len(sel)), replace=False)
h = 1e-3
cos = {m: [] for m in results}
for i in sel:
    c = cfg[i].astype(np.float64)
    n = np.array([(clearance(c + h * e) - clearance(c - h * e)) / (2 * h) for e in np.eye(6)[:3]])
    if np.linalg.norm(n) < 1e-6:
        continue
    n /= np.linalg.norm(n)
    for m, (f, G) in results.items():
        g = G[i, :3]
        if np.linalg.norm(g) > 1e-9:
            cos[m].append(np.dot(g / np.linalg.norm(g), n))
print(f'\n===== direction check on {len(sel)} waypoints with 0 < clearance < 0.01 =====')
print('cosine( translational speed-gradient g , numerical d(clearance)/dxyz ):  +1 = g points away from obstacle')
for m, v in cos.items():
    v = np.array(v)
    print(f'  {m:6s}: median {np.median(v):+.3f}   frac > 0: {np.mean(v > 0):.3f}   frac > 0.5: {np.mean(v > 0.5):.3f}')
np.savez(os.path.join(a.run, 'spread_features.npz'), clear=clear, sp_full=sp_full,
         **{f'{m}_{k}': v for m, (f, G) in results.items() for k, v in f.items()},
         **{f'{m}_g': G for m, (f, G) in results.items()})
