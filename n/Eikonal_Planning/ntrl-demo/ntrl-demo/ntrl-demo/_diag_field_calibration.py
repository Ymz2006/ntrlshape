"""How well does a checkpoint's speed field track the clearance labels it was trained on?

For rows of the TRAINING set, compare the network's 1/|grad_xyz tau| at x0 (the
translational speed the Eikonal loss constrains) against the warped label
speed_dist**2 * (2 - speed_dist)**2 the trainer fits, binned by the raw label.
Also evaluates the same x0 paired with an INDEPENDENT far x1 (random other row's
x1) so near-pair vs far-pair behaviour can be told apart.

    python _diag_field_calibration.py --dataPath datasets/3dshape/Ashape3d_env2 \
        --checkpoint ./Experiments/3dshape/3dshape_09_04_09_09/latest.pt --device cuda:0
"""
import sys
sys.path.append('.')
import os, argparse
import numpy as np
import torch
from models.metric import model_train_metric as md

DIM = 6
parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--modelPath', default='./Experiments/3dshape')
parser.add_argument('--device', default='cuda')
parser.add_argument('--n', type=int, default=40000)
args = parser.parse_args()

ck = torch.load(args.checkpoint, map_location='cpu')
B = ck.get('B_state_dict', None)
if B is not None:
    bs = B if torch.is_tensor(B) else list(B.values())[0]
    print(f'B std = {bs.float().std():.3f}')

womodel = md.Model(args.modelPath, args.dataPath, DIM, [0.0] * DIM, device=args.device)
womodel.load(args.checkpoint)
womodel.network.eval()

X = np.load(os.path.join(args.dataPath, 'sampled_points.npy'), mmap_mode='r')
SD = np.load(os.path.join(args.dataPath, 'speed_dists.npy'))
SA = np.load(os.path.join(args.dataPath, 'speed_angles.npy'))
rs = np.random.RandomState(0)
idx = np.sort(rs.choice(len(X), args.n, replace=False))
X = np.asarray(X[idx], dtype=np.float32)
sd0 = SD[idx, 0]
sa0 = SA[idx, 0]
warp = lambda s: np.clip(s ** 2 * (2 - s) ** 2, 0.001, None)
tgt_t = warp(sd0)

perm = rs.permutation(args.n)
X_far = np.concatenate([X[:, :DIM], X[perm, DIM:]], 1)


def speeds(Xn):
    Xt = torch.tensor(Xn, device=args.device)
    out_t, out_a, out_f, taus = [], [], [], []
    for s in range(0, len(Xt), 10000):
        tau, w, coords = womodel.network.out(Xt[s:s + 10000])
        g = womodel.function.gradient(tau, coords, create_graph=False)[:, :DIM]
        out_t.append((1 / torch.norm(g[:, :3], dim=1)).detach().cpu().numpy())
        out_a.append((1 / torch.norm(g[:, 3:], dim=1)).detach().cpu().numpy())
        out_f.append((1 / torch.norm(g, dim=1)).detach().cpu().numpy())
        taus.append(tau[:, 0].detach().cpu().numpy())
    return (np.concatenate(out_t), np.concatenate(out_a), np.concatenate(out_f),
            np.concatenate(taus))


def report(name, Xn):
    st, sa, sf, tau = speeds(Xn)
    D = np.linalg.norm(Xn[:, DIM:DIM + 3] - Xn[:, :3], axis=1)
    print(f'\n=== {name}: |dtrans| median {np.median(D):.3f} ===')
    print('  raw speed_dist bin   n      target(warped)  pred_trans(med)  p10-p90          resid |g|*tgt-1 (med abs)')
    edges = [0.0, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sd0 >= lo) & (sd0 < hi)
        if m.sum() < 20:
            continue
        resid = np.abs(tgt_t[m] / st[m] - 1)
        print(f'  [{lo:.2f},{hi:.2f})      {m.sum():6d}   {np.median(tgt_t[m]):.4f}          {np.median(st[m]):.3f}            {np.percentile(st[m],10):.3f}-{np.percentile(st[m],90):.3f}      {np.median(resid):.2f}')
    from scipy.stats import spearmanr
    print(f'  spearman(pred_trans, target)      = {spearmanr(st, tgt_t)[0]:.3f}')
    print(f'  contrast: median pred_trans at sd<0.1 / at sd>0.8 = {np.median(st[sd0 < 0.1]) / np.median(st[sd0 > 0.8]):.3f}   (target ratio {np.median(tgt_t[sd0 < 0.1]) / np.median(tgt_t[sd0 > 0.8]):.4f})')
    print(f'  overall pred_trans p5/p50/p95 = {np.percentile(st,5):.3f}/{np.percentile(st,50):.3f}/{np.percentile(st,95):.3f};  pred_full p50 = {np.median(sf):.3f};  pred_ang p50 = {np.median(sa):.3f}')


report('TRAINING PAIRS (correlated x1)', X)
report('FAR PAIRS (x0 with an independent x1)', X_far)
