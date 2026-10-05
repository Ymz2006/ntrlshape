"""Architecture-vs-loss expressivity probe for |grad tau|.

Question this answers: the trained field cannot make |grad tau| reach the
required range at the EXTREMES (too small at tight clearance, too large at open
clearance -- see dtau_by_transformed_speed_grid.py).  Is that a limit of the
NETWORK FAMILY (architecture) or of the PLANNING LOSS / optimisation?

Method: take the SAME architecture and the SAME Fourier matrix B as the trained
model, then fit it with a PURE eikonal-residual objective on the query endpoint
x0 -- stripped of every planning-loss confound:

    min_theta  mean[ (|grad_xyz tau| * speed_dist  - 1)^2
                   + (|grad_rot tau| * speed_angle - 1)^2 ]

NO accept/reject, NO exp(-T) weighting, NO polynomial warps beyond the ones that
DEFINE the target speeds (transform_speeds, identical to training), NO speed<0.9
reweighting, NO 5-batch/epoch cap, NO in-loop LR override.  Full Adam to
convergence.  Then we re-render the signed mismatch grids (actual - predicted).

Interpretation:
  * ends collapse to ~0  -> capacity EXISTS  -> the problem is the LOSS/optim.
  * ends stay red/blue   -> |grad| saturates -> the problem is the ARCHITECTURE.

--init ckpt  : start from the trained weights (most generous to "capacity").
--init fresh : re-init the weights (same B) -- pure from-scratch expressivity.
--init both  : run both (default).

This script DOES NOT modify or save over any existing model; it trains a throw-
away copy of the network in memory only.

Run (from the nested ntrl-demo root, inside the pytorch docker):
    python expressivity_probe.py --dataPath testing_data/3dshape/Lshape3d_env1
"""

import sys
sys.path.append('.')

import os
import argparse

import numpy as np
import torch

from models.metric import model_train_metric as md
from hessian_by_dist_angle_grid import grid_stats, _print_grid
from dtau_by_transformed_speed_grid import (
    transform_speeds, compute_grad_mag, DIM, HALF)


# edges_with_unit_bin is defined inside main() of the other script, so redefine.
def edges_with_unit_bin(lo, nbins):
    lo = min(lo, 0.99)
    return np.append(np.linspace(lo, 0.99, nbins), 1.0)


def build_model(model_path, data_path, ckpt, device):
    """Build a Model and load the checkpoint (for its trained B + weights)."""
    model = md.Model(model_path, data_path, DIM, [0.0] * DIM, device=device)
    if ckpt is None:
        ckpt = os.path.join(model_path, 'latest.pt')
    print('loading checkpoint (for B + trained weights):', ckpt)
    model.load(ckpt)
    return model


def diff_grids(model, pairs, xd, xa, d_edges, a_edges, batch):
    """Return (dist_diff_median_grid, angle_diff_median_grid, stats dict)."""
    full, trans, rot = compute_grad_mag(model, pairs, batch=batch)
    dist_diff = 1.0 / xd - trans      # actual - predicted (translation)
    angle_diff = 1.0 / xa - rot       # actual - predicted (rotation)
    gd = grid_stats(xd, xa, dist_diff, d_edges, a_edges)
    ga = grid_stats(xd, xa, angle_diff, d_edges, a_edges)
    stats = dict(trans_med=np.median(trans), trans_max=trans.max(),
                 rot_med=np.median(rot), rot_max=rot.max(),
                 dist_diff_med=np.median(dist_diff),
                 angle_diff_med=np.median(angle_diff))
    return gd['median'], ga['median'], stats


def pure_fit(model, pairs, xd, xa, epochs, batch, lr, device, print_every):
    """Pure eikonal-residual fit of |grad tau| on x0.  Trains model.network."""
    net = model.network
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    P = torch.tensor(pairs, dtype=torch.float32, device=device)
    SD = torch.tensor(xd, dtype=torch.float32, device=device)
    SA = torch.tensor(xa, dtype=torch.float32, device=device)
    N = len(pairs)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        for s in range(0, N, batch):
            idx = perm[s:s + batch]
            xp = P[idx]
            tau, _w, coords = net.out(xp)
            g = torch.autograd.grad(tau.sum(), coords, create_graph=True)[0]
            trans = torch.linalg.norm(g[:, :HALF], dim=1)
            rot = torch.linalg.norm(g[:, HALF:DIM], dim=1)
            res = (trans * SD[idx] - 1.0) ** 2 + (rot * SA[idx] - 1.0) ** 2
            loss = res.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if epoch % print_every == 0 or epoch == epochs - 1:
            print('  epoch %4d   pure-eikonal loss = %.5f' % (epoch, tot / N))
    net.eval()


def render(tag, gd, ga, d_edges, a_edges, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('(matplotlib unavailable; skipped %s heatmap)' % tag)
        return
    dc = 0.5 * (d_edges[:-1] + d_edges[1:])
    ac = 0.5 * (a_edges[:-1] + a_edges[1:])
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for ax, g, t in [(axs[0], gd, 'DIST diff (1/speed_dist - |grad_xyz|)'),
                     (axs[1], ga, 'ANGLE diff (1/speed_angle - |grad_rot|)')]:
        finite = g[np.isfinite(g)]
        m = np.max(np.abs(finite)) if finite.size else 1.0
        im = ax.imshow(g, origin='lower', aspect='auto', cmap='RdBu_r',
                       vmin=-m, vmax=m,
                       extent=[d_edges[0], d_edges[-1], a_edges[0], a_edges[-1]])
        for i, yy in enumerate(ac):
            for j, xx in enumerate(dc):
                v = g[i, j]
                if np.isfinite(v):
                    ax.text(xx, yy, '%.2f' % v, ha='center', va='center',
                            color='k', fontsize=7)
        ax.set_title(t)
        ax.set_xlabel('speed_dist (tight->open)')
        ax.set_ylabel('speed_angle (tight->open)')
        fig.colorbar(im, ax=ax, fraction=0.046, label='actual - predicted')
    plt.suptitle('expressivity probe [%s]  red=+ grad too small, blue=- grad too large'
                 % tag, fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'probe_diff_%s.png' % tag)
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print('saved -> %s' % path)


def run_one(init, args, pairs, xd, xa, d_edges, a_edges):
    print('\n' + '=' * 92)
    print('INIT = %s' % init.upper())
    print('=' * 92)
    model = build_model(args.modelPath, args.dataPath, args.ckpt, args.device)
    if init == 'fresh':
        # reset all weights but keep the SAME Fourier matrix B
        model.network.apply(model.network.init_weights)
        print('re-initialised network weights (same B) -> from-scratch fit')

    gd0, ga0, s0 = diff_grids(model, pairs, xd, xa, d_edges, a_edges, args.batch)
    print('BEFORE fit:  |grad_xyz| med=%.3f max=%.3f   |grad_rot| med=%.3f max=%.3f'
          % (s0['trans_med'], s0['trans_max'], s0['rot_med'], s0['rot_max']))
    print('             dist_diff med=%.3f   angle_diff med=%.3f'
          % (s0['dist_diff_med'], s0['angle_diff_med']))

    print('--- pure eikonal-residual fit (%d epochs, batch %d, lr %g) ---'
          % (args.epochs, args.batch, args.lr))
    pure_fit(model, pairs, xd, xa, args.epochs, args.batch, args.lr,
             args.device, args.print_every)

    gd1, ga1, s1 = diff_grids(model, pairs, xd, xa, d_edges, a_edges, args.batch)
    print('\nAFTER fit:   |grad_xyz| med=%.3f max=%.3f   |grad_rot| med=%.3f max=%.3f'
          % (s1['trans_med'], s1['trans_max'], s1['rot_med'], s1['rot_max']))
    print('             dist_diff med=%.3f   angle_diff med=%.3f'
          % (s1['dist_diff_med'], s1['angle_diff_med']))
    print('\nMEDIAN DIST  diff AFTER fit (actual-pred) per cell:')
    _print_grid('', gd1, d_edges, a_edges)
    print('\nMEDIAN ANGLE diff AFTER fit (actual-pred) per cell:')
    _print_grid('', ga1, d_edges, a_edges)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        render(init, gd1, ga1, d_edges, a_edges, args.out)
    return s0, s1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./testing_data/3dshape/Lshape3d_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--ckpt', default=None, help='checkpoint .pt (default latest.pt)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--init', choices=['ckpt', 'fresh', 'both'], default='both')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--print_every', type=int, default=25)
    p.add_argument('--nbins_dist', type=int, default=8)
    p.add_argument('--nbins_angle', type=int, default=8)
    p.add_argument('--out', default='./results/expressivity_probe')
    args = p.parse_args()

    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    sd = np.load(os.path.join(args.dataPath, 'speed_dists.npy')).astype(np.float64)
    sa = np.load(os.path.join(args.dataPath, 'speed_angles.npy')).astype(np.float64)
    xd, xa = transform_speeds(sd[:, 0], sa[:, 0])
    print('input pairs: %d (from %s)' % (len(pairs), args.dataPath))
    print('target |grad_xyz| = 1/speed_dist  in [%.3f, %.3f]'
          % ((1.0 / xd).min(), (1.0 / xd).max()))
    print('target |grad_rot| = 1/speed_angle in [%.3f, %.3f]'
          % ((1.0 / xa).min(), (1.0 / xa).max()))

    d_edges = edges_with_unit_bin(xd.min(), args.nbins_dist)
    a_edges = edges_with_unit_bin(xa.min(), args.nbins_angle)

    modes = ['ckpt', 'fresh'] if args.init == 'both' else [args.init]
    results = {}
    for m in modes:
        results[m] = run_one(m, args, pairs, xd, xa, d_edges, a_edges)

    print('\n' + '=' * 92)
    print('SUMMARY  (if AFTER-fit |grad| reaches the target range and the diff '
          'medians collapse\n         toward 0 -> capacity EXISTS -> LOSS issue; '
          'if it stays stuck -> ARCH issue)')
    print('=' * 92)
    for m in modes:
        s0, s1 = results[m]
        print('[%5s] |grad_xyz| max %.2f->%.2f  |grad_rot| max %.2f->%.2f   '
              'dist_diff med %.3f->%.3f  angle_diff med %.3f->%.3f'
              % (m, s0['trans_max'], s1['trans_max'], s0['rot_max'], s1['rot_max'],
                 s0['dist_diff_med'], s1['dist_diff_med'],
                 s0['angle_diff_med'], s1['angle_diff_med']))


if __name__ == '__main__':
    main()
