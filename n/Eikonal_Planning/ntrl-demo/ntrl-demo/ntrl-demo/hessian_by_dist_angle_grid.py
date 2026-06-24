"""Hessian magnitude as a function of distance/angle clearance to the obstacle.

For every sampled QUERY config x0 we already store two clearance proxies (written
by dataprocessing/preprocess_obj.main):

    speed_dist  = clip(dist  / margin, offset/margin, 1)   # translation clearance
    speed_angle = angle / pi                               # rotational clearance

(0 = touching / tightest, 1 = most open).  This script bins those two proxies for
x0 into a 2-D grid and, in each cell, reports the distribution of the per-point
Hessian spectral norm  max|eig(d2 tau / d q_i d q_j)|  of the trained field --
the cell-wise mean / median / min / max.  That tells you where in clearance space
the field has to bend the hardest.

Hessian + spectral-norm + model-loading helpers are reused from
inspect_d2tau_range; the speed_dist / speed_angle convention mirrors
dataprocessing/preprocess_obj.

Run (from the nested ntrl-demo root, inside the pytorch docker):
    python hessian_by_dist_angle_grid.py --dataPath testing_data/3dshape/Lshape3d_env1
    python hessian_by_dist_angle_grid.py --dataPath datasets/3dshape/Lshape3d_env1 \
        --n 100000 --nbins_dist 8 --nbins_angle 8 --block full --out ./results/hessian_grid
"""

import sys
sys.path.append('.')

import os
import argparse

import numpy as np

from inspect_d2tau_range import (
    load_model, compute_hessian, _sym_eig_absmax, DIM, HALF,
)

BLOCKS = {
    'full':  (0, DIM),    # 6x6: full config curvature
    'trans': (0, HALF),   # 3x3: translation block (pairs with speed_dist)
    'rot':   (HALF, DIM), # 3x3: rotation block    (pairs with speed_angle)
}


def block_spectral_norm(H, block):
    lo, hi = BLOCKS[block]
    return _sym_eig_absmax(H[:, lo:hi, lo:hi])


def grid_stats(xd, xa, val, d_edges, a_edges):
    """Bin (xd, xa) into the 2-D grid and reduce `val` per cell.

    Returns dict of (ny_a, nx_d) arrays: count, mean, median, min, max.
    Rows = speed_angle bins (ascending), cols = speed_dist bins (ascending).
    Empty cells are NaN (count 0).
    """
    nd = len(d_edges) - 1
    na = len(a_edges) - 1
    # np.digitize -> 1..n inside range; clamp so the right edge lands in the last bin.
    di = np.clip(np.digitize(xd, d_edges) - 1, 0, nd - 1)
    ai = np.clip(np.digitize(xa, a_edges) - 1, 0, na - 1)

    count = np.zeros((na, nd), dtype=np.int64)
    mean = np.full((na, nd), np.nan)
    median = np.full((na, nd), np.nan)
    vmin = np.full((na, nd), np.nan)
    vmax = np.full((na, nd), np.nan)
    for a in range(na):
        for d in range(nd):
            m = (ai == a) & (di == d)
            c = int(m.sum())
            count[a, d] = c
            if c:
                v = val[m]
                mean[a, d] = v.mean()
                median[a, d] = np.median(v)
                vmin[a, d] = v.min()
                vmax[a, d] = v.max()
    return dict(count=count, mean=mean, median=median, min=vmin, max=vmax)


def _print_grid(title, arr, d_edges, a_edges, fmt='%11.4g', int_fmt=False):
    nd = len(d_edges) - 1
    print('\n' + title)
    # column header: speed_dist bin centers / ranges
    hdr = ' angle\\dist '
    for d in range(nd):
        hdr += ('%11s' % ('%.2f' % (0.5 * (d_edges[d] + d_edges[d + 1]))))
    print(hdr)
    # rows: speed_angle bins, top = highest angle so the grid reads like a plot
    for a in range(len(a_edges) - 2, -1, -1):
        center = 0.5 * (a_edges[a] + a_edges[a + 1])
        row = '%10s ' % ('%.2f' % center)
        for d in range(nd):
            x = arr[a, d]
            if int_fmt:
                row += '%11d' % x
            elif np.isnan(x):
                row += '%11s' % '.'
            else:
                row += fmt % x
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./testing_data/3dshape/Lshape3d_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--n', type=int, default=0,
                   help='number of input pairs to use (0 = all)')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--transform', choices=['sq', 'sqrt'], default='sq',
                   help="'sq' = trained field tau (square baked in); "
                        "'sqrt' = sqrt(tau), curvature of the bare metric d.")
    p.add_argument('--block', choices=list(BLOCKS), default='full',
                   help="which Hessian block's spectral norm to bin "
                        "(full 6x6, trans=xyz 3x3, rot=rotvec 3x3)")
    p.add_argument('--nbins_dist', type=int, default=8)
    p.add_argument('--nbins_angle', type=int, default=8)
    p.add_argument('--fixed01', action='store_true',
                   help='use fixed [0,1] bin edges instead of data min..max')
    p.add_argument('--out', default=None,
                   help='dir to save grid .npz (and per-stat heatmaps if plotly)')
    args = p.parse_args()

    model = load_model(args.modelPath, args.dataPath)

    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    sd = np.load(os.path.join(args.dataPath, 'speed_dists.npy')).astype(np.float64)
    sa = np.load(os.path.join(args.dataPath, 'speed_angles.npy')).astype(np.float64)
    if args.n and args.n < len(pairs):
        pairs, sd, sa = pairs[:args.n], sd[:args.n], sa[:args.n]
    xd = sd[:, 0]   # x0 distance clearance proxy
    xa = sa[:, 0]   # x0 angular clearance proxy
    print('input pairs: %d  (from %s)' % (len(pairs), args.dataPath))
    print('speed_dist  x0: min=%.4g max=%.4g   speed_angle x0: min=%.4g max=%.4g'
          % (xd.min(), xd.max(), xa.min(), xa.max()))

    H = compute_hessian(model, pairs, batch=args.batch, transform=args.transform)
    val = block_spectral_norm(H, args.block)   # (N,) spectral norm of chosen block
    print('Hessian block=%s  transform=%s   |H| spectral norm: median=%.4g max=%.4g\n'
          % (args.block, args.transform, np.median(val), val.max()))

    if args.fixed01:
        d_edges = np.linspace(0.0, 1.0, args.nbins_dist + 1)
        a_edges = np.linspace(0.0, 1.0, args.nbins_angle + 1)
    else:
        d_edges = np.linspace(xd.min(), xd.max(), args.nbins_dist + 1)
        a_edges = np.linspace(xa.min(), xa.max(), args.nbins_angle + 1)

    g = grid_stats(xd, xa, val, d_edges, a_edges)

    print('=' * 92)
    print('Hessian spectral norm (block=%s) over the speed_dist x speed_angle grid'
          % args.block)
    print('cols = speed_dist bin center (0=tight ->1=open), rows = speed_angle bin center')
    print('=' * 92)
    _print_grid('COUNT per cell:', g['count'], d_edges, a_edges, int_fmt=True)
    _print_grid('MEAN   |H| per cell:', g['mean'], d_edges, a_edges)
    _print_grid('MEDIAN |H| per cell:', g['median'], d_edges, a_edges)
    _print_grid('MIN    |H| per cell:', g['min'], d_edges, a_edges)
    _print_grid('MAX    |H| per cell:', g['max'], d_edges, a_edges)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        npz = os.path.join(args.out, 'hessian_grid_%s.npz' % args.block)
        np.savez(npz, d_edges=d_edges, a_edges=a_edges, block=args.block, **g)
        print('\nsaved grid arrays -> %s' % npz)
        try:
            import plotly.graph_objects as go
            d_cen = 0.5 * (d_edges[:-1] + d_edges[1:])
            a_cen = 0.5 * (a_edges[:-1] + a_edges[1:])
            for stat in ('mean', 'median', 'min', 'max'):
                fig = go.Figure(go.Heatmap(
                    z=g[stat], x=d_cen, y=a_cen, colorscale='Viridis',
                    colorbar=dict(title='|H|')))
                fig.update_layout(
                    title='Hessian %s spectral norm (block=%s)' % (stat, args.block),
                    xaxis_title='speed_dist (0=tight -> 1=open)',
                    yaxis_title='speed_angle (0=tight -> 1=open)')
                html = os.path.join(args.out, 'hessian_grid_%s_%s.html' % (args.block, stat))
                fig.write_html(html)
            print('saved heatmaps -> %s/hessian_grid_%s_*.html' % (args.out, args.block))
        except ImportError:
            print('(plotly not available; skipped heatmaps)')


if __name__ == '__main__':
    main()
