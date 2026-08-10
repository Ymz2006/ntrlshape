"""|grad tau| binned over the TRANSFORMED speed_dist x speed_angle grid.

Unlike hessian_by_dist_angle_grid.py (which bins by the *raw* clearance proxies
speed_dist / speed_angle written by the preprocessor), this script first maps
those raw proxies through the EXACT transformations the training loop applies
before they enter the loss, then bins by those transformed model-input speeds.

The transforms mirror models/metric/model_train_metric.py (lines ~296-308):

    # speed_angle is stored as angle/pi; rescale to the pi/18 (10 deg) cut, clamp.
    angle_max = pi/18 ;  angle_min = 1e-4
    speed_angle = clamp(speed_angle / angle_max, angle_min/angle_max, 1)

    speed_dist  = speed_dist**2  * (2 - speed_dist)**2          # warp
    speed_angle = speed_angle**2 * (2 - speed_angle)**2         # warp
    speed_dist  = clamp(speed_dist,  min=1e-3)
    speed_angle = clamp(speed_angle, min=1e-3)

Each query config x0 (column 0 of speed_dists / speed_angles) is binned into a
2-D grid:  COLUMNS = transformed speed_dist,  ROWS = transformed speed_angle.

Output opens with two occupancy charts on that grid -- per-cell counts for the
START endpoint x0 and for the END endpoint x1 (column 1) -- so you can see the
sample support behind every statistic that follows.  Both use the same bin edges
as the rest of the script; those edges come from x0, so x1 values below the first
edge are clamped into it (reported when it happens).

Those two charts are then followed by their ONE-AXIS MARGINALS: the speed_dist
distribution with every speed_angle bin summed away, and the speed_angle
distribution with every speed_dist bin summed away.  Use these when only one axis
matters -- they read the spread along that axis alone, with no angle/dist
cross-term to disentangle.  They are collapses of the same grid, not a re-bin, so
edges and clamping match the 2-D charts exactly.

In each cell we report the mean magnitude of the field gradient |grad tau| of
the QUERY endpoint (x0) for three coordinate blocks -- three grids in total:

    FULL  : |grad_{x y z rx ry rz} tau|     (all 6 dims)
    TRANS : |grad_{x y z} tau|              (first 3, translational)
    ROT   : |grad_{rx ry rz} tau|           (last 3, rotational)

Model loading mirrors inspect_d2tau_range / evaluate_training_3d; defaults to
the latest checkpoint and testing_data/3dshape/Lshape3d_env1 (both flags).

Run (from the nested ntrl-demo root, inside the pytorch docker):
    python dtau_by_transformed_speed_grid.py
    python dtau_by_transformed_speed_grid.py --dataPath testing_data/3dshape/rectangle_env1 \
        --nbins_dist 8 --nbins_angle 8 --stat mean --out ./results/dtau_grid
"""

import sys
sys.path.append('.')

import os
import argparse
from glob import glob

import numpy as np
import torch

from models.metric import model_train_metric as md
from hessian_by_dist_angle_grid import grid_stats, _print_grid

DIM = 6
HALF = 3   # translation = first 3 query coords (xyz), rotation = last 3 (rotvec)


def load_model(model_path, data_path, ckpt=None, device='cuda'):
    """Mirror inspect_d2tau_range.load_model but allow an explicit --ckpt."""
    model = md.Model(model_path, data_path, DIM, [0.0] * DIM, device=device)
    pt = ckpt
    if pt is None:
        latest = os.path.join(model_path, 'latest.pt')
        if os.path.exists(latest):
            pt = latest
        else:
            ckpts = sorted(glob(os.path.join(model_path, '*', 'Model_Epoch_*.pt')))
            if not ckpts:
                raise FileNotFoundError('no latest.pt and no Model_Epoch_*.pt found')
            pt = ckpts[-1]
    print('checkpoint:', pt)

    #pt = './Experiments/3dshape/3dshape_06_22_11_16/latest.pt'
    #pt = './Experiments/3dshape/3dshape_06_26_16_46/Model_Epoch_06000_ValLoss_3.277650e-01.pt'
    #pt = './Experiments/3dshape/3dshape_07_01_12_45/latest.pt'
    #pt = './Experiments/3dshape/3dshape_07_17_15_47/Model_Epoch_03000_ValLoss_2.554522e-02.pt'
    #pt = './Experiments/3dshape/3dshape_07_25_12_40/latest.pt'
    pt = './Experiments/3dshape/3dshape_08_08_22_26/latest.pt'
    model.load(pt)
    model.network.eval()
    return model


def transform_speeds(sd_raw, sa_raw):
    """Map raw speed_dist / speed_angle proxies to the trained model inputs.

    Numpy mirror of models/metric/model_train_metric.py:296-308. Inputs and
    outputs are the x0 (query) column only.
    """
    angle_max = np.pi / 18.0
    angle_min = 0.0001

    sa = np.clip(sa_raw / angle_max, angle_min / angle_max, 1.0)

    sd = sd_raw ** 2 * (2 - sd_raw) ** 2
    sa = sa ** 2 * (2 - sa) ** 2

    sd = np.clip(sd, 1e-3, None)
    sa = np.clip(sa, 1e-3, None)
    return sd, sa


def eikonal_loss_x0(trans, rot, sd0, sa0):
    """Per-point eikonal residual loss for the QUERY endpoint x0.

    Numpy mirror of the x0 terms of Function.Loss (model_function_metric.py
    103-127):  per channel  (|grad| * speed - 1)**2, multiplied by the tight-cell
    weight clamp(1/speed, max=10) wherever speed < 0.9.
    trans / rot are |grad_{xyz} tau| / |grad_{rotvec} tau|; sd0 / sa0 are the
    TRANSFORMED speed_dist / speed_angle (same as the binning axes).

    Returns (dist_loss, angle_loss); the total is their sum.
    """
    Ld = (np.sqrt(trans ** 2 + 1e-8) * sd0 - 1.0) ** 2
    La = (np.sqrt(rot ** 2 + 1e-8) * sa0 - 1.0) ** 2
    # wd = np.clip(1.0 / sd0, None, 10.0)
    # wa = np.clip(1.0 / sa0, None, 10.0)
    # Ld = np.where(sd0 < 0.9, Ld * wd, Ld)
    # La = np.where(sa0 < 0.9, La * wa, La)
    return Ld, La


def compute_grad_mag(model, pairs, batch=2048):
    """pairs: (N, 2*DIM) numpy -> (full, trans, rot) magnitudes of grad tau (x0).

    full  = |grad_{all 6} tau| ,  trans = |grad_{xyz} tau| ,  rot = |grad_{rotvec} tau|
    Differentiates tau w.r.t. the fresh leaf `coords` returned by network.out,
    exactly like Function.Loss / inspect_d2tau_range.
    """
    full = np.empty(len(pairs), dtype=np.float64)
    trans = np.empty(len(pairs), dtype=np.float64)
    rot = np.empty(len(pairs), dtype=np.float64)
    for s in range(0, len(pairs), batch):
        xp = torch.tensor(pairs[s:s + batch], dtype=torch.float32,
                          device=model.Params['Device'])
        tau, _w, coords = model.network.out(xp)
        g = model.function.gradient(tau, coords)        # (B, 2*DIM)
        gq = g[:, :DIM]                                 # query endpoint x0 block
        full[s:s + batch] = torch.linalg.norm(gq, dim=1).detach().cpu().numpy()
        trans[s:s + batch] = torch.linalg.norm(gq[:, :HALF], dim=1).detach().cpu().numpy()
        rot[s:s + batch] = torch.linalg.norm(gq[:, HALF:DIM], dim=1).detach().cpu().numpy()
    return full, trans, rot


def _print_marginal(title, edges, start_vals, end_vals):
    """One-axis occupancy: per-bin counts for both endpoints on a single speed axis.

    The 2-D charts show where a pair sits in speed_dist AND speed_angle jointly.
    These collapse one axis away entirely, answering how the mass is spread along
    speed_dist irrespective of angle (and vice versa) -- the reading you want when
    only one of the two axes matters.
    """
    tot_s = max(int(start_vals.sum()), 1)
    tot_e = max(int(end_vals.sum()), 1)
    print('\n' + title)
    print('%10s %18s %10s %8s %10s %8s'
          % ('center', 'range', 'start', 'start%', 'end', 'end%'))
    for i in range(len(edges) - 1):
        print('%10.2f %18s %10d %7.1f%% %10d %7.1f%%'
              % (0.5 * (edges[i] + edges[i + 1]),
                 '[%.3f, %.3f]' % (edges[i], edges[i + 1]),
                 int(start_vals[i]), 100.0 * start_vals[i] / tot_s,
                 int(end_vals[i]), 100.0 * end_vals[i] / tot_e))
    print('%10s %18s %10d %7.1f%% %10d %7.1f%%'
          % ('total', '', tot_s, 100.0, tot_e, 100.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--ckpt', default=None,
                   help='explicit checkpoint .pt (default: <modelPath>/latest.pt)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--n', type=int, default=0,
                   help='number of input pairs to use (0 = all)')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--nbins_dist', type=int, default=8)
    p.add_argument('--nbins_angle', type=int, default=8)
    p.add_argument('--stat', choices=['mean', 'median', 'min', 'max'], default='mean',
                   help='per-cell reduction shown in the three |grad tau| grids')
    p.add_argument('--fixed01', action='store_true',
                   help='use fixed [0,1] bin edges instead of data min..max')
    p.add_argument('--out', default=None,
                   help='dir to save grid .npz (and heatmaps if plotly available)')
    args = p.parse_args()

    model = load_model(args.modelPath, args.dataPath, ckpt=args.ckpt, device=args.device)

    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    sd = np.load(os.path.join(args.dataPath, 'speed_dists.npy')).astype(np.float64)
    sa = np.load(os.path.join(args.dataPath, 'speed_angles.npy')).astype(np.float64)
    if args.n and args.n < len(pairs):
        pairs, sd, sa = pairs[:args.n], sd[:args.n], sa[:args.n]

    # raw proxies -> transformed model-input speeds (binning axes).  Column 0 is
    # the START endpoint x0 -- the query point every grid below is binned by --
    # and column 1 is the END endpoint x1, used only for its own occupancy chart.
    xd, xa = transform_speeds(sd[:, 0], sa[:, 0])
    xd_end, xa_end = transform_speeds(sd[:, 1], sa[:, 1])
    print('input pairs: %d  (from %s)' % (len(pairs), args.dataPath))
    print('transformed speed_dist  x0: min=%.4g max=%.4g   '
          'transformed speed_angle x0: min=%.4g max=%.4g'
          % (xd.min(), xd.max(), xa.min(), xa.max()))
    print('transformed speed_dist  x1: min=%.4g max=%.4g   '
          'transformed speed_angle x1: min=%.4g max=%.4g'
          % (xd_end.min(), xd_end.max(), xa_end.min(), xa_end.max()))

    # Bin edges, shared by every grid in this script (both endpoint distributions
    # included, so the two charts are directly comparable to each other and to the
    # COUNT grid further down).  Reserve the last bin as a dedicated "speed = 1"
    # cell spanning [0.99, 1.0]; the remaining nbins-1 bins evenly tile [lo, 0.99].
    def edges_with_unit_bin(lo, nbins):
        lo = min(lo, 0.99)
        return np.append(np.linspace(lo, 0.99, nbins), 1.0)

    if args.fixed01:
        d_edges = edges_with_unit_bin(0.0, args.nbins_dist)
        a_edges = edges_with_unit_bin(0.0, args.nbins_angle)
    else:
        d_edges = edges_with_unit_bin(xd.min(), args.nbins_dist)
        a_edges = edges_with_unit_bin(xa.min(), args.nbins_angle)

    # ── Endpoint occupancy: where the two endpoints of each pair actually live on
    # this grid.  Printed up front because it is the sample support for every
    # statistic below -- a cell with a handful of points is noise, not signal.
    # grid_stats only needs the (xd, xa) coordinates for a count, so the value
    # array it reduces is irrelevant here.
    start_count = grid_stats(xd, xa, xd, d_edges, a_edges)['count']
    end_count = grid_stats(xd_end, xa_end, xd_end, d_edges, a_edges)['count']

    print('\n' + '=' * 92)
    print('ENDPOINT DISTRIBUTION over the TRANSFORMED speed_dist x speed_angle grid')
    print('cols = transformed speed_dist (0=tight ->1=open), '
          'rows = transformed speed_angle')
    print('=' * 92)
    _print_grid('COUNT per cell -- START endpoint x0 (the binning axis):',
                start_count, d_edges, a_edges, int_fmt=True)
    _print_grid('COUNT per cell -- END endpoint x1:',
                end_count, d_edges, a_edges, int_fmt=True)
    # The edges are derived from x0, so x1 can fall below the first edge; grid_stats
    # clamps those into the edge bin rather than dropping them.  Say so, otherwise
    # the leftmost / bottom cells of the x1 chart read as genuine occupancy.
    n_lo_d = int((xd_end < d_edges[0]).sum())
    n_lo_a = int((xa_end < a_edges[0]).sum())
    if n_lo_d or n_lo_a:
        print('  note: x1 below the x0-derived grid range and clamped into the '
              'edge bin -- speed_dist %d, speed_angle %d (of %d)'
              % (n_lo_d, n_lo_a, len(xd_end)))

    # ── One-axis marginals: the same occupancy with the other axis summed away ──
    # Collapsing the grid rather than re-binning keeps these exactly consistent
    # with the 2-D charts (identical edges, identical clamping).
    _print_marginal(
        'DIST marginal -- transformed speed_dist, all speed_angle bins summed:',
        d_edges, start_count.sum(axis=0), end_count.sum(axis=0))
    _print_marginal(
        'ANGLE marginal -- transformed speed_angle, all speed_dist bins summed:',
        a_edges, start_count.sum(axis=1), end_count.sum(axis=1))

    full, trans, rot = compute_grad_mag(model, pairs, batch=args.batch)
    print('|grad tau| (full)  median=%.4g max=%.4g' % (np.median(full), full.max()))
    print('|grad tau| (trans) median=%.4g max=%.4g' % (np.median(trans), trans.max()))
    print('|grad tau| (rot)   median=%.4g max=%.4g\n' % (np.median(rot), rot.max()))

    dist_loss, angle_loss = eikonal_loss_x0(trans, rot, xd, xa)
    loss = dist_loss + angle_loss
    print('eikonal loss x0   median=%.4g max=%.4g' % (np.median(loss), loss.max()))
    print('  dist  loss x0   median=%.4g max=%.4g' % (np.median(dist_loss), dist_loss.max()))
    print('  angle loss x0   median=%.4g max=%.4g\n' % (np.median(angle_loss), angle_loss.max()))

    # Signed Eikonal mismatch (actual - predicted) of the gradient magnitudes.
    # The Eikonal target is |grad tau| * speed == 1, so the ACTUAL (target)
    # gradient magnitude is 1/speed and the PREDICTED one is what the model
    # gives (trans / rot).  diff = actual - predicted = 1/speed - |grad tau|.
    #   diff > 0  ->  model gradient too SMALL  (field too fast / under-slowed)
    #   diff < 0  ->  model gradient too LARGE  (field too slow / over-slowed)
    dist_diff = 1.0 / xd - trans          # actual - predicted, translation
    angle_diff = 1.0 / xa - rot           # actual - predicted, rotation
    print('dist  diff x0 (1/speed_dist  - |grad_xyz|)   '
          'median=%.4g mean=%.4g' % (np.median(dist_diff), dist_diff.mean()))
    print('angle diff x0 (1/speed_angle - |grad_rot|)   '
          'median=%.4g mean=%.4g\n' % (np.median(angle_diff), angle_diff.mean()))

    blocks = {'full': full, 'trans': trans, 'rot': rot,
              'loss': loss, 'dist_loss': dist_loss, 'angle_loss': angle_loss,
              'dist_diff': dist_diff, 'angle_diff': angle_diff}
    grids = {name: grid_stats(xd, xa, val, d_edges, a_edges)
             for name, val in blocks.items()}

    print('=' * 92)
    print('|grad tau| over the TRANSFORMED speed_dist x speed_angle grid')
    print('cols = transformed speed_dist (0=tight ->1=open), '
          'rows = transformed speed_angle')
    print('=' * 92)
    # one shared count grid (same binning for all three; identical to the START
    # chart above, repeated here so this section stands on its own)
    _print_grid('COUNT per cell:', grids['full']['count'], d_edges, a_edges, int_fmt=True)
    titles = {'full': 'FULL  |grad_{xyzrxryrz} tau|',
              'trans': 'TRANS |grad_{xyz} tau|',
              'rot': 'ROT   |grad_{rxryrz} tau|'}
    for name in ('full', 'trans', 'rot'):
        _print_grid('%s %s per cell:' % (args.stat.upper(), titles[name]),
                    grids[name][args.stat], d_edges, a_edges)
    # eikonal loss is reported as a median per cell regardless of --stat
    _print_grid('MEDIAN eikonal LOSS (x0, dist+angle) per cell:',
                grids['loss']['median'], d_edges, a_edges)
    _print_grid('MEDIAN DIST  loss (x0) per cell:',
                grids['dist_loss']['median'], d_edges, a_edges)
    _print_grid('MEDIAN ANGLE loss (x0) per cell:',
                grids['angle_loss']['median'], d_edges, a_edges)
    # signed mismatch (actual - predicted); + = grad too small, - = grad too large
    _print_grid('MEDIAN DIST  diff (actual-pred = 1/speed_dist - |grad_xyz|) per cell:',
                grids['dist_diff']['median'], d_edges, a_edges)
    _print_grid('MEDIAN ANGLE diff (actual-pred = 1/speed_angle - |grad_rot|) per cell:',
                grids['angle_diff']['median'], d_edges, a_edges)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        npz = os.path.join(args.out, 'dtau_transformed_grid.npz')
        save = dict(d_edges=d_edges, a_edges=a_edges,
                    start_count=start_count, end_count=end_count,
                    start_count_dist=start_count.sum(axis=0),
                    end_count_dist=end_count.sum(axis=0),
                    start_count_angle=start_count.sum(axis=1),
                    end_count_angle=end_count.sum(axis=1))
        for name in blocks:
            for stat, arr in grids[name].items():
                save['%s_%s' % (name, stat)] = arr
        np.savez(npz, **save)
        print('\nsaved grid arrays -> %s' % npz)
        try:
            import plotly.graph_objects as go
            d_cen = 0.5 * (d_edges[:-1] + d_edges[1:])
            a_cen = 0.5 * (a_edges[:-1] + a_edges[1:])
            for tag, cnt, lbl in (('start', start_count, 'START endpoint x0'),
                                  ('end', end_count, 'END endpoint x1')):
                fig = go.Figure(go.Heatmap(
                    z=cnt, x=d_cen, y=a_cen,
                    colorscale='Viridis', colorbar=dict(title='count')))
                fig.update_layout(
                    title='endpoint distribution -- %s' % lbl,
                    xaxis_title='transformed speed_dist (0=tight -> 1=open)',
                    yaxis_title='transformed speed_angle (0=tight -> 1=open)')
                fig.write_html(os.path.join(args.out,
                                            'dtau_grid_count_%s.html' % tag))
            print('saved endpoint distributions -> %s/dtau_grid_count_{start,end}.html'
                  % args.out)
            # One-axis marginals as grouped bars, start vs end side by side.
            for tag, cen, axis, lbl in (
                    ('dist', d_cen, 0, 'transformed speed_dist'),
                    ('angle', a_cen, 1, 'transformed speed_angle')):
                fig = go.Figure([
                    go.Bar(x=cen, y=start_count.sum(axis=axis), name='start x0'),
                    go.Bar(x=cen, y=end_count.sum(axis=axis), name='end x1')])
                fig.update_layout(
                    barmode='group', title='%s marginal (other axis summed)' % tag,
                    xaxis_title='%s (0=tight -> 1=open)' % lbl,
                    yaxis_title='count')
                fig.write_html(os.path.join(args.out,
                                            'dtau_marginal_%s.html' % tag))
            print('saved marginals -> %s/dtau_marginal_{dist,angle}.html' % args.out)
            for name in ('full', 'trans', 'rot'):
                fig = go.Figure(go.Heatmap(
                    z=grids[name][args.stat], x=d_cen, y=a_cen,
                    colorscale='Viridis', colorbar=dict(title='|grad tau|')))
                fig.update_layout(
                    title='%s |grad tau| (%s block)' % (args.stat, name),
                    xaxis_title='transformed speed_dist (0=tight -> 1=open)',
                    yaxis_title='transformed speed_angle (0=tight -> 1=open)')
                fig.write_html(os.path.join(args.out, 'dtau_grid_%s_%s.html'
                                            % (name, args.stat)))
            print('saved heatmaps -> %s/dtau_grid_*_%s.html' % (args.out, args.stat))
        except ImportError:
            print('(plotly not available; skipped heatmaps)')


if __name__ == '__main__':
    main()
