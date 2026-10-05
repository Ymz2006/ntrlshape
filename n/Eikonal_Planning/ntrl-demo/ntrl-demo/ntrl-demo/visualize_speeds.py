"""Visualize the clearance speeds stored in a 2-D shape training-data dir.

Reads the npy bundle that ``dataprocessing/preprocess_dxf.py`` writes:

    sampled_points.npy  (N, 6)  [x0, y0, theta0, x1, y1, theta1]
    speed.npy           (N, 2)  [speed0, speed1]
    env.npy             (M, 2)  environment boundary points
    meta.json           (optional, used to recover shape_scale)

Mirrors ``2dshape_baseline/visualize_speeds.py`` but always writes PNGs to
disk -- no GUI is touched.

Modes (--mode)
--------------
    map      [default] continuous heatmap interpolated from (x, y, speed)
    scatter  scatter of (x, y) points coloured by speed
    shapes   F-shape outlines drawn at each placement, coloured by speed
    hist     histogram of speed values
    theta    scatter of (x, y) coloured by theta

Usage
-----
    python visualize_speeds.py --data datasets/2dshape/Fshape_FmazeEasy
    python visualize_speeds.py --data datasets/2dshape/Fshape_FmazeEasy --mode shapes
    python visualize_speeds.py --data datasets/2dshape/Fshape_FmazeEasy --mode hist
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from shapely.geometry import Polygon

sys.path.append('.')
from dataprocessing.parse_shape import dxf_to_shape, shape_to_points


DEFAULT_DATA  = './datasets/2dshape/Fmaze2_norm'
DEFAULT_SHAPE = './datasets/2dshape/Fshape_norm.dxf'
DEFAULT_CMAP  = 'viridis'


# ── helpers ──────────────────────────────────────────────────────────────────
def _load_env(data_dir):
    env_path = os.path.join(data_dir, 'env.npy')
    if os.path.exists(env_path):
        env = np.load(env_path)
        if env.ndim == 1:
            env = env.reshape(-1, 2)
        return env
    return np.zeros((0, 2))


def _load_shape_scale(data_dir, cli_value):
    if cli_value is not None:
        return float(cli_value)
    meta_path = os.path.join(data_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return float(json.load(f).get('shape_scale', 1.0))
    return 1.0


def _rotate_pts(shape_points, x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    pts = np.array(shape_points, dtype=np.float32)
    rot = np.array([[c, -s], [s, c]])
    return [tuple(p) for p in (rot @ pts.T).T + np.array([x, y])]


# ── plot modes ───────────────────────────────────────────────────────────────
def plot_scatter(ax, x, y, speed, env_pts, cmap, vmin, vmax, title):
    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c='black', s=3, zorder=3, rasterized=True)
    sc = ax.scatter(x, y, c=speed, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=8, alpha=0.6, zorder=4, rasterized=True)
    plt.colorbar(sc, ax=ax, label='speed')
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)


def plot_shapes(ax, xy, theta, speed, shape_points, env_pts,
                cmap, vmin, vmax, title, max_cnt):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_ = get_cmap(cmap)

    n = min(max_cnt, len(xy))
    idx = (np.random.choice(len(xy), n, replace=False)
           if len(xy) > n else np.arange(len(xy)))

    for i in idx:
        rotated = _rotate_pts(shape_points,
                              float(xy[i, 0]), float(xy[i, 1]),
                              float(theta[i]))
        if len(rotated) < 3:
            continue
        poly = Polygon(rotated)
        if not poly.is_valid:
            continue
        ax.add_patch(plt.Polygon(
            list(poly.exterior.coords),
            facecolor=cmap_(norm(float(speed[i]))),
            edgecolor='black', linewidth=0.3, alpha=0.75, zorder=4))

    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c='grey', s=4, zorder=3, rasterized=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='speed')
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)


def plot_hist(axes, s0, s1, index):
    if index in ('0', 'both'):
        axes[0].hist(s0, bins=60, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('speed'); axes[0].set_ylabel('count')
        axes[0].set_title('Speed distribution -- start configs')
    if index in ('1', 'both'):
        axes[-1].hist(s1, bins=60, color='salmon', alpha=0.8)
        axes[-1].set_xlabel('speed'); axes[-1].set_ylabel('count')
        axes[-1].set_title('Speed distribution -- end configs')


def plot_theta(ax, x, y, theta_rad, env_pts, cmap, title):
    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c='black', s=3, zorder=3, rasterized=True)
    sc = ax.scatter(x, y, c=theta_rad, cmap=cmap, s=8, alpha=0.6,
                    zorder=4, rasterized=True)
    plt.colorbar(sc, ax=ax, label='theta (rad)')
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)


def plot_map(ax, x, y, speed, env_pts, cmap, vmin, vmax, title,
             grid_n=200, iso_levels=12):
    if env_pts.shape[0] > 0:
        xmin, xmax = env_pts[:, 0].min(), env_pts[:, 0].max()
        ymin, ymax = env_pts[:, 1].min(), env_pts[:, 1].max()
    else:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([x, y])

    ZZ = griddata(pts, speed, (XX, YY), method='linear')
    ZZ_nn = griddata(pts, speed, (XX, YY), method='nearest')
    nan_mask = np.isnan(ZZ)
    ZZ[nan_mask] = ZZ_nn[nan_mask]

    im = ax.imshow(ZZ, origin='lower', extent=[xmin, xmax, ymin, ymax],
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', alpha=0.85)
    cs = ax.contour(xs, ys, ZZ, levels=iso_levels,
                    colors='white', linewidths=0.7, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.3f')
    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c='black', s=4, zorder=3, rasterized=True)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('speed')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA,
                        help='Directory with sampled_points.npy, speed.npy, env.npy')
    parser.add_argument('--shape_dxf', default=DEFAULT_SHAPE,
                        help='F-shape DXF (used by --mode shapes)')
    parser.add_argument('--shape_scale', type=float, default=None,
                        help='F-shape scale; defaults to meta.json then 1.0')
    parser.add_argument('--mode', default='scatter',
                        choices=['map', 'scatter', 'shapes', 'hist', 'theta'])
    parser.add_argument('--index', default='both', choices=['0', '1', 'both'],
                        help='Which endpoint(s) to plot: 0=start, 1=end, both=side-by-side')
    parser.add_argument('--cnt', type=int, default=3000,
                        help='Max shapes/points to render in scatter/shapes/theta')
    parser.add_argument('--cmap', default=DEFAULT_CMAP)
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--grid_n', type=int, default=200,
                        help='Resolution of the interpolated map')
    parser.add_argument('--iso', type=int, default=12,
                        help='# isochrone contour levels in map mode')
    parser.add_argument('--raw_theta', action='store_true',
                        help='Theta in sampled_points is raw radians (not /2pi)')
    parser.add_argument('--out_dir', default='.',
                        help='Directory the PNG is written to')
    parser.add_argument('--out', default=None,
                        help='Output filename (default: speeds_<mode>_<index>.png)')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pts_path = os.path.join(args.data, 'sampled_points.npy')
    spd_path = os.path.join(args.data, 'speed.npy')
    if not (os.path.exists(pts_path) and os.path.exists(spd_path)):
        print(f'ERROR: missing sampled_points.npy / speed.npy in {args.data}')
        sys.exit(1)

    pts   = np.load(pts_path)         # (N, 6)
    speed = np.load(spd_path)         # (N, 2)
    env   = _load_env(args.data)
    shape_scale = _load_shape_scale(args.data, args.shape_scale)

    N = len(pts)
    print(f'Loaded {N} samples from {args.data}')
    print(f'  points shape: {pts.shape}   speed shape: {speed.shape}')
    print(f'  speed range:  [{speed.min():.4f}, {speed.max():.4f}]')
    print(f'  shape_scale:  {shape_scale}')

    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = pts[:, 3], pts[:, 4]
    scale = 1.0 if args.raw_theta else (2 * np.pi)
    t0, t1 = pts[:, 2] * scale, pts[:, 5] * scale
    s0, s1 = speed[:, 0], speed[:, 1]

    vmin = args.vmin if args.vmin is not None else float(speed.min())
    vmax = args.vmax if args.vmax is not None else float(speed.max())

    sel = (np.random.choice(N, args.cnt, replace=False)
           if N > args.cnt and args.mode in ('scatter', 'shapes', 'theta')
           else np.arange(N))

    ncols = 2 if args.index == 'both' else 1
    figw  = 9 * ncols if args.mode != 'hist' else 7 * ncols

    if args.mode == 'hist':
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 5), squeeze=False)
        plot_hist(axes[0], s0, s1, args.index)

    elif args.mode == 'scatter':
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 7), squeeze=False)
        if args.index in ('0', 'both'):
            plot_scatter(axes[0, 0], x0[sel], y0[sel], s0[sel], env,
                         args.cmap, vmin, vmax, 'Start configs -- speed')
        if args.index in ('1', 'both'):
            plot_scatter(axes[0, -1], x1[sel], y1[sel], s1[sel], env,
                         args.cmap, vmin, vmax, 'End configs -- speed')

    elif args.mode == 'shapes':
        if not os.path.exists(args.shape_dxf):
            print(f'ERROR: shape DXF not found: {args.shape_dxf}')
            sys.exit(1)
        shape_poly = dxf_to_shape(args.shape_dxf)
        shape_points = shape_to_points(shape_poly, shape_scale)
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 7), squeeze=False)
        if args.index in ('0', 'both'):
            xy0 = np.stack([x0[sel], y0[sel]], axis=1)
            plot_shapes(axes[0, 0], xy0, t0[sel], s0[sel], shape_points,
                        env, args.cmap, vmin, vmax,
                        'Start configs -- speed', args.cnt)
        if args.index in ('1', 'both'):
            xy1 = np.stack([x1[sel], y1[sel]], axis=1)
            plot_shapes(axes[0, -1], xy1, t1[sel], s1[sel], shape_points,
                        env, args.cmap, vmin, vmax,
                        'End configs -- speed', args.cnt)

    elif args.mode == 'theta':
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 7), squeeze=False)
        if args.index in ('0', 'both'):
            plot_theta(axes[0, 0], x0[sel], y0[sel], t0[sel], env,
                       args.cmap, 'Start configs -- theta (rad)')
        if args.index in ('1', 'both'):
            plot_theta(axes[0, -1], x1[sel], y1[sel], t1[sel], env,
                       args.cmap, 'End configs -- theta (rad)')

    else:  # map
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 7), squeeze=False)
        if args.index in ('0', 'both'):
            plot_map(axes[0, 0], x0, y0, s0, env, args.cmap, vmin, vmax,
                     'Start configs -- speed map',
                     grid_n=args.grid_n, iso_levels=args.iso)
        if args.index in ('1', 'both'):
            plot_map(axes[0, -1], x1, y1, s1, env, args.cmap, vmin, vmax,
                     'End configs -- speed map',
                     grid_n=args.grid_n, iso_levels=args.iso)

    fig.suptitle(
        f'Training data: {os.path.basename(os.path.normpath(args.data))}  '
        f'(N={N}, mode={args.mode}, index={args.index}, shape_scale={shape_scale})',
        fontsize=10)
    fig.tight_layout()

    out_name = args.out or f'speeds_{args.mode}_{args.index}.png'
    out_path = os.path.join(args.out_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')

    # Always also write a histogram of the overall speed distribution
    # (x = speed value, y = number of points), regardless of --mode.
    all_speeds = speed.ravel()
    hfig, hax = plt.subplots(figsize=(7, 5))
    hax.hist(all_speeds, bins=60, color='steelblue', edgecolor='black', alpha=0.8)
    hax.set_xlabel('speed value')
    hax.set_ylabel('number of points')
    hax.set_title(f'Speed distribution (N={all_speeds.size})')
    hax.grid(axis='y', linestyle='--', alpha=0.5)
    hfig.tight_layout()
    hist_path = os.path.join(args.out_dir, 'speeds_distribution.png')
    hfig.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close(hfig)
    print(f'Saved {hist_path}')


if __name__ == '__main__':
    main()
