"""Visualize the trained travel-time field for the 2-D shape task.

For a fixed start configuration ``(x0, y0, theta0)`` and a fixed destination
orientation ``theta_vis``, this queries

    T(start -> (x, y, theta_vis))

over a dense (x, y) grid and renders it as a heatmap with isochrone contour
lines, the environment boundary, and the origin F-shape drawn for context.

It mirrors ``2dshape_new/visualize_travel_field.py`` but runs against the
ntrl-demo network (``models.metric.model_train_metric``).  Note that theta is
stored normalized by ``2*pi`` in the training data, so theta arguments given on
the command line (in radians) are divided by ``2*pi`` before being fed to the
network.

Output PNGs are always written to disk (no GUI is used).

Usage
-----
    python visualize_travel_field.py --pt Experiments/2dshape/<run>/Model_Epoch_05000_*.pt
    python visualize_travel_field.py --run_dir Experiments/2dshape/<run> --animate
"""

import sys, os, re, json, argparse
sys.path.append('.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from glob import glob
from shapely.geometry import Polygon

from models.metric import model_train_metric as md
from dataprocessing.parse_shape import dxf_to_shape, shape_to_points


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_ENV    = './datasets/2dshape/Fmaze2_norm/env.npy'
DEFAULT_FSHAPE = './datasets/2dshape/Fshape_norm.dxf'
MODEL_PATH     = './Experiments/2dshape'              # only used to build a folder name
DATA_PATH      = './datasets/2dshape/Fmaze2_norm'
DEVICE         = 'cuda'
DIM            = 3          # (x, y, theta)
INFER_BATCH    = 8192


# ─── Helpers ─────────────────────────────────────────────────────────────────
def rotate_points_np(points, x, y, theta):
    """Rotate 2-D shape points by raw (x, y, theta) -> list of (x, y) tuples."""
    c, s = np.cos(theta), np.sin(theta)
    pts = np.array(points, dtype=np.float32)
    rot = np.array([[c, -s], [s, c]])
    return [tuple(p) for p in (rot @ pts.T).T + np.array([x, y])]


def infer_travel_times(model, origin_x, origin_y, origin_theta,
                       x_flat, y_flat, theta_vis):
    """Query T(origin -> (x, y, theta_vis)) for every grid point.

    theta values arrive in radians and are normalized by 2*pi to match the
    convention used by preprocess_dxf.py / the training data.
    """
    n = len(x_flat)
    two_pi = 2.0 * np.pi
    q = np.stack([
        np.full(n, origin_x,                dtype=np.float32),
        np.full(n, origin_y,                dtype=np.float32),
        np.full(n, origin_theta / two_pi,   dtype=np.float32),
        x_flat.astype(np.float32),
        y_flat.astype(np.float32),
        np.full(n, theta_vis / two_pi,      dtype=np.float32),
    ], axis=1)                                           # (N, 6)

    tt = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, INFER_BATCH):
            j = min(i + INFER_BATCH, n)
            batch = torch.tensor(q[i:j], device=DEVICE)
            tt[i:j] = model.function.TravelTimes(batch).cpu().numpy()
    return tt


def epoch_from_path(pt_path):
    m = re.search(r'Epoch_(\d+)', os.path.basename(pt_path))
    return int(m.group(1)) if m else 0


def load_checkpoints(pt_path, run_dir):
    """Return a sorted list of (epoch, filepath) pairs to visualize."""
    if run_dir:
        files = sorted(glob(os.path.join(run_dir, '*.pt')), key=epoch_from_path)
        seen = {}
        for f in files:
            ep = epoch_from_path(f)
            if ep not in seen:
                seen[ep] = f
        return sorted(seen.items())
    return [(epoch_from_path(pt_path), pt_path)]


# ─── Per-checkpoint rendering ────────────────────────────────────────────────
def render_field(ax, model, env_pts, Fshape_pts,
                 origin_x, origin_y, origin_theta,
                 xs, ys, XX, YY, theta_vis, cmap, iso_levels,
                 vmin=None, vmax=None):
    """Draw the travel-time field on `ax`; returns the imshow handle."""
    grid_n = len(xs)
    tt = infer_travel_times(model, origin_x, origin_y, origin_theta,
                            XX.ravel(), YY.ravel(), theta_vis)
    TT = tt.reshape(grid_n, grid_n)

    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = ys[0], ys[-1]
    if vmin is None:
        vmin = tt.min()
    if vmax is None:
        vmax = tt.max()

    im = ax.imshow(TT, origin='lower', extent=[xmin, xmax, ymin, ymax],
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', alpha=0.80)

    cs = ax.contour(xs, ys, TT, levels=iso_levels,
                    colors='white', linewidths=0.7, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')

    if env_pts is not None:
        ax.scatter(env_pts[:, 0], env_pts[:, 1], c='black', s=4, zorder=3,
                   rasterized=True)

    origin_pts = rotate_points_np(Fshape_pts, origin_x, origin_y, origin_theta)
    if len(origin_pts) >= 3:
        op = Polygon(origin_pts)
        if op.is_valid:
            ax.add_patch(plt.Polygon(list(op.exterior.coords),
                                     facecolor='red', edgecolor='black',
                                     linewidth=0.8, alpha=1.0, zorder=10))
    ax.scatter([origin_x], [origin_y], c='red', s=40, zorder=11,
               label=f'origin ({origin_x:.2f}, {origin_y:.2f}, theta={origin_theta:.2f})')

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=7, loc='upper right')
    return im, TT, vmin, vmax


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pt', default=None, help='Single checkpoint .pt file.')
    parser.add_argument('--run_dir', default=None,
                        help='Training run folder; overrides --pt for the multi-epoch view.')
    parser.add_argument('--env', default=DEFAULT_ENV)
    parser.add_argument('--fshape', default=DEFAULT_FSHAPE)
    parser.add_argument('--x0', type=float, default=0.48, help='origin x')
    parser.add_argument('--y0', type=float, default=-0.1, help='origin y')
    parser.add_argument('--theta0', type=float, default=1.57, help='origin theta (rad)')
    parser.add_argument('--theta_vis', type=float, default=1.57,
                        help='fixed destination theta (rad)')
    parser.add_argument('--grid_n', type=int, default=120, help='grid resolution')
    parser.add_argument('--iso', type=int, default=12, help='# isochrone levels')
    parser.add_argument('--cmap', default='plasma')
    parser.add_argument('--shape_scale', type=float, default=None,
                        help='F-shape scale (defaults to value in meta.json, else 1.0)')
    parser.add_argument('--data', default=DATA_PATH,
                        help='Training-data dir (used to read meta.json)')
    parser.add_argument('--out_dir', default='.',
                        help='directory the PNG / MP4 outputs are written to')
    parser.add_argument('--animate', action='store_true',
                        help='save an MP4 of the field over training (needs --run_dir + ffmpeg)')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.pt is None and args.run_dir is None:
        parser.error('provide --pt <checkpoint> or --run_dir <run folder>')

    # ── shape_scale: meta.json wins, then --shape_scale, then 1.0 ───────────
    shape_scale = args.shape_scale
    meta_path = os.path.join(args.data, 'meta.json')
    if shape_scale is None and os.path.exists(meta_path):
        with open(meta_path) as f:
            shape_scale = float(json.load(f).get('shape_scale', 1.0))
        print(f'Loaded shape_scale = {shape_scale} from {meta_path}')
    if shape_scale is None:
        shape_scale = 1.0

    # ── Environment + shape ──────────────────────────────────────────────────
    env_pts = np.load(args.env) if os.path.exists(args.env) else None
    Fshape = dxf_to_shape(args.fshape)
    Fshape_pts = shape_to_points(Fshape, shape_scale)

    if env_pts is not None:
        xmin, xmax = env_pts[:, 0].min(), env_pts[:, 0].max()
        ymin, ymax = env_pts[:, 1].min(), env_pts[:, 1].max()
    else:
        xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5
    print(f'Env bounds: x=[{xmin:.3f}, {xmax:.3f}]  y=[{ymin:.3f}, {ymax:.3f}]')

    xs = np.linspace(xmin, xmax, args.grid_n)
    ys = np.linspace(ymin, ymax, args.grid_n)
    XX, YY = np.meshgrid(xs, ys)

    checkpoints = load_checkpoints(args.pt, args.run_dir)
    print(f'Found {len(checkpoints)} checkpoint(s).')

    # Dummy model -- only .load() is exercised.
    womodel = md.Model(MODEL_PATH, DATA_PATH, DIM, [0.0] * DIM, device=DEVICE)

    # ── Single checkpoint ────────────────────────────────────────────────────
    if len(checkpoints) == 1:
        epoch, pt_file = checkpoints[0]
        print(f'Loading epoch {epoch}: {pt_file}')
        womodel.load(pt_file)
        womodel.network.eval()

        fig, ax = plt.subplots(figsize=(8, 7))
        im, TT, vmin, vmax = render_field(
            ax, womodel, env_pts, Fshape_pts,
            args.x0, args.y0, args.theta0,
            xs, ys, XX, YY, args.theta_vis, args.cmap, args.iso)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Travel time  T(start -> x, y)')
        ax.set_title(f'Travel-time field — epoch {epoch}\n'
                     f'origin ({args.x0:.2f}, {args.y0:.2f}, theta={args.theta0:.2f})   '
                     f'dest theta = {args.theta_vis:.2f}')
        fig.tight_layout()
        out = os.path.join(args.out_dir, f'travel_field_epoch{epoch:05d}.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'Saved {out}')
        return

    # ── Multi-checkpoint: compute every field first ──────────────────────────
    print('Pre-computing travel-time fields ...')
    fields, epochs = [], []
    for epoch, pt_file in checkpoints:
        print(f'  epoch {epoch}: {pt_file}')
        womodel.load(pt_file)
        womodel.network.eval()
        tt = infer_travel_times(womodel, args.x0, args.y0, args.theta0,
                                XX.ravel(), YY.ravel(), args.theta_vis)
        fields.append(tt.reshape(args.grid_n, args.grid_n))
        epochs.append(epoch)

    global_min = min(f.min() for f in fields)
    global_max = max(f.max() for f in fields)
    norm_global = Normalize(vmin=global_min, vmax=global_max)
    cmap_obj = get_cmap(args.cmap)

    # ── Multi-panel figure ───────────────────────────────────────────────────
    ncols = min(len(checkpoints), 4)
    nrows = (len(checkpoints) + ncols - 1) // ncols
    fig_grid, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    for idx, (epoch, TT) in enumerate(zip(epochs, fields)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.imshow(TT, origin='lower', extent=[xmin, xmax, ymin, ymax],
                  cmap=args.cmap, vmin=global_min, vmax=global_max,
                  aspect='equal', alpha=0.80)
        cs = ax.contour(xs, ys, TT, levels=args.iso,
                        colors='white', linewidths=0.5, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=5, fmt='%.1f')
        if env_pts is not None:
            ax.scatter(env_pts[:, 0], env_pts[:, 1],
                       c='lightgrey', s=1, zorder=3, rasterized=True)
        op = rotate_points_np(Fshape_pts, args.x0, args.y0, args.theta0)
        if len(op) >= 3 and Polygon(op).is_valid:
            ax.add_patch(plt.Polygon(list(Polygon(op).exterior.coords),
                                     facecolor='red', edgecolor='black',
                                     linewidth=0.5, alpha=1.0, zorder=10))
        ax.scatter([args.x0], [args.y0], c='red', s=20, zorder=11)
        ax.set_title(f'Epoch {epoch}', fontsize=9)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    for idx in range(len(checkpoints), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=norm_global)
    sm.set_array([])
    fig_grid.colorbar(sm, ax=axes.ravel().tolist(),
                      label='Travel time  T(start -> x, y)', shrink=0.6)
    fig_grid.suptitle(f'Travel-time field evolution\n'
                      f'origin ({args.x0:.2f}, {args.y0:.2f}, theta={args.theta0:.2f})   '
                      f'dest theta = {args.theta_vis:.2f}', y=1.01)
    fig_grid.tight_layout()
    grid_out = os.path.join(args.out_dir, 'travel_field_all_epochs.png')
    fig_grid.savefig(grid_out, dpi=120, bbox_inches='tight')
    plt.close(fig_grid)
    print(f'Saved {grid_out}')

    # ── Animation ────────────────────────────────────────────────────────────
    if args.animate:
        print('Building animation ...')
        fig_anim, ax_anim = plt.subplots(figsize=(7, 6))
        im_anim = ax_anim.imshow(fields[0], origin='lower',
                                 extent=[xmin, xmax, ymin, ymax], cmap=args.cmap,
                                 vmin=global_min, vmax=global_max,
                                 aspect='equal', alpha=0.80)
        sm2 = plt.cm.ScalarMappable(cmap=args.cmap, norm=norm_global)
        sm2.set_array([])
        fig_anim.colorbar(sm2, ax=ax_anim, label='Travel time')
        if env_pts is not None:
            ax_anim.scatter(env_pts[:, 0], env_pts[:, 1],
                            c='lightgrey', s=2, zorder=3, rasterized=True)
        op = rotate_points_np(Fshape_pts, args.x0, args.y0, args.theta0)
        if len(op) >= 3 and Polygon(op).is_valid:
            ax_anim.add_patch(plt.Polygon(list(Polygon(op).exterior.coords),
                                          facecolor='red', edgecolor='black',
                                          linewidth=0.8, alpha=1.0, zorder=10))
        contour_holder = [None]
        title_obj = ax_anim.set_title('')
        ax_anim.set_xlim(xmin, xmax); ax_anim.set_ylim(ymin, ymax)
        ax_anim.set_xlabel('x'); ax_anim.set_ylabel('y')

        def update(frame):
            TT = fields[frame]
            im_anim.set_data(TT)
            if contour_holder[0] is not None:
                # matplotlib >=3.8 removed QuadContourSet.collections; use .remove()
                try:
                    contour_holder[0].remove()
                except AttributeError:
                    for coll in contour_holder[0].collections:
                        coll.remove()
            contour_holder[0] = ax_anim.contour(xs, ys, TT, levels=args.iso,
                                                colors='white', linewidths=0.6, alpha=0.7)
            title_obj.set_text(f'Epoch {epochs[frame]}  |  '
                               f'origin ({args.x0:.2f}, {args.y0:.2f}, theta={args.theta0:.2f})')
            return [im_anim]

        ani = animation.FuncAnimation(fig_anim, update, frames=len(checkpoints),
                                      interval=800, blit=False)
        anim_out = os.path.join(args.out_dir, 'travel_field_training.mp4')
        ani.save(anim_out, writer='ffmpeg', fps=2, dpi=120)
        print(f'Saved {anim_out}')
        plt.close(fig_anim)


if __name__ == '__main__':
    main()
