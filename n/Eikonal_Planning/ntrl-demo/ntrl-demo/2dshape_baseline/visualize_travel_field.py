"""
visualize_travel_field.py

Visualize the neural network travel-time field in 2-D by collapsing the theta
dimension.  For a fixed start configuration (x0, y0, theta0) and a fixed
destination orientation theta_vis, the script queries

    T(start → (x, y, theta_vis))

over a dense (x, y) grid and plots the result as a colour heatmap with
isochrone (equal-travel-time) contour lines and F-shapes drawn at a coarser
sub-grid for physical context.

Optionally, pass a training run directory to visualize how the field evolves
across saved checkpoints (produces a multi-panel figure + an animation).

Usage
-----
    python visualize_travel_field.py                  # single checkpoint, defaults
    python visualize_travel_field.py --help           # see all arguments

Key parameters (edit defaults below or pass as CLI args):
    --pt       path to a single .pt checkpoint
    --run_dir  path to a training run folder (auto-finds all *.pt files)
    --x0 / --y0 / --theta0   origin configuration
    --theta_vis              fixed destination theta (radians)
    --grid_n                 fine grid resolution  (default 100)
    --shape_n                coarser F-shape grid resolution (default 14)
    --save                   save figure(s) instead of showing interactively
"""

import sys, os, re, argparse
sys.path.append('.')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from glob import glob
from shapely.geometry import Polygon

import model2d as md
from parse_shape import dxf_to_shape, shape_to_points


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_PT = './Experiments/latest/Model_latest.pt'
DEFAULT_ENV   = './testing_data2d/Fshape_FmazeEasy/Fmaze3env.npy'
DEFAULT_FSHAPE = './datasets/Fshape_norm.dxf'
MODEL_PATH    = './Experiments/UR5'      # dummy path required by md.Model
DATA_PATH     = './datasets/arm/Auburn'  # dummy path required by md.Model
DEVICE        = 'cuda'
DIM           = 3   # (x, y, theta)
INFER_BATCH   = 8192


# ─── Helpers ─────────────────────────────────────────────────────────────────

def rotate_points_np(points, x, y, theta):
    """Rotate 2-D shape_points by (x, y, theta) and return list of (x,y) tuples."""
    c, s = np.cos(theta), np.sin(theta)
    pts  = np.array(points, dtype=np.float32)   # (N, 2)
    rot  = np.array([[c, -s], [s, c]])
    out  = (rot @ pts.T).T + np.array([x, y])
    return [tuple(p) for p in out]


def infer_travel_times(model, origin_x, origin_y, origin_theta,
                       x_flat, y_flat, theta_vis):
    """Query T(origin → (x, y, theta_vis)) for every point in x_flat / y_flat."""
    n    = len(x_flat)
    q    = np.stack([
        np.full(n, origin_x,     dtype=np.float32),
        np.full(n, origin_y,     dtype=np.float32),
        np.full(n, origin_theta, dtype=np.float32),
        x_flat.astype(np.float32),
        y_flat.astype(np.float32),
        np.full(n, theta_vis,    dtype=np.float32),
    ], axis=1)                               # (N, 6)

    tt = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, INFER_BATCH):
            j     = min(i + INFER_BATCH, n)
            batch = torch.tensor(q[i:j], device=DEVICE)
            tt[i:j] = model.function.TravelTimes(batch).cpu().numpy()
    return tt


def epoch_from_path(pt_path):
    """Extract epoch number from checkpoint filename."""
    m = re.search(r'Epoch_(\d+)', os.path.basename(pt_path))
    return int(m.group(1)) if m else 0


def load_checkpoints(pt_path, run_dir):
    """Return a sorted list of (epoch, filepath) pairs to visualize."""
    if run_dir:
        files = sorted(glob(os.path.join(run_dir, '*.pt')),
                       key=epoch_from_path)
        # skip duplicate epoch=1 files; keep the one with lowest loss
        seen = {}
        for f in files:
            ep = epoch_from_path(f)
            if ep not in seen:
                seen[ep] = f
        return sorted(seen.items())
    return [(epoch_from_path(pt_path), pt_path)]


# ─── Per-checkpoint rendering ─────────────────────────────────────────────────

def render_field(ax, model, env_pts, Fshape_pts,
                 origin_x, origin_y, origin_theta,
                 xs, ys, XX, YY, theta_vis,
                 shape_xs, shape_ys, shape_XX, shape_YY,
                 cmap, vmin=None, vmax=None, iso_levels=12):
    """
    Draw the travel-time field on `ax`.  Returns the imshow object so the
    caller can share a colourbar.
    """
    GRID_N   = len(xs)
    x_flat   = XX.ravel()
    y_flat   = YY.ravel()

    # ── Fine heatmap ─────────────────────────────────────────────────────────
    tt = infer_travel_times(model, origin_x, origin_y, origin_theta,
                            x_flat, y_flat, theta_vis)
    TT = tt.reshape(GRID_N, GRID_N)

    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = ys[0], ys[-1]

    if vmin is None: vmin = tt.min()
    if vmax is None: vmax = tt.max()

    im = ax.imshow(
        TT,
        origin='lower',
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        aspect='equal',
        alpha=0.80,
    )

    # ── Isochrone contours ────────────────────────────────────────────────────
    cs = ax.contour(
        xs, ys, TT,
        levels=iso_levels,
        colors='white',
        linewidths=0.7,
        alpha=0.75,
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')

    # # ── Coarser F-shape grid ──────────────────────────────────────────────────
    # norm  = Normalize(vmin=vmin, vmax=vmax)
    # cmap_ = get_cmap(cmap)

    # sx_flat = shape_XX.ravel()
    # sy_flat = shape_YY.ravel()
    # tt_shape = infer_travel_times(model, origin_x, origin_y, origin_theta,
    #                               sx_flat, sy_flat, theta_vis)

    # for k, (gx, gy, tval) in enumerate(zip(sx_flat, sy_flat, tt_shape)):
    #     rotated = rotate_points_np(Fshape_pts, gx, gy, theta_vis)
    #     if len(rotated) < 3:
    #         continue
    #     poly = Polygon(rotated)
    #     if not poly.is_valid:
    #         continue
    #     color = cmap_(norm(tval))
    #     patch = plt.Polygon(
    #         list(poly.exterior.coords),
    #         facecolor=color, edgecolor='black',
    #         linewidth=0.3, alpha=0.85, zorder=4,
    #     )
    #     ax.add_patch(patch)

    # ── Environment boundary ──────────────────────────────────────────────────
    ax.scatter(env_pts[:, 0], env_pts[:, 1],
               c='black', s=4, zorder=3, rasterized=True)

    # ── Origin F-shape (red) ─────────────────────────────────────────────────
    origin_pts = rotate_points_np(Fshape_pts, origin_x, origin_y, origin_theta)
    if len(origin_pts) >= 3:
        op = Polygon(origin_pts)
        if op.is_valid:
            ax.add_patch(plt.Polygon(
                list(op.exterior.coords),
                facecolor='red', edgecolor='black',
                linewidth=0.8, alpha=1.0, zorder=10,
            ))
    ax.scatter([origin_x], [origin_y], c='red', s=50, zorder=11,
               label=f'Origin ({origin_x:.2f},{origin_y:.2f}, θ={origin_theta:.2f})')

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=7, loc='upper right')

    return im, TT, vmin, vmax


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pt',        default=DEFAULT_PT,  help='Single checkpoint .pt')
    parser.add_argument('--run_dir',   default=None,
                        help='Folder of checkpoints; overrides --pt for multi-epoch view')
    parser.add_argument('--env',       default=DEFAULT_ENV)
    parser.add_argument('--fshape',    default=DEFAULT_FSHAPE)
    parser.add_argument('--x0',        type=float, default=0.0,  help='Origin x')
    parser.add_argument('--y0',        type=float, default=0.0,  help='Origin y')
    parser.add_argument('--theta0',    type=float, default=0.0,  help='Origin theta (rad)')
    parser.add_argument('--theta_vis', type=float, default=0.0,
                        help='Fixed destination theta (rad)')
    parser.add_argument('--grid_n',    type=int,   default=100,  help='Fine grid resolution')
    parser.add_argument('--shape_n',   type=int,   default=14,
                        help='Coarser F-shape grid resolution')
    parser.add_argument('--iso',       type=int,   default=12,   help='# isochrone levels')
    parser.add_argument('--cmap',      default='plasma')
    parser.add_argument('--save',      action='store_true', help='Save figures to disk')
    parser.add_argument('--animate',   action='store_true',
                        help='Save MP4 animation (requires ffmpeg, only with --run_dir)')
    args = parser.parse_args()

    # ── Load environment ──────────────────────────────────────────────────────
    env_pts    = np.load(args.env)          # (M, 2)
    Fshape     = dxf_to_shape(args.fshape)
    Fshape_pts = list(Fshape.exterior.coords)
    scale = 0.1  # shrink to 50%

    Fshape_pts_scaled = [
        (x * scale, y * scale)
        for x, y in Fshape_pts
    ]
    xmin, xmax = env_pts[:, 0].min(), env_pts[:, 0].max()
    ymin, ymax = env_pts[:, 1].min(), env_pts[:, 1].max()
    print(f'Env bounds: x=[{xmin:.3f}, {xmax:.3f}]  y=[{ymin:.3f}, {ymax:.3f}]')

    # ── Grids ─────────────────────────────────────────────────────────────────
    xs = np.linspace(xmin, xmax, args.grid_n)
    ys = np.linspace(ymin, ymax, args.grid_n)
    XX, YY = np.meshgrid(xs, ys)

    # coarser grid for F-shapes (exclude points too close to the boundary)
    pad = (xmax - xmin) / (args.shape_n * 2)
    sxs = np.linspace(xmin + pad, xmax - pad, args.shape_n)
    sys_ = np.linspace(ymin + pad, ymax - pad, args.shape_n)
    shape_XX, shape_YY = np.meshgrid(sxs, sys_)

    # ── Gather checkpoints ────────────────────────────────────────────────────
    checkpoints = load_checkpoints(args.pt, args.run_dir)
    print(f'Found {len(checkpoints)} checkpoint(s).')

    # ── Initialise a dummy model (just for .load) ─────────────────────────────
    womodel = md.Model(MODEL_PATH, DATA_PATH, DIM, [0.0] * 4, device=DEVICE)

    # ── Single checkpoint ─────────────────────────────────────────────────────
    if len(checkpoints) == 1:
        epoch, pt_file = checkpoints[0]
        print(f'Loading epoch {epoch}: {pt_file}')
        womodel.load(pt_file)
        womodel.network.eval()

        fig, ax = plt.subplots(figsize=(8, 7))
        im, TT, vmin, vmax = render_field(
            ax, womodel, env_pts, Fshape_pts,
            args.x0, args.y0, args.theta0,
            xs, ys, XX, YY, args.theta_vis,
            sxs, sys_, shape_XX, shape_YY,
            args.cmap, iso_levels=args.iso,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Travel time  T(start → x,y)', fontsize=10)
        ax.set_title(
            f'Travel-time field — epoch {epoch}\n'
            f'Origin ({args.x0:.2f}, {args.y0:.2f}, θ={args.theta0:.2f} rad)   '
            f'dest θ = {args.theta_vis:.2f} rad',
            fontsize=10,
        )
        fig.tight_layout()
        if args.save:
            out = f'travel_field_epoch{epoch:05d}.png'
            fig.savefig(out, dpi=150)
            print(f'Saved {out}')
        else:
            plt.show()
        return

    # ── Multi-checkpoint: compute all fields first ────────────────────────────
    print('Pre-computing travel-time fields for all checkpoints …')
    fields   = []   # list of TT arrays  (grid_n, grid_n)
    epochs   = []
    x_flat   = XX.ravel()
    y_flat   = YY.ravel()
    sx_flat  = shape_XX.ravel()
    sy_flat  = shape_YY.ravel()
    tt_shapes = []

    for epoch, pt_file in checkpoints:
        print(f'  epoch {epoch}: {pt_file}')
        womodel.load(pt_file)
        womodel.network.eval()

        tt = infer_travel_times(womodel, args.x0, args.y0, args.theta0,
                                x_flat, y_flat, args.theta_vis)
        fields.append(tt.reshape(args.grid_n, args.grid_n))
        epochs.append(epoch)

        tt_s = infer_travel_times(womodel, args.x0, args.y0, args.theta0,
                                  sx_flat, sy_flat, args.theta_vis)
        tt_shapes.append(tt_s)

    # Shared colour scale across all epochs for fair comparison
    global_min = min(f.min() for f in fields)
    global_max = max(f.max() for f in fields)
    norm_global = Normalize(vmin=global_min, vmax=global_max)
    cmap_obj    = get_cmap(args.cmap)

    # ── Multi-panel figure ────────────────────────────────────────────────────
    ncols = min(len(checkpoints), 4)
    nrows = (len(checkpoints) + ncols - 1) // ncols

    fig_grid, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4.5 * nrows),
                                  squeeze=False)

    for idx, (epoch, TT, tt_s) in enumerate(zip(epochs, fields, tt_shapes)):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]

        im = ax.imshow(
            TT,
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap=args.cmap,
            vmin=global_min, vmax=global_max,
            aspect='equal',
            alpha=0.80,
        )
        cs = ax.contour(xs, ys, TT, levels=args.iso,
                        colors='white', linewidths=0.5, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=5, fmt='%.1f')

        # F-shapes at coarser grid
        for gx, gy, tval in zip(sx_flat, sy_flat, tt_s):
            rotated = rotate_points_np(Fshape_pts_scaled, gx, gy, args.theta_vis)
            if len(rotated) < 3:
                continue
            poly = Polygon(rotated)
            if not poly.is_valid:
                continue
            color = cmap_obj(norm_global(tval))
            ax.add_patch(plt.Polygon(
                list(poly.exterior.coords),
                facecolor=color, edgecolor='black',
                linewidth=0.2, alpha=0.85, zorder=4,
            ))

        # Env boundary
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c='lightgrey', s=1, zorder=3, rasterized=True)

        # Origin
        op = rotate_points_np(Fshape_pts, args.x0, args.y0, args.theta0)
        if len(op) >= 3:
            poly_o = Polygon(op)
            if poly_o.is_valid:
                ax.add_patch(plt.Polygon(
                    list(poly_o.exterior.coords),
                    facecolor='red', edgecolor='black',
                    linewidth=0.5, alpha=1.0, zorder=10,
                ))
        ax.scatter([args.x0], [args.y0], c='red', s=20, zorder=11)

        ax.set_title(f'Epoch {epoch}', fontsize=9)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)

    # Hide any unused axes
    for idx in range(len(checkpoints), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=norm_global)
    sm.set_array([])
    fig_grid.colorbar(sm, ax=axes.ravel().tolist(),
                      label='Travel time  T(start → x,y)', shrink=0.6)

    fig_grid.suptitle(
        f'Travel-time field evolution\n'
        f'Origin ({args.x0:.2f}, {args.y0:.2f}, θ={args.theta0:.2f} rad)   '
        f'dest θ = {args.theta_vis:.2f} rad',
        fontsize=11, y=1.01,
    )
    fig_grid.tight_layout()

    if args.save:
        out_grid = 'travel_field_all_epochs.png'
        fig_grid.savefig(out_grid, dpi=120, bbox_inches='tight')
        print(f'Saved {out_grid}')
    else:
        plt.show()

    # ── Animation ─────────────────────────────────────────────────────────────
    if args.animate:
        print('Building animation …')
        fig_anim, ax_anim = plt.subplots(figsize=(7, 6))
        fig_anim.subplots_adjust(right=0.85)
        cax = fig_anim.add_axes([0.87, 0.15, 0.03, 0.7])

        im_anim = ax_anim.imshow(
            fields[0],
            origin='lower',
            extent=[xmin, xmax, ymin, ymax],
            cmap=args.cmap,
            vmin=global_min, vmax=global_max,
            aspect='equal',
            alpha=0.80,
        )
        sm2 = plt.cm.ScalarMappable(cmap=args.cmap, norm=norm_global)
        sm2.set_array([])
        fig_anim.colorbar(sm2, cax=cax, label='Travel time')

        # Environment boundary drawn once
        ax_anim.scatter(env_pts[:, 0], env_pts[:, 1],
                        c='lightgrey', s=2, zorder=3, rasterized=True)

        # Origin drawn once
        op = rotate_points_np(Fshape_pts, args.x0, args.y0, args.theta0)
        if len(op) >= 3:
            poly_o = Polygon(op)
            if poly_o.is_valid:
                ax_anim.add_patch(plt.Polygon(
                    list(poly_o.exterior.coords),
                    facecolor='red', edgecolor='black',
                    linewidth=0.8, alpha=1.0, zorder=10,
                ))

        contour_holder = [None]
        shape_patches  = []
        title_obj      = ax_anim.set_title('')
        ax_anim.set_xlim(xmin, xmax)
        ax_anim.set_ylim(ymin, ymax)
        ax_anim.set_xlabel('x'); ax_anim.set_ylabel('y')

        def update(frame):
            TT   = fields[frame]
            tt_s = tt_shapes[frame]
            ep   = epochs[frame]

            im_anim.set_data(TT)

            # Remove old contours
            if contour_holder[0] is not None:
                for coll in contour_holder[0].collections:
                    coll.remove()

            contour_holder[0] = ax_anim.contour(
                xs, ys, TT, levels=args.iso,
                colors='white', linewidths=0.6, alpha=0.7,
            )

            # Remove old F-shape patches
            for p in shape_patches:
                p.remove()
            shape_patches.clear()

            for gx, gy, tval in zip(sx_flat, sy_flat, tt_s):
                rotated = rotate_points_np(Fshape_pts, gx, gy, args.theta_vis)
                if len(rotated) < 3:
                    continue
                poly = Polygon(rotated)
                if not poly.is_valid:
                    continue
                color = cmap_obj(norm_global(tval))
                p = plt.Polygon(
                    list(poly.exterior.coords),
                    facecolor=color, edgecolor='black',
                    linewidth=0.2, alpha=0.85, zorder=4,
                )
                ax_anim.add_patch(p)
                shape_patches.append(p)

            title_obj.set_text(
                f'Epoch {ep}  |  '
                f'Origin ({args.x0:.2f},{args.y0:.2f},θ={args.theta0:.2f})  '
                f'dest θ={args.theta_vis:.2f}'
            )
            return [im_anim]

        ani = animation.FuncAnimation(
            fig_anim, update,
            frames=len(checkpoints),
            interval=800,
            blit=False,
        )
        out_anim = 'travel_field_training.mp4'
        ani.save(out_anim, writer='ffmpeg', fps=2, dpi=120)
        print(f'Saved animation: {out_anim}')
        plt.close(fig_anim)


if __name__ == '__main__':
    main()
