"""
animate_travel_field.py

Render a video of the neural travel-time field evolving over training epochs.

Usage
-----
    python animate_travel_field.py \\
        --run_dir ./Experiments/Fshape_FmazeEasy/training_data2d_05_11_13_53 \\
        --x0 0 --start_epoch 1 --end_epoch 1000 --n_samples 20

Parameters
----------
--run_dir       Directory containing Model_Epoch_*.pt checkpoint files (required)
--start_epoch   First epoch to include  (default: earliest found)
--end_epoch     Last epoch to include   (default: latest found)
--n_samples     Number of evenly-spaced frames to sample from the range
                (default: use every checkpoint found in range)
--output        Output filename          (default: travel_field_evolution.mp4)
--fps           Frames per second        (default: 5)
--x0/--y0/--theta0   Origin configuration
--theta_vis     Fixed destination theta (rad)
--grid_n        Grid resolution for the heatmap (default: 100)
--iso           Number of isochrone contour levels (default: 12)
--cmap          Matplotlib colormap name (default: plasma)
--env           Path to environment .npy file
--fshape        Path to F-shape .dxf file
"""

import sys, os, re, argparse
sys.path.append('.')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed — we save to file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from glob import glob

import model2d as md
from parse_shape import dxf_to_shape, shape_to_points

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_ENV    = './testing_data2d/Fshape_FmazeEasy/Fmaze3env.npy'
DEFAULT_FSHAPE = './datasets/Fshape_norm.dxf'
MODEL_PATH     = './Experiments/UR5'
DATA_PATH      = './datasets/arm/Auburn'
DEVICE         = 'cuda'
DIM            = 3
INFER_BATCH    = 8192


# ─── Helpers ──────────────────────────────────────────────────────────────────

def epoch_from_path(p):
    m = re.search(r'Epoch_(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0


def collect_checkpoints(run_dir, start_epoch, end_epoch):
    """Return sorted list of (epoch, path) within [start_epoch, end_epoch]."""
    files = sorted(glob(os.path.join(run_dir, '*.pt')), key=epoch_from_path)
    seen = {}
    for f in files:
        ep = epoch_from_path(f)
        if ep not in seen:
            seen[ep] = f
    pairs = sorted((ep, p) for ep, p in seen.items()
                   if start_epoch <= ep <= end_epoch)
    return pairs


def subsample(pairs, n):
    """Pick n evenly-spaced entries from pairs (by index)."""
    if n is None or n >= len(pairs):
        return pairs
    indices = np.round(np.linspace(0, len(pairs) - 1, n)).astype(int)
    seen_idx = set()
    result = []
    for i in indices:
        if i not in seen_idx:
            result.append(pairs[i])
            seen_idx.add(i)
    return result


def infer_field(model, ox, oy, ot, XX, YY, theta_vis):
    """Return (grid_n, grid_n) travel-time array."""
    x_flat = XX.ravel().astype(np.float32)
    y_flat = YY.ravel().astype(np.float32)
    n = len(x_flat)
    q = np.stack([
        np.full(n, ox,       dtype=np.float32),
        np.full(n, oy,       dtype=np.float32),
        np.full(n, ot,       dtype=np.float32),
        x_flat,
        y_flat,
        np.full(n, theta_vis, dtype=np.float32),
    ], axis=1)
    tt = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, INFER_BATCH):
            j = min(i + INFER_BATCH, n)
            batch = torch.tensor(q[i:j], device=DEVICE)
            tt[i:j] = model.function.TravelTimes(batch).cpu().numpy()
    return tt.reshape(XX.shape)


def rotate_pts(points, x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    pts = np.array(points, dtype=np.float32)
    rot = np.array([[c, -s], [s, c]])
    return (rot @ pts.T).T + np.array([x, y])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_dir',      required=True,
                        help='Folder containing Model_Epoch_*.pt files')
    parser.add_argument('--start_epoch',  type=int, default=0)
    parser.add_argument('--end_epoch',    type=int, default=10**9)
    parser.add_argument('--n_samples',    type=int, default=None,
                        help='Number of evenly-sampled frames (default: all in range)')
    parser.add_argument('--output',       default='travel_field_evolution.mp4')
    parser.add_argument('--fps',          type=int, default=5)
    parser.add_argument('--x0',          type=float, default=0.0)
    parser.add_argument('--y0',          type=float, default=-0.03)
    parser.add_argument('--theta0',      type=float, default=np.pi / 2)
    parser.add_argument('--theta_vis',   type=float, default=np.pi / 2)
    parser.add_argument('--grid_n',      type=int,   default=100)
    parser.add_argument('--iso',         type=int,   default=12)
    parser.add_argument('--cmap',        default='plasma')
    parser.add_argument('--env',         default=DEFAULT_ENV)
    parser.add_argument('--fshape',      default=DEFAULT_FSHAPE)
    args = parser.parse_args()

    # ── Environment & shape ───────────────────────────────────────────────────
    env_pts = np.load(args.env)
    Fshape_pts = shape_to_points(dxf_to_shape(args.fshape), 0.25)

    xmin, xmax = env_pts[:, 0].min(), env_pts[:, 0].max()
    ymin, ymax = env_pts[:, 1].min(), env_pts[:, 1].max()
    print(f'Env bounds  x=[{xmin:.3f}, {xmax:.3f}]  y=[{ymin:.3f}, {ymax:.3f}]')

    xs = np.linspace(xmin, xmax, args.grid_n)
    ys = np.linspace(ymin, ymax, args.grid_n)
    XX, YY = np.meshgrid(xs, ys)

    # ── Select checkpoints ────────────────────────────────────────────────────
    all_pairs = collect_checkpoints(args.run_dir, args.start_epoch, args.end_epoch)
    if not all_pairs:
        print('No checkpoints found in the specified epoch range. Exiting.')
        return

    pairs = subsample(all_pairs, args.n_samples)
    print(f'Found {len(all_pairs)} checkpoint(s) in range '
          f'[{args.start_epoch}, {args.end_epoch}], '
          f'using {len(pairs)} frame(s).')
    for ep, p in pairs:
        print(f'  epoch {ep:6d}: {os.path.basename(p)}')

    # ── Infer all fields ──────────────────────────────────────────────────────
    model = md.Model(MODEL_PATH, DATA_PATH, DIM, [0.0] * 4, device=DEVICE)
    fields = []
    epochs = []
    for ep, pt_file in pairs:
        print(f'Inferring epoch {ep} …', end=' ', flush=True)
        model.load(pt_file)
        model.network.eval()
        TT = infer_field(model, args.x0, args.y0, args.theta0,
                         XX, YY, args.theta_vis)
        fields.append(TT)
        epochs.append(ep)
        print('done')

    global_min = min(f.min() for f in fields)
    global_max = max(f.max() for f in fields)
    print(f'Global travel-time range: [{global_min:.4f}, {global_max:.4f}]')

    # ── Build animation ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, 0.15, 0.03, 0.70])

    im = ax.imshow(
        fields[0],
        origin='lower',
        extent=[xmin, xmax, ymin, ymax],
        cmap=args.cmap,
        vmin=global_min, vmax=global_max,
        aspect='equal',
        alpha=0.85,
    )

    norm = Normalize(vmin=global_min, vmax=global_max)
    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label='Travel time  T(start → x, y)')

    # Static elements drawn once
    ax.scatter(env_pts[:, 0], env_pts[:, 1],
               c='lightgrey', s=2, zorder=3, rasterized=True)

    origin_rot = rotate_pts(Fshape_pts, args.x0, args.y0, args.theta0)
    if len(origin_rot) >= 3:
        from shapely.geometry import Polygon as SPoly
        op = SPoly(origin_rot)
        if op.is_valid:
            ax.add_patch(plt.Polygon(
                list(op.exterior.coords),
                facecolor='red', edgecolor='black',
                linewidth=0.8, alpha=1.0, zorder=10,
            ))
    ax.scatter([args.x0], [args.y0], c='red', s=40, zorder=11,
               label=f'origin ({args.x0:.2f},{args.y0:.2f},θ={args.theta0:.2f})')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    contour_holder = [None]
    title = ax.set_title('')

    def update(frame):
        TT = fields[frame]
        ep = epochs[frame]

        im.set_data(TT)

        if contour_holder[0] is not None:
            contour_holder[0].remove()
        contour_holder[0] = ax.contour(
            xs, ys, TT,
            levels=args.iso,
            colors='white', linewidths=0.6, alpha=0.75,
        )

        title.set_text(
            f'Epoch {ep}  |  dest θ = {args.theta_vis:.2f} rad'
        )
        return [im]

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(fields),
        interval=max(1, 1000 // args.fps),
        blit=False,
    )

    # Pick writer: prefer ffmpeg (mp4), fall back to pillow (gif)
    if animation.FFMpegWriter.isAvailable():
        writer = 'ffmpeg'
        out = args.output
    else:
        writer = 'pillow'
        out = os.path.splitext(args.output)[0] + '.gif'
        print('ffmpeg not found — saving as GIF instead.')

    print(f'Saving {out} at {args.fps} fps …')
    ani.save(out, writer=writer, fps=args.fps, dpi=130)
    print(f'Done → {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
