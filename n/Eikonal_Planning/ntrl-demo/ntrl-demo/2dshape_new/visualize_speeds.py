"""
visualize_speeds.py

Visualise the speed (clearance) values stored in a training-data directory.

The data directory is expected to contain:
    sampled_points.npy   (N, 6)  [x0, y0, theta0, x1, y1, theta1]
    speed.npy            (N, 2)  [speed0, speed1]
    env.npy  or  Fmaze3env.npy   (M, 2)  env boundary points

Visualisation modes (--mode)
-----------------------------
scatter  – scatter of (x, y) positions coloured by speed  [default]
shapes   – F-shapes drawn at each position, coloured by speed
hist     – histogram of speed values
theta    – scatter of (x, y) coloured by theta value
map      – continuous heatmap interpolated from scattered (x, y, speed) data
           (theta is ignored; works like the travel-field visualisation)

Map mode – two ways to supply data
-----------------------------------
1. From a training-data directory (theta ignored, index selects start or end):
       python visualize_speeds.py --mode map --data ./training_data2d/Fshape_FmazeEasy
       python visualize_speeds.py --mode map --data ./training_data2d/Fshape_FmazeEasy --index 1

2. From raw numpy files (positions: (N,2) xy array; speeds: (N,) array):
       python visualize_speeds.py --mode map --pts_npy pos.npy --spd_npy spd.npy

Usage
-----
    python visualize_speeds.py --data ./training_data2d/Fshape_FmazeEasy
    python visualize_speeds.py --data ./training_data2d/Fshape_FmazeEasy \\
        --mode shapes --shape_dxf ./datasets/Fshape_norm.dxf
    python visualize_speeds.py --data ./training_data2d/Fshape_FmazeEasy \\
        --mode scatter --index both --save
    python visualize_speeds.py --mode map --data ./training_data2d/Fshape_FmazeEasy \\
        --grid_n 200 --iso 12 --cmap plasma --save
"""

import sys, os, argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from shapely.geometry import Polygon
import torch

sys.path.append(".")
from parse_shape import dxf_to_shape, shape_to_points


# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA   = "./training_data2d/Fshape_FmazeEasy"
DEFAULT_SHAPE  = "./datasets/Fshape_norm.dxf"
DEFAULT_CMAP   = "viridis"


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _load_env(data_path):
    for name in ("env.npy", "Fmaze3env.npy", "Easyenv.npy"):
        p = os.path.join(data_path, name)
        if os.path.exists(p):
            return np.load(p)
    # fall back: search for any *env*.npy
    hits = glob(os.path.join(data_path, "*env*.npy"))
    if hits:
        return np.load(hits[0])
    return np.zeros((0, 2))


def _rotate_pts(shape_points, x, y, theta):
    rot = torch.tensor([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0, 0, 1],
    ], dtype=torch.float32)
    sp  = torch.tensor(shape_points, dtype=torch.float32)
    sp  = torch.cat([sp, torch.ones(sp.shape[0], 1)], dim=1)
    out = (rot @ sp.T).T
    return [tuple(p[:2].numpy()) for p in out]


# ─── Visualisation functions ───────────────────────────────────────────────────

def plot_scatter(ax, x, y, speed, env_pts, cmap, vmin, vmax, title, label):
    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c="black", s=3, zorder=3, rasterized=True, label="env")
    sc = ax.scatter(x, y, c=speed, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=8, alpha=0.6, zorder=4, rasterized=True)
    plt.colorbar(sc, ax=ax, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)


def plot_shapes(ax, positions, theta, speed, shape_points,
                env_pts, cmap, vmin, vmax, title, max_cnt):
    norm  = Normalize(vmin=vmin, vmax=vmax)
    cmap_ = get_cmap(cmap)

    n = min(max_cnt, len(positions))
    idx = np.random.choice(len(positions), n, replace=False) if len(positions) > n \
          else np.arange(len(positions))

    for i in idx:
        rotated = _rotate_pts(shape_points, float(positions[i, 0]),
                               float(positions[i, 1]), float(theta[i]))
        if len(rotated) < 3:
            continue
        poly = Polygon(rotated)
        if not poly.is_valid:
            continue
        color = cmap_(norm(float(speed[i])))
        ax.add_patch(plt.Polygon(
            list(poly.exterior.coords),
            facecolor=color, edgecolor="black",
            linewidth=0.3, alpha=0.75, zorder=4,
        ))

    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c="grey", s=4, zorder=3, rasterized=True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Speed")
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)


def plot_hist(axes, s0, s1, index):
    if index in ("0", "both"):
        axes[0].hist(s0, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
        axes[0].set_xlabel("Speed"); axes[0].set_ylabel("Count")
        axes[0].set_title("Speed distribution – start configs")
    if index in ("1", "both"):
        axes[-1].hist(s1, bins=60, color="salmon", edgecolor="none", alpha=0.8)
        axes[-1].set_xlabel("Speed"); axes[-1].set_ylabel("Count")
        axes[-1].set_title("Speed distribution – end configs")


def plot_theta(ax, x, y, theta_raw, env_pts, cmap, title):
    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c="black", s=3, zorder=3, rasterized=True)
    sc = ax.scatter(x, y, c=theta_raw, cmap=cmap, s=8, alpha=0.6, zorder=4,
                    rasterized=True)
    plt.colorbar(sc, ax=ax, label="θ (rad)")
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)


def plot_map(ax, x, y, speed, env_pts, cmap, vmin, vmax, title,
             grid_n=200, iso_levels=12):
    """Continuous heatmap interpolated from scattered (x, y, speed) points.

    Theta is ignored — only x and y positions are used.  Uses linear
    interpolation on a regular grid with nearest-neighbour fill at the edges,
    plus isochrone-style contour lines, matching the style of
    visualize_travel_field.py.
    """
    # Grid bounds: prefer env extent so we cover the full environment
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

    # Linear interpolation then fill edges with nearest-neighbour
    ZZ = griddata(pts, speed, (XX, YY), method="linear")
    ZZ_nn = griddata(pts, speed, (XX, YY), method="nearest")
    nan_mask = np.isnan(ZZ)
    ZZ[nan_mask] = ZZ_nn[nan_mask]

    im = ax.imshow(
        ZZ,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        aspect="equal",
        alpha=0.85,
    )

    # Isochrone contours
    cs = ax.contour(xs, ys, ZZ, levels=iso_levels,
                    colors="white", linewidths=0.7, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.3f")

    # Env boundary
    if env_pts.shape[0] > 0:
        ax.scatter(env_pts[:, 0], env_pts[:, 1],
                   c="black", s=4, zorder=3, rasterized=True)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Speed")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data",      default=DEFAULT_DATA,
                        help="Directory with sampled_points.npy and speed.npy")
    parser.add_argument("--shape_dxf", default=DEFAULT_SHAPE,
                        help="Robot shape DXF (needed for --mode shapes)")
    parser.add_argument("--mode",      default="scatter",
                        choices=["scatter", "shapes", "hist", "theta", "map"],
                        help="Visualisation mode")
    parser.add_argument("--index",     default="0",
                        choices=["0", "1", "both"],
                        help="Which endpoint to show: 0=start, 1=end, both=side-by-side")
    parser.add_argument("--cnt",       type=int, default=1000,
                        help="Max shapes to draw (shapes mode) or points to scatter")
    parser.add_argument("--cmap",      default=DEFAULT_CMAP)
    parser.add_argument("--vmin",      type=float, default=None)
    parser.add_argument("--vmax",      type=float, default=None)
    parser.add_argument("--raw_theta", action="store_true",
                        help="Theta is stored in raw radians (not divided by 2π)")
    # map-mode args
    parser.add_argument("--pts_npy",   default=None,
                        help="map mode: (N,2) npy of [x,y] positions (overrides --data)")
    parser.add_argument("--spd_npy",   default=None,
                        help="map mode: (N,) npy of speed values (paired with --pts_npy)")
    parser.add_argument("--env_npy",   default=None,
                        help="map mode: (M,2) npy of env boundary points (optional)")
    parser.add_argument("--grid_n",    type=int, default=200,
                        help="map mode: grid resolution (default 200)")
    parser.add_argument("--iso",       type=int, default=12,
                        help="map mode: number of isochrone contour levels (default 12)")
    parser.add_argument("--save",      action="store_true",
                        help="Save figure to disk instead of showing")
    parser.add_argument("--out",       default=None,
                        help="Output filename (default: auto-named)")
    args = parser.parse_args()

    # ── Map mode with raw npy files ───────────────────────────────────────────
    if args.mode == "map" and args.pts_npy is not None:
        if args.spd_npy is None:
            print("ERROR: --spd_npy is required when --pts_npy is given")
            sys.exit(1)
        xy  = np.load(args.pts_npy)          # (N, 2)
        spd = np.load(args.spd_npy).ravel()  # (N,)
        env = np.load(args.env_npy) if args.env_npy else np.zeros((0, 2))
        if env.ndim == 1:
            env = env.reshape(-1, 2)

        vmin = args.vmin if args.vmin is not None else float(spd.min())
        vmax = args.vmax if args.vmax is not None else float(spd.max())
        N    = len(xy)
        print(f"Loaded {N} points from {args.pts_npy}")
        print(f"speed range: [{spd.min():.4f}, {spd.max():.4f}]")

        fig, ax = plt.subplots(figsize=(8, 7))
        plot_map(ax, xy[:, 0], xy[:, 1], spd, env,
                 args.cmap, vmin, vmax,
                 f"Speed map – {os.path.basename(args.pts_npy)}",
                 grid_n=args.grid_n, iso_levels=args.iso)
        fig.tight_layout()

        if args.save:
            out = args.out or "speeds_map.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved {out}")
        else:
            plt.show()
        return

    # ── Load training-data directory ──────────────────────────────────────────
    pts_path = os.path.join(args.data, "sampled_points.npy")
    spd_path = os.path.join(args.data, "speed.npy")
    if not os.path.exists(pts_path) or not os.path.exists(spd_path):
        print(f"ERROR: could not find sampled_points.npy / speed.npy in {args.data}")
        sys.exit(1)

    pts   = np.load(pts_path)   # (N, 6)
    speed = np.load(spd_path)   # (N, 2)
    env   = _load_env(args.data)

    N = len(pts)
    print(f"Loaded {N} samples from {args.data}")
    print(f"  points shape: {pts.shape}  speed shape: {speed.shape}")
    print(f"  speed range:  [{speed.min():.4f}, {speed.max():.4f}]")

    # Parse columns
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = pts[:, 3], pts[:, 4]

    # Theta: stored as theta/(2π) by default; multiply back to get radians
    scale = 1.0 if args.raw_theta else (2 * np.pi)
    t0 = pts[:, 2] * scale
    t1 = pts[:, 5] * scale

    s0, s1 = speed[:, 0], speed[:, 1]

    vmin = args.vmin if args.vmin is not None else float(speed.min())
    vmax = args.vmax if args.vmax is not None else float(speed.max())

    # Subsample for scatter/shapes to keep rendering fast
    if N > args.cnt and args.mode in ("scatter", "shapes", "theta"):
        sel = np.random.choice(N, args.cnt, replace=False)
    else:
        sel = np.arange(N)

    # ── Build figure ──────────────────────────────────────────────────────────
    ncols = 2 if args.index == "both" else 1
    figw  = 9 * ncols if args.mode != "hist" else 7 * ncols

    if args.mode == "hist":
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 5), squeeze=False)
        axes = axes[0]
        plot_hist(axes, s0, s1, args.index)

    elif args.mode == "scatter":
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 7), squeeze=False)
        if args.index in ("0", "both"):
            plot_scatter(axes[0, 0],
                         x0[sel], y0[sel], s0[sel], env,
                         args.cmap, vmin, vmax,
                         "Start configs – speed", "speed")
        if args.index in ("1", "both"):
            plot_scatter(axes[0, -1],
                         x1[sel], y1[sel], s1[sel], env,
                         args.cmap, vmin, vmax,
                         "End configs – speed", "speed")

    elif args.mode == "shapes":
        if not os.path.exists(args.shape_dxf):
            print(f"ERROR: shape DXF not found: {args.shape_dxf}")
            sys.exit(1)
        shape_poly   = dxf_to_shape(args.shape_dxf)
        shape_points = list(shape_poly.exterior.coords)

        fig, axes = plt.subplots(1, ncols, figsize=(figw, 7), squeeze=False)
        if args.index in ("0", "both"):
            ax = axes[0, 0]
            xy0 = np.stack([x0[sel], y0[sel]], axis=1)
            plot_shapes(ax, xy0, t0[sel], s0[sel],
                        shape_points, env,
                        args.cmap, vmin, vmax,
                        "Start configs – speed", args.cnt)
        if args.index in ("1", "both"):
            ax = axes[0, -1]
            xy1 = np.stack([x1[sel], y1[sel]], axis=1)
            plot_shapes(ax, xy1, t1[sel], s1[sel],
                        shape_points, env,
                        args.cmap, vmin, vmax,
                        "End configs – speed", args.cnt)

    elif args.mode == "theta":
        fig, axes = plt.subplots(1, ncols, figsize=(figw, 7), squeeze=False)
        if args.index in ("0", "both"):
            plot_theta(axes[0, 0],
                       x0[sel], y0[sel], t0[sel], env,
                       args.cmap, "Start configs – θ (rad)")
        if args.index in ("1", "both"):
            plot_theta(axes[0, -1],
                       x1[sel], y1[sel], t1[sel], env,
                       args.cmap, "End configs – θ (rad)")

    elif args.mode == "map":
        fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 7), squeeze=False)
        if args.index in ("0", "both"):
            plot_map(axes[0, 0], x0, y0, s0, env,
                     args.cmap, vmin, vmax,
                     "Start configs – speed map",
                     grid_n=args.grid_n, iso_levels=args.iso)
        if args.index in ("1", "both"):
            plot_map(axes[0, -1], x1, y1, s1, env,
                     args.cmap, vmin, vmax,
                     "End configs – speed map",
                     grid_n=args.grid_n, iso_levels=args.iso)

    fig.suptitle(
        f"Training data: {os.path.basename(args.data)}  "
        f"(N={N}, mode={args.mode}, index={args.index})",
        fontsize=11,
    )
    fig.tight_layout()

    if args.save:
        out = args.out or f"speeds_{args.mode}_{args.index}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
