"""Per-config 3-D scatter colored by (speed_angle - speed_dist).

Each sampled config (the x0 endpoint) is plotted at its (x, y, z) position and
colored by speed_angle - speed_dist:

    > 0 (red)   translation is the tighter constraint (angle is more open)
    < 0 (blue)  rotation is the tighter constraint (you can't turn here)

By default uses the WARPED speeds (what the model is trained on, exactly as in
model_train_metric.py); pass --raw to use the raw stored values.

Usage:
    python visualize_angle_minus_dist.py --dataPath datasets/3dshape/rectangle_env1
    python visualize_angle_minus_dist.py --dataPath testing_data/3dshape/rectangle_env1 --raw
"""

import os
import argparse

import numpy as np
import plotly.graph_objects as go

ANGLE_MAX = np.pi / 18
ANGLE_MIN = 0.0001
SPEED_FLOOR = 0.01


def warp_dist(sd):
    return np.clip(sd ** 2 * (2 - sd) ** 2, SPEED_FLOOR, None)


def warp_angle(sa):
    sa = np.clip(sa / ANGLE_MAX, ANGLE_MIN / ANGLE_MAX, 1.0)
    return np.clip(sa ** 2 * (2 - sa) ** 2, SPEED_FLOOR, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1',
                    help='Directory holding sampled_points/speed_dists/speed_angles .npy')
    ap.add_argument('--raw', action='store_true',
                    help='Use raw stored speeds instead of the training-warped ones.')
    ap.add_argument('--max_points', type=int, default=20000,
                    help='Subsample to at most this many configs for the HTML.')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pts = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    sd = np.load(os.path.join(args.dataPath, 'speed_dists.npy')).astype(np.float64)[:, 0]
    sa = np.load(os.path.join(args.dataPath, 'speed_angles.npy')).astype(np.float64)[:, 0]

    pos = pts[:, :3]                       # (x, y, z) of the x0 endpoint
    if not args.raw:
        sd, sa = warp_dist(sd), warp_angle(sa)
    diff = sa - sd

    n = pos.shape[0]
    if n > args.max_points:
        idx = np.random.default_rng(0).choice(n, args.max_points, replace=False)
        pos, diff = pos[idx], diff[idx]

    lim = float(np.quantile(np.abs(diff), 0.98))   # symmetric clip for color range
    kind = 'raw' if args.raw else 'warped'
    print(f'{n} configs ({len(diff)} plotted)  |  {kind}  '
          f'speed_angle-speed_dist: min {diff.min():.3f}  max {diff.max():.3f}  '
          f'mean {diff.mean():.3f}  frac(angle tighter)={np.mean(diff < 0):.1%}')

    fig = go.Figure(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='markers',
        marker=dict(size=2, color=diff, colorscale='RdBu', cmid=0,
                    cmin=-lim, cmax=lim, opacity=0.8,
                    colorbar=dict(title='angle - dist')),
        text=[f'{d:.3f}' for d in diff], hoverinfo='text'))
    fig.update_layout(
        title=f'speed_angle - speed_dist per config ({kind})  —  {os.path.basename(args.dataPath.rstrip("/"))}',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))

    out = args.out or os.path.join(args.dataPath, f'angle_minus_dist_{kind}.html')
    fig.write_html(out, include_plotlyjs='cdn')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
