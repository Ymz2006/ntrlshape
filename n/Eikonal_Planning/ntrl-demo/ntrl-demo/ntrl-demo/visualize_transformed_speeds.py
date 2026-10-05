"""Visualize the speed_dist / speed_angle distributions before and after the
training-time transform.

Both signals are warped inside ``models/metric/model_train_metric.py`` right
before they are handed to ``Function.Loss`` -- the warped values are what the
model actually sees as ``speed``.  This script reproduces those exact transforms
(with ``alpha = 1``, so the ``alpha*x + 1 - alpha`` lines are no-ops) and writes
a 2x2 HTML histogram:

    +-----------------------------+-----------------------------+
    | raw speed_dist              | transformed speed_dist      |
    +-----------------------------+-----------------------------+
    | raw speed_angle             | transformed speed_angle     |
    +-----------------------------+-----------------------------+

Usage:

    python visualize_transformed_speeds.py \
        --speed_dist  testing_data/3dshape/rectangle_env1/speed_dists.npy \
        --speed_angle testing_data/3dshape/rectangle_env1/speed_angles.npy

            python visualize_transformed_speeds.py \
        --speed_dist  datasets/3dshape/Lshape3d_env1/speed_dists.npy \
        --speed_angle datasets/3dshape/Lshape3d_env1/speed_angles.npy
    python visualize_transformed_speeds.py            # uses the default paths
"""

import os
import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Matches model_train_metric.py
ANGLE_MAX = np.pi / 18
ANGLE_MIN = 0.0001
SPEED_FLOOR = 0.01


def transform_speed_dist(sd):
    """Exact training warp for speed_dist (model_train_metric.py)."""
    sd = sd ** 2 * (2 - sd) ** 2
    sd = np.clip(sd, SPEED_FLOOR, None)
    return sd


def transform_speed_angle(sa):
    """Exact training warp for speed_angle (model_train_metric.py).

    speed_angle is stored as angle/pi; training first rescales by 1/(pi/18) and
    clamps to [angle_min/angle_max, 1], then applies the same polynomial warp as
    speed_dist, then floors at 0.01.
    """
    sa = np.clip(sa / ANGLE_MAX, ANGLE_MIN / ANGLE_MAX, 1.0)
    sa = sa ** 2 * (2 - sa) ** 2
    sa = np.clip(sa, SPEED_FLOOR, None)
    return sa


def _describe(name, arr):
    print(f'  {name:24s}: n={arr.size}  min={arr.min():.4f}  max={arr.max():.4f}  '
          f'mean={arr.mean():.4f}  frac@floor(<=0.0101)={(arr <= 0.0101).mean():.1%}')


def main():
    parser = argparse.ArgumentParser(
        description='Histogram speed_dist / speed_angle before and after the '
                    'training-time warp that is fed into the model.')
    parser.add_argument(
        '--speed_dist',
        default='./datasets/3dshape/rectangle_env1/speed_dists.npy',
        help='Path to the speed_dists .npy file.')
    parser.add_argument(
        '--speed_angle',
        default='./datasets/3dshape/rectangle_env1/speed_angles.npy',
        help='Path to the speed_angles .npy file (stored as angle/pi).')
    parser.add_argument(
        '--out', default=None,
        help='Output HTML path (default: <speed_dist dir>/transformed_speeds.html).')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins.')
    args = parser.parse_args()

    sd_raw = np.load(args.speed_dist).astype(np.float64).reshape(-1)
    sa_raw = np.load(args.speed_angle).astype(np.float64).reshape(-1)
    
    #sa_raw = np.sin(sa_raw*np.pi)
    sd_t = transform_speed_dist(sd_raw)
    sa_t = np.sin(transform_speed_angle(sa_raw))

    print(f'speed_dist : {args.speed_dist}')
    _describe('raw speed_dist', sd_raw)
    _describe('transformed speed_dist', sd_t)
    print(f'speed_angle: {args.speed_angle}')
    _describe('raw speed_angle', sa_raw)
    _describe('transformed speed_angle', sa_t)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Raw speed_dist (n={sd_raw.size})',
            'Transformed speed_dist (fed to model)',
            f'Raw speed_angle / pi (n={sa_raw.size})',
            'Transformed speed_angle (fed to model)'))

    panels = [
        (1, 1, sd_raw, 'steelblue', 'speed_dist (raw)'),
        (1, 2, sd_t, 'indianred', 'warped speed_dist'),
        (2, 1, sa_raw, 'seagreen', 'speed_angle (raw, angle/pi)'),
        (2, 2, sa_t, 'darkorange', 'warped speed_angle'),
    ]
    for row, col, data, color, xlabel in panels:
        fig.add_trace(go.Histogram(
            x=data, nbinsx=args.bins,
            marker=dict(color=color, line=dict(color='black', width=1)),
            opacity=0.75, showlegend=False), row=row, col=col)
        fig.update_xaxes(title_text=xlabel, row=row, col=col)
        fig.update_yaxes(title_text='count', row=row, col=col)

    fig.update_layout(
        title_text=f'Transformed speeds  —  {os.path.basename(os.path.dirname(args.speed_dist) or ".")}',
        bargap=0.05)

    out = args.out or os.path.join(os.path.dirname(args.speed_dist) or '.',
                                   'transformed_speeds.html')
    fig.write_html(out, include_plotlyjs='cdn')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
