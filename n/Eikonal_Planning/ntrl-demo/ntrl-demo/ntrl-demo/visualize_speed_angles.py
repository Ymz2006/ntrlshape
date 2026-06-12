"""Visualize the speed-angle distribution before/after the training transform.

Loads an ``.npy`` file of angle values (default ``speed_angles.npy``, stored as
angle/pi in [0, 1] -- exactly what ``dataprocessing/preprocess_obj.py`` writes)
and applies the same per-value transform used in training
(``models/metric/model_train_metric.py``):

    speed_angle = torch.clamp(speed_angle * torch.pi, min=0, max=torch.pi/18) / (torch.pi/18)

i.e. recover radians, cut at the max angle pi/18 (10 deg), then renormalize to
[0, 1].  Writes a standalone HTML histogram (raw vs transformed) for inspection.

Usage:

    python visualize_speed_angles.py --npy testing_data/3dshape/rectangle_env1/speed_angles.npy
    python visualize_speed_angles.py                       # uses the default path above
"""

import os
import argparse

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MAX_ANGLE = 2 * torch.pi / 18  


def transform(angles):
    """Apply the exact training transform to a tensor of angle/pi values."""
    return torch.clamp(angles * torch.pi, min=0, max=MAX_ANGLE) / MAX_ANGLE


def main():
    parser = argparse.ArgumentParser(
        description='Histogram the speed-angle distribution before/after the '
                    'training clamp+renormalize transform.')
    parser.add_argument(
        '--npy',
        default='./datasets/3dshape/rectangle_env1/speed_angles.npy',
        help='Path to the .npy file of angle values (stored as angle/pi in [0,1]).')
    parser.add_argument(
        '--out', default=None,
        help='Output HTML path (default: <npy dir>/speed_angles_transformed.html).')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins.')
    args = parser.parse_args()

    raw = np.load(args.npy).astype(np.float64).reshape(-1)          # flatten all values
    raw_t = torch.from_numpy(raw)
    transformed = transform(raw_t).numpy()

    n = raw.size
    saturated = float((transformed >= 0.999).mean())               # fraction hitting the clamp
    print(f'loaded {args.npy}  ({n} values)')
    print(f'  raw         : min {raw.min():.4f}  max {raw.max():.4f}  mean {raw.mean():.4f}')
    print(f'  transformed : min {transformed.min():.4f}  max {transformed.max():.4f}  '
          f'mean {transformed.mean():.4f}')
    print(f'  fraction clamped to 1.0 (angle >= 10 deg): {saturated:.1%}')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Raw (angle/pi, n={n})',
                        'Transformed: clamp(angle, 10 deg) / 10 deg'))
    fig.add_trace(go.Histogram(
        x=raw, nbinsx=args.bins, marker=dict(color='steelblue',
        line=dict(color='black', width=1)), opacity=0.75, name='raw'),
        row=1, col=1)
    fig.add_trace(go.Histogram(
        x=transformed, nbinsx=args.bins, marker=dict(color='indianred',
        line=dict(color='black', width=1)), opacity=0.75, name='transformed'),
        row=1, col=2)
    fig.update_xaxes(title_text='angle / pi', row=1, col=1)
    fig.update_xaxes(title_text='normalized speed_angle', row=1, col=2)
    fig.update_yaxes(title_text='count', row=1, col=1)
    fig.update_yaxes(title_text='count', row=1, col=2)
    fig.update_layout(
        title_text=f'Speed-angle distribution  —  {os.path.basename(args.npy)}',
        bargap=0.05, showlegend=False)

    out = args.out or os.path.join(os.path.dirname(args.npy) or '.',
                                   'speed_angles_transformed.html')
    fig.write_html(out, include_plotlyjs='cdn')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
