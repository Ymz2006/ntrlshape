"""Report the highest / lowest dtau/dconfig the trained network predicts.

Feeds input configuration pairs through the network and differentiates the
predicted travel time tau w.r.t. the input coordinates (the same `dtau` the
Eikonal loss uses).  Summarizes the range of:

  * |grad tau| per endpoint (full 6-D)         -> implied speed = 1/|grad|
  * |grad tau| per group: dist (xyz) / ang (rotvec), per endpoint
  * each individual coordinate's dtau          (signed)

This tells you the dynamic range of clearance-speed the field encodes, and
whether any direction has collapsed (|grad| -> 0, speed -> inf) or blown up
(|grad| huge, speed -> 0).

Run (from the nested ntrl-demo root, inside the pytorch docker):
    python inspect_dtau_range.py --dataPath testing_data/3dshape/rectangle_env1
    python inspect_dtau_range.py --dataPath datasets/3dshape/rectangle_env1  # training pairs
"""

import sys
sys.path.append('.')

import os
import argparse
from glob import glob

import numpy as np
import torch

from models.metric import model_train_metric as md

DIM = 6
HALF = 3   # dist = first 3 coords (xyz), ang = last 3 (rotvec), per endpoint


def load_model(model_path, data_path):
    model = md.Model(model_path, data_path, DIM, [0.0] * DIM, device='cuda')
    latest = os.path.join(model_path, 'latest.pt')
    pt = latest if os.path.exists(latest) else sorted(
        glob(os.path.join(model_path, '*', 'Model_Epoch_*.pt')))[-1]
    print('checkpoint:', pt)
    model.load(pt)
    model.network.eval()
    return model


def compute_dtau(model, pairs, batch=4096):
    """pairs: (N, 12) numpy -> dtau (N, 12) numpy (grad of tau wrt input coords)."""
    out = np.empty_like(pairs, dtype=np.float64)
    for s in range(0, len(pairs), batch):
        xp = torch.tensor(pairs[s:s + batch], dtype=torch.float32, device='cuda')
        # network.out detaches the input and returns a fresh grad-enabled leaf;
        # differentiate tau w.r.t. THAT (matches Function.gradient / the loss).
        tau, _w, coords = model.network.out(xp)
        g = torch.autograd.grad(tau.sum(), coords)[0]
        out[s:s + batch] = g.detach().cpu().numpy()
    return out


def _stats(name, v):
    """v: (N,) magnitudes -> printed min/median/max plus implied speed=1/v."""
    v = v[np.isfinite(v)]
    lo, md_, hi = v.min(), np.median(v), v.max()
    print("  %-18s |grad| min=%.4g  median=%.4g  max=%.4g   "
          "speed(1/|grad|) min=%.4g  max=%.4g"
          % (name, lo, md_, hi, 1.0 / hi, 1.0 / max(lo, 1e-12)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--n', type=int, default=0,
                   help='limit number of input pairs (0 = all)')
    args = p.parse_args()

    model = load_model(args.modelPath, args.dataPath)

    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    if args.n and args.n < len(pairs):
        pairs = pairs[:args.n]
    print('input pairs: %d  (from %s)\n' % (len(pairs), args.dataPath))

    dtau = compute_dtau(model, pairs)            # (N, 12)

    DT0, DT1 = dtau[:, :DIM], dtau[:, DIM:]
    groups = {
        'endpoint0 (full)': np.linalg.norm(DT0, axis=1),
        'endpoint1 (full)': np.linalg.norm(DT1, axis=1),
        'ep0 dist (xyz)':   np.linalg.norm(dtau[:, 0:HALF], axis=1),
        'ep0 ang (rotvec)': np.linalg.norm(dtau[:, HALF:DIM], axis=1),
        'ep1 dist (xyz)':   np.linalg.norm(dtau[:, DIM:DIM + HALF], axis=1),
        'ep1 ang (rotvec)': np.linalg.norm(dtau[:, DIM + HALF:], axis=1),
    }

    print("=" * 90)
    print("|grad tau| by group   (speed = 1/|grad|; small |grad| => fast/open, large => slow/tight)")
    print("=" * 90)
    for name, v in groups.items():
        _stats(name, v)

    print("\n" + "=" * 90)
    print("Per-coordinate dtau/dconfig  (signed; col layout: [x y z rx ry rz] x2 endpoints)")
    print("=" * 90)
    labels = ['x0', 'y0', 'z0', 'rx0', 'ry0', 'rz0',
              'x1', 'y1', 'z1', 'rx1', 'ry1', 'rz1']
    print("  %-5s %12s %12s %12s %12s" % ("coord", "min", "max", "mean|.|", "max|.|"))
    for j, lab in enumerate(labels):
        col = dtau[:, j]
        print("  %-5s %12.4g %12.4g %12.4g %12.4g"
              % (lab, col.min(), col.max(), np.abs(col).mean(), np.abs(col).max()))

    # Extremes: which input pair gives the largest / smallest full |grad tau|.
    full = np.concatenate([groups['endpoint0 (full)'], groups['endpoint1 (full)']])
    print("\n" + "=" * 90)
    print("Global extremes of full per-endpoint |grad tau| over %d endpoints:" % full.size)
    print("=" * 90)
    print("  lowest  |grad tau| = %.4g  -> highest predicted speed = %.4g"
          % (full.min(), 1.0 / max(full.min(), 1e-12)))
    print("  highest |grad tau| = %.4g  -> lowest  predicted speed = %.4g"
          % (full.max(), 1.0 / full.max()))


if __name__ == '__main__':
    main()
