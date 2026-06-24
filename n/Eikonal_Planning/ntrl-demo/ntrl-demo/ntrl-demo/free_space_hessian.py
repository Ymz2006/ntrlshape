"""Distribution of the second derivative d2tau/dq2 in FREE SPACE.

The training/inspection data only has x0 in the narrow clearance band, so it
can't tell us the Hessian where the robot is far from any obstacle.  This script
samples genuinely free configs -- collision-free AND clearance > margin (the
region where the target speed saturates at 1) -- pairs each with a random free
goal, and reports the distribution of the per-point Hessian spectral norm.

Geometry setup mirrors dataprocessing/preprocess_obj.main (same normalization,
tetrahedralization, uniform SE(3) sampling, collision + clearance query).
Hessian + spectral-norm helpers are reused from inspect_d2tau_range.

Run:  python free_space_hessian.py --dataPath datasets/3dshape/rectangle_env1 \
          --n_free 50000
"""
import sys; sys.path.append('.')
import os, json, argparse
import numpy as np
import torch

from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import (
    load_obj, tetrahedralize_shape, sample_surface_points,
    generate_radius_surface_points, evaluate_placements, _sample_configs,
    DEFAULT_ENV_POINTS,
)
from inspect_d2tau_range import load_model, compute_hessian, _sym_eig_absmax, DIM, HALF


def build_geometry(meta, device):
    """Reproduce preprocess_obj.main's normalized env+shape geometry."""
    V_env, F_env, names_env = load_obj(meta['env_obj'])
    bb_min, bb_max = V_env.min(0), V_env.max(0)
    center = 0.5 * (bb_min + bb_max)
    scale = float((bb_max - bb_min).max())
    V_env_n = (V_env - center) / scale
    sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
    env_points = sample_surface_points(V_env_n, F_env[sample_mask], DEFAULT_ENV_POINTS).astype(np.float32)
    half_extent = (bb_max - bb_min) / scale * 0.5 - 0.01

    V_sh, F_sh, _ = load_obj(meta['shape_obj'])
    sh_c = 0.5 * (V_sh.min(0) + V_sh.max(0))
    V_sh_local = (V_sh - sh_c) / scale * float(meta['shape_scale'])
    TV, TT, TF = tetrahedralize_shape(V_sh_local, F_sh)
    tet = torch.tensor(TV[TT], dtype=torch.float32)
    face = torch.tensor(TV[TF], dtype=torch.float32)
    rad_pts, rad_bins = generate_radius_surface_points(V_sh_local, F_sh, 1000, 10)
    return dict(env_points=torch.tensor(env_points), tet=tet, face=face,
                rad_pts=rad_pts, rad_bins=rad_bins, half=half_extent, device=device)


def sample_free(geo, n_target, min_clear, batch=800, max_iter=4000):
    """Uniformly sample SE(3) configs, keep collision-free with clearance>min_clear.
    Returns (M,6) raw configs (rotvec in radians) and their clearance (M,)."""
    hx, hy, hz = [float(v) for v in geo['half']]
    keep_cfg, keep_d = [], []
    got = 0
    for _ in range(max_iter):
        cfg = _sample_configs(batch, hx, hy, hz)
        is_free, dist, *_ = evaluate_placements(
            cfg, geo['tet'], geo['face'], geo['env_points'],
            geo['rad_pts'], geo['rad_bins'], geo['device'])
        m = is_free & (dist > min_clear)
        if m.any():
            keep_cfg.append(cfg[m]); keep_d.append(dist[m]); got += int(m.sum())
        if got >= n_target:
            break
    cfg = torch.cat(keep_cfg)[:n_target].numpy()
    d = torch.cat(keep_d)[:n_target].numpy()
    return cfg, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--n_free', type=int, default=50000)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--sample_batch', type=int, default=800,
                   help='configs per collision/clearance eval (lower if OOM)')
    p.add_argument('--transform', choices=['sq', 'sqrt'], default='sq')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    with open(os.path.join(args.dataPath, 'meta.json')) as f:
        meta = json.load(f)
    margin = float(meta['margin'])
    device = 'cuda'

    model = load_model(args.modelPath, args.dataPath)
    geo = build_geometry(meta, device)

    print('sampling free-space configs (collision-free & clearance > margin=%.3g)...' % margin)
    q_cfg, q_clear = sample_free(geo, args.n_free, min_clear=margin, batch=args.sample_batch)
    g_cfg, _ = sample_free(geo, len(q_cfg), min_clear=margin, batch=args.sample_batch)   # free goals
    print('free-space query configs: %d   clearance: median=%.3g max=%.3g (margin=%.3g)\n'
          % (len(q_cfg), np.median(q_clear), q_clear.max(), margin))

    # Build (x0, x1) pairs with rotvec normalized /2pi (training convention).
    q = q_cfg.copy(); q[:, 3:6] /= (2 * np.pi)
    g = g_cfg.copy(); g[:, 3:6] /= (2 * np.pi)
    pairs = np.concatenate([q, g], axis=1).astype(np.float64)   # (M, 12)

    H = compute_hessian(model, pairs, batch=args.batch, transform=args.transform)
    spec = _sym_eig_absmax(H)                       # (M,) full 6x6 spectral norm
    spec_t = _sym_eig_absmax(H[:, :HALF, :HALF])    # translation block
    spec_r = _sym_eig_absmax(H[:, HALF:, HALF:])    # rotation block

    def dist(name, v):
        qs = np.percentile(v, [50, 75, 90, 99])
        print("  %-20s median=%.4g  p75=%.4g  p90=%.4g  p99=%.4g  max=%.4g  mean=%.4g"
              % (name, qs[0], qs[1], qs[2], qs[3], v.max(), v.mean()))

    print("=" * 92)
    print("FREE-SPACE Hessian spectral norm distribution  (N=%d, transform=%s)"
          % (len(spec), args.transform))
    print("=" * 92)
    dist('full (6x6)', spec)
    dist('translation (3x3)', spec_t)
    dist('rotation (3x3)', spec_r)

    # Trend deeper into free space: bucket by clearance / margin.
    print("\n  spectral norm (full) vs clearance/margin (deeper = more open):")
    rel = q_clear / margin
    edges = [1, 1.5, 2, 3, 5, 1e9]
    print("    %-14s %8s %10s %10s %10s" % ("clr/margin", "count", "median", "p90", "max"))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rel >= lo) & (rel < hi)
        if not m.any():
            continue
        s = spec[m]
        tag = '[%.1f,%.1f)' % (lo, hi) if hi < 1e8 else '[%.1f, inf)' % lo
        print("    %-14s %8d %10.4g %10.4g %10.4g" % (tag, m.sum(), np.median(s), np.percentile(s, 90), s.max()))

    print("\n  reference: clearance-band medians were ~86 (tight) .. ~42 (band edge).")


if __name__ == '__main__':
    main()
