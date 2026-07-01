"""Sample ~100k collision-free SE(3) configs and report each one's RAW closest
distance (clearance) and RAW closest angle, then graph them.

Reuses the exact geometry pipeline from dataprocessing/preprocess_obj.py
(evaluate_placements returns is_free, dist, min_angle, normal).  No narrow-band
filtering -- we keep every collision-free config so the full raw distribution is
visible.  Writes a scatter (dist vs angle) with marginal histograms into the
dataset dir.

Usage:
    python test_sample_dist_angle.py --out datasets/3dshape/rectangle_env1 \
        --num_samples 100000 --device cpu
"""
import sys
sys.path.append('.')
import os
import json
import time
import argparse

import numpy as np
import torch

import dataprocessing.preprocess_obj as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='datasets/3dshape/rectangle_env1',
                    help='Dataset dir (must contain meta.json); graph is written here.')
    ap.add_argument('--num_samples', type=int, default=100000)
    ap.add_argument('--batch_size', type=int, default=2000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--num_env_points', type=int, default=P.DEFAULT_ENV_POINTS)
    ap.add_argument('--num_radius_points', type=int, default=1000)
    ap.add_argument('--radius_bins', type=int, default=10)
    ap.add_argument('--tet_switches', default=P.DEFAULT_TET_SWITCHES)
    ap.add_argument('--plot_points', type=int, default=20000,
                    help='Max points drawn in the scatter (stats use all).')
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.out, 'meta.json')))
    env_obj, shape_obj, shape_scale = meta['env_obj'], meta['shape_obj'], meta['shape_scale']

    # ── Environment setup (mirrors preprocess_obj.main) ──
    V_env, F_env, names_env = P.load_obj(env_obj)
    bb_min, bb_max = V_env.min(0), V_env.max(0)
    center_env = 0.5 * (bb_min + bb_max)
    scale = float((bb_max - bb_min).max())
    V_env_n = (V_env - center_env) / scale
    sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
    env_points = P.sample_surface_points(V_env_n, F_env[sample_mask],
                                         args.num_env_points).astype(np.float32)
    half_extent = (bb_max - bb_min) / scale * 0.5 - 0.01

    # ── Shape setup ──
    V_sh, F_sh, _ = P.load_obj(shape_obj)
    shape_center = 0.5 * (V_sh.min(0) + V_sh.max(0))
    V_sh_local = (V_sh - shape_center) / scale * shape_scale
    TV, TT, TF = P.tetrahedralize_shape(V_sh_local, F_sh, switches=args.tet_switches)
    tet_verts_local = torch.tensor(TV[TT], dtype=torch.float32)
    face_verts_local = torch.tensor(TV[TF], dtype=torch.float32)
    rad_points, rad_bins = P.generate_radius_surface_points(
        V_sh_local, F_sh, args.num_radius_points, args.radius_bins)

    dev = args.device
    env_t = torch.tensor(env_points).to(dev)
    tet_verts_local, face_verts_local = tet_verts_local.to(dev), face_verts_local.to(dev)
    rad_points, rad_bins = rad_points.to(dev), rad_bins.to(dev)
    hx, hy, hz = (float(half_extent[0]), float(half_extent[1]), float(half_extent[2]))

    # ── Sample collision-free configs, record raw dist + raw angle ──
    dists, angles = [], []
    n = 0
    t0 = time.time()
    while n < args.num_samples:
        cfg = P._sample_configs(args.batch_size, hx, hy, hz)
        free, d, a, *_ = P.evaluate_placements(
            cfg, tet_verts_local, face_verts_local, env_t, rad_points, rad_bins, dev)
        dists.append(d[free].numpy())
        angles.append(a[free].numpy())
        n += int(free.sum())
        print(f'  collected {n}/{args.num_samples}', flush=True)
    dist = np.concatenate(dists)[:args.num_samples]
    angle = np.concatenate(angles)[:args.num_samples]
    print(f'done in {time.time()-t0:.1f}s')

    def desc(name, x):
        q = np.quantile(x, [0, .05, .5, .95, 1])
        print(f'  {name:18s}: min {q[0]:.4f}  p5 {q[1]:.4f}  med {q[2]:.4f}  '
              f'p95 {q[3]:.4f}  max {q[4]:.4f}  mean {x.mean():.4f}')
    print(f'RAW signals over {len(dist)} collision-free configs:')
    desc('closest distance', dist)
    desc('closest angle (rad)', angle)

    # ── Graph: dist vs angle scatter with marginal histograms ──
    if len(dist) > args.plot_points:
        idx = np.random.default_rng(0).choice(len(dist), args.plot_points, replace=False)
        ds, as_ = dist[idx], angle[idx]
    else:
        ds, as_ = dist, angle

    out_html = os.path.join(args.out, 'raw_dist_vs_angle.html')
    try:
        import plotly.express as px
        fig = px.scatter(
            x=ds, y=as_, marginal_x='histogram', marginal_y='histogram',
            opacity=0.4, labels={'x': 'closest distance (clearance)',
                                 'y': 'closest angle (rad)'},
            title=f'Raw closest distance vs closest angle  '
                  f'({len(dist)} collision-free configs, {os.path.basename(args.out.rstrip("/"))})')
        fig.update_traces(marker=dict(size=3), selector=dict(type='scatter'))
    except Exception:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=ds, y=as_, mode='markers',
                                   marker=dict(size=3, opacity=0.4)))
        fig.update_layout(xaxis_title='closest distance (clearance)',
                          yaxis_title='closest angle (rad)')
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f'wrote {out_html}')


if __name__ == '__main__':
    main()
