"""Visual sanity-check for the ``trans_n`` / ``rot_n`` fields.

For each sampled placement this draws three copies of the shape mesh:

    * the original placement                                   (grey)
    * the placement nudged by  ``trans_n * trans_eps``         (green)
        -- trans_n is a pure-translation unit direction that should point the
           shape *towards* the nearest obstacle (reduces clearance distance).
    * the placement rotated by  ``rot_n * rot_eps``            (red)
        -- rot_n is a pure-rotation unit axis (applied about the placement
           centre in the world frame) that should swing the shape so as to
           *reduce the rotational clearance* (the min shape-dir / env-dir angle).

It mirrors the look of ``sampled_placements.html`` (grey env points, translucent
walls) but adds a plotly slider so you can step through placements one at a time
-- overlaying every triple at once is unreadable.

Inputs are read straight from a preprocess ``--out`` directory:
    sampled_points.npy (N,12)  trans_n.npy (N,12)  rot_n.npy (N,12)  env.npy
    meta.json  (shape_obj / env_obj / env_scale / env_center / shape_scale / rot_norm)

Usage (run from the nested ntrl-demo root, inside the docker container):
    python visualize_trans_rot.py --dataPath testing_data/3dshape/Lshape3d_env1 \
        --cnt 30 --trans_eps 0.05 --rot_eps 0.3
"""

import sys
sys.path.append('.')

import os
import json
import argparse

import numpy as np
import plotly.graph_objects as go

from dataprocessing.preprocess_obj import load_obj, _rotvec_to_matrix_np


def _resolve(path, basename_dir='datasets/3dshape'):
    """Best-effort resolution of an OBJ path stored in meta.json.

    meta paths are absolute container paths (``/workspace/...``); fall back to a
    same-basename file under ``basename_dir`` so the script also works on the host.
    """
    if path and os.path.exists(path):
        return path
    cand = os.path.join(basename_dir, os.path.basename(path)) if path else None
    if cand and os.path.exists(cand):
        return cand
    return path


def _combined_mesh(V_local, F, configs, color, name, eps_show=0):
    """One Mesh3d holding every placement in ``configs`` (each (R, p)).

    configs : list of (R(3,3), p(3,)) tuples.
    """
    nv = V_local.shape[0]
    verts, faces = [], []
    for k, (R, p) in enumerate(configs):
        verts.append(V_local @ R.T + p)
        faces.append(F + k * nv)
    if not verts:
        return None
    V = np.concatenate(verts, axis=0)
    Fc = np.concatenate(faces, axis=0)
    return go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=Fc[:, 0], j=Fc[:, 1], k=Fc[:, 2],
        color=color, opacity=0.5, name=name, flatshading=True, showlegend=True)


def main():
    ap = argparse.ArgumentParser(description='Visualize trans_n / rot_n directions.')
    ap.add_argument('--dataPath', required=True,
                    help='Preprocess output dir (sampled_points.npy, trans_n.npy, ...).')
    ap.add_argument('--out', default=None,
                    help='Output HTML (default <dataPath>/trans_rot_check.html).')
    ap.add_argument('--shape', default=None, help='Override shape OBJ path.')
    ap.add_argument('--env', default=None, help='Override env OBJ path (for walls).')
    ap.add_argument('--cnt', type=int, default=30,
                    help='Number of placements to include (one per slider step).')
    ap.add_argument('--endpoint', type=int, default=0, choices=(0, 1),
                    help='Which endpoint of each pair to show (0=x0, 1=x1).')
    ap.add_argument('--trans_eps', type=float, default=0.05,
                    help='Translation step length along trans_n (env-normalized units).')
    ap.add_argument('--rot_eps', type=float, default=0.3,
                    help='Rotation angle along rot_n (radians).')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.dataPath, 'meta.json')) as f:
        meta = json.load(f)

    pts = np.load(os.path.join(args.dataPath, 'sampled_points.npy'))   # (N,12)
    trans = np.load(os.path.join(args.dataPath, 'trans_n.npy'))        # (N,12)
    rot = np.load(os.path.join(args.dataPath, 'rot_n.npy'))            # (N,12)
    env_points = np.load(os.path.join(args.dataPath, 'env.npy'))       # (M,3)

    off = 0 if args.endpoint == 0 else 6
    rot_norm = float(meta.get('rot_norm', 2 * np.pi))

    # ── Shape mesh, normalized exactly like preprocess (V_sh - centre)/scale*sc ──
    shape_obj = _resolve(args.shape or meta.get('shape_obj'))
    scale = float(meta['env_scale'])
    V_sh, F_sh, _ = load_obj(shape_obj)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    V_sh_local = (V_sh - shape_center) / scale * float(meta.get('shape_scale', 1.0))

    # ── Walls (translucent context), reconstructed from the env OBJ if available ──
    wall_V = wall_F = None
    env_obj = _resolve(args.env or meta.get('env_obj'))
    if env_obj and os.path.exists(env_obj):
        V_env, F_env, names_env = load_obj(env_obj)
        center_env = np.asarray(meta['env_center'], dtype=np.float64)
        wall_V = (V_env - center_env) / scale
        sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
        wall_F = F_env[~sample_mask]

    rng = np.random.default_rng(args.seed)
    N = pts.shape[0]
    m = min(args.cnt, N)
    sel = rng.choice(N, size=m, replace=False)

    # ── Static traces: env points + walls (always visible) ──
    static = []
    if len(env_points) > 0:
        sub = env_points[rng.choice(
            len(env_points), size=min(len(env_points), 8000), replace=False)]
        static.append(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2], mode='markers',
            name='environment', marker=dict(size=1.5, color='grey', opacity=0.3)))
    if wall_V is not None and wall_F is not None and len(wall_F) > 0:
        static.append(go.Mesh3d(
            x=wall_V[:, 0], y=wall_V[:, 1], z=wall_V[:, 2],
            i=wall_F[:, 0], j=wall_F[:, 1], k=wall_F[:, 2],
            color='lightblue', opacity=0.12, name='walls', flatshading=True))
    n_static = len(static)

    def _build(i):
        """Five dynamic traces for placement index ``i`` (orig/trans/rot + 2 lines)."""
        c = pts[i, off:off + 6]
        p = c[0:3].astype(np.float64)
        R0 = _rotvec_to_matrix_np(c[3:6] * rot_norm)

        tdir = trans[i, off:off + 3].astype(np.float64)         # unit translation dir
        p_t = p + tdir * args.trans_eps

        axis = rot[i, off + 3:off + 6].astype(np.float64)       # unit rotation axis
        Rd = _rotvec_to_matrix_np(axis * args.rot_eps)
        R_rot = Rd @ R0                                          # world-frame rot about centre

        orig = _combined_mesh(V_sh_local, F_sh, [(R0, p)], 'grey', 'original')
        tran = _combined_mesh(V_sh_local, F_sh, [(R0, p_t)], 'limegreen',
                              'placement + trans_n*eps')
        rota = _combined_mesh(V_sh_local, F_sh, [(R_rot, p)], 'crimson',
                              'placement + rot_n*eps')

        # trans direction segment (centre -> nudged centre)
        tline = go.Scatter3d(
            x=[p[0], p_t[0]], y=[p[1], p_t[1]], z=[p[2], p_t[2]], mode='lines',
            line=dict(color='green', width=6), name='trans_n')
        # rot axis segment through the centre (both directions, for visibility)
        L = max(np.linalg.norm(V_sh_local, axis=1).max(), 1e-6)
        a0, a1 = p - axis * L, p + axis * L
        rline = go.Scatter3d(
            x=[a0[0], a1[0]], y=[a0[1], a1[1]], z=[a0[2], a1[2]], mode='lines',
            line=dict(color='red', width=4, dash='dot'), name='rot_n axis')
        return [orig, tran, rota, tline, rline]

    # Initial figure: statics + first placement's dynamic traces.
    dyn0 = _build(sel[0])
    fig = go.Figure(data=static + dyn0)

    # Frames update only the dynamic traces (indices n_static .. n_static+4).
    dyn_idx = list(range(n_static, n_static + 5))
    frames = []
    for s in sel:
        frames.append(go.Frame(data=_build(s), traces=dyn_idx, name=str(int(s))))
    fig.frames = frames

    steps = [dict(method='animate', label=str(int(s)),
                  args=[[str(int(s))],
                        dict(mode='immediate', frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))])
             for s in sel]
    fig.update_layout(
        title='trans_n (green) / rot_n (red) check  —  '
              'trans_eps={}, rot_eps={} rad'.format(args.trans_eps, args.rot_eps),
        sliders=[dict(active=0, steps=steps,
                      currentvalue=dict(prefix='placement idx: '))],
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z',
                   aspectmode='data', camera=dict(up=dict(x=0, y=1, z=0))))

    out = args.out or os.path.join(args.dataPath, 'trans_rot_check.html')
    fig.write_html(out, include_plotlyjs='cdn')
    print('Saved trans_n / rot_n visualization: {}  ({} placements)'.format(out, m))


if __name__ == '__main__':
    main()
