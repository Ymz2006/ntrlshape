"""Independent collision check of a 3-D shape dataset, using igl.

The preprocessor decides "collision free" with its own point-in-tetrahedron test
(preprocess_obj.points_inside_tets): an env surface point inside any tet of the
placed shape is a collision.  This script re-derives that verdict from scratch
with igl.signed_distance and reports the proportion of sampled placements that
come out collision free, so the two implementations can be compared.

Method.  Rather than transforming the shape per placement (which would force a
fresh AABB tree for every one of the ~1.6M placements), we transform the ENV
POINTS into each placement's LOCAL frame:

    p_local = R^T (p_world - t)

The shape mesh is then static, so a single igl.signed_distance call over all
candidate points answers everything.  A GPU broad phase first discards env
points farther than the shape's bounding radius from the placement centre --
those provably cannot be inside -- which leaves only a small candidate set.

Run inside the pytorch docker, from the nested ntrl-demo root.
"""

import sys
sys.path.append('.')

import os
import json
import time
import argparse

import numpy as np
import torch
import igl

from dataprocessing.preprocess_obj import load_obj


def rotvec_to_matrix_np(rv):
    """(N,3) rotation vectors (radians) -> (N,3,3) rotation matrices."""
    theta = np.linalg.norm(rv, axis=1, keepdims=True)          # (N,1)
    k = rv / np.maximum(theta, 1e-12)
    K = np.zeros((len(rv), 3, 3), dtype=np.float64)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    th = theta[:, :, None]
    I = np.eye(3)[None].repeat(len(rv), axis=0)
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def inside_counts(centers, R, env, V_loc, F_loc, R_shape, device, chunk, sign_type):
    """Number of env points strictly inside each placement.

    centers (N,3), R (N,3,3) world<-local, env (E,3), local shape mesh (V_loc,F_loc).
    Returns (N,) int array of interior env-point counts.
    """
    N = len(centers)
    env_t = torch.tensor(env, dtype=torch.float32, device=device)
    cen_t = torch.tensor(centers, dtype=torch.float32, device=device)

    counts = np.zeros(N, dtype=np.int64)
    n_cand_total = 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        # ── broad phase: |p - t| <= R_shape is necessary for p to be inside ──
        d = torch.cdist(cen_t[s:e], env_t)                     # (C,E)
        pi, ei = torch.nonzero(d <= R_shape, as_tuple=True)
        if pi.numel() == 0:
            continue
        pi_np = pi.cpu().numpy() + s
        ei_np = ei.cpu().numpy()
        n_cand_total += len(pi_np)

        # ── exact test: pull candidates into the placement's local frame ──
        rel = env[ei_np] - centers[pi_np]                      # (K,3)
        p_loc = np.einsum('kji,kj->ki', R[pi_np], rel)         # R^T @ rel
        S, _, _, _ = igl.signed_distance(
            np.ascontiguousarray(p_loc), V_loc, F_loc, sign_type)
        hit = S < 0.0
        if hit.any():
            np.add.at(counts, pi_np[hit], 1)
    return counts, n_cand_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1')
    p.add_argument('--shape', default=None,
                   help='shape OBJ (default: meta.json shape_obj basename)')
    p.add_argument('--n', type=int, default=0, help='rows to check (0 = all)')
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--device', default='cuda')
    p.add_argument('--sign', default='winding',
                   choices=['winding', 'fast_winding', 'pseudonormal', 'default'])
    args = p.parse_args()

    sign_type = {
        'winding': igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
        'fast_winding': igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
        'pseudonormal': igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
        'default': igl.SIGNED_DISTANCE_TYPE_DEFAULT,
    }[args.sign]

    meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))
    scale = float(meta['env_scale'])
    shape_scale = float(meta.get('shape_scale', 1.0))
    shape_obj = args.shape or os.path.join(
        'datasets/3dshape', os.path.basename(meta['shape_obj']))

    # Local shape mesh, normalized exactly as preprocess_obj.main does.
    V_sh, F_sh, _ = load_obj(shape_obj)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    V_loc = np.ascontiguousarray(
        ((V_sh - shape_center) / scale * shape_scale).astype(np.float64))
    F_loc = np.ascontiguousarray(F_sh.astype(np.int64))
    R_shape = float(np.linalg.norm(V_loc, axis=1).max())

    env = np.load(os.path.join(args.dataPath, 'env.npy')).astype(np.float64)
    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    if args.n and args.n < len(pairs):
        pairs = pairs[:args.n]

    print('shape      : {}  ({} verts, {} tris)  R_shape={:.4f}'.format(
        shape_obj, len(V_loc), len(F_loc), R_shape))
    print('env points : {}'.format(len(env)))
    print('pairs      : {}   -> {} placements'.format(len(pairs), 2 * len(pairs)))
    print('sign type  : {}\n'.format(args.sign))

    results = {}
    for tag, sl in (('x0', slice(0, 6)), ('x1', slice(6, 12))):
        cfg = pairs[:, sl]
        centers = np.ascontiguousarray(cfg[:, 0:3])
        rotvec = cfg[:, 3:6] * (2.0 * np.pi)          # stored /2pi -> radians
        R = rotvec_to_matrix_np(rotvec)

        t0 = time.time()
        counts, ncand = inside_counts(centers, R, env, V_loc, F_loc,
                                      R_shape, args.device, args.chunk, sign_type)
        free = counts == 0
        results[tag] = free
        print('{}: collision-free {} / {}  = {:.4f}%   '
              '(colliding {},  candidates tested {},  {:.1f}s)'.format(
                  tag, int(free.sum()), len(free), 100.0 * free.mean(),
                  int((~free).sum()), ncand, time.time() - t0))
        if (~free).any():
            bad = np.nonzero(~free)[0]
            print('    worst offenders (row, interior env pts):',
                  [(int(i), int(counts[i])) for i in bad[np.argsort(-counts[bad])][:5]])

    both = results['x0'] & results['x1']
    print('\npairs with BOTH endpoints collision-free: {} / {} = {:.4f}%'.format(
        int(both.sum()), len(both), 100.0 * both.mean()))

    # ── positive control ──────────────────────────────────────────────────────
    # A checker that never reports a collision would also print 100% above, so
    # verify it fires: snap placements onto env points (centre = an env point),
    # which forces that point to sit at the shape's centre, i.e. inside it.
    n_ctrl = min(2000, len(pairs))
    ctrl_centers = np.ascontiguousarray(env[:n_ctrl])
    ctrl_R = rotvec_to_matrix_np(np.zeros((n_ctrl, 3)))
    ctrl_counts, _ = inside_counts(ctrl_centers, ctrl_R, env, V_loc, F_loc,
                                   R_shape, args.device, args.chunk, sign_type)
    ctrl_free = float((ctrl_counts == 0).mean())
    print('\ncontrol (shape centred ON env points, n={}): collision-free {:.2f}%  '
          '-- expect ~0%, confirms the checker fires'.format(n_ctrl, 100 * ctrl_free))


if __name__ == '__main__':
    main()
