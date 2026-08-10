"""Exact MESH-TO-MESH collision check of a 3-D shape dataset.

The preprocessor (and my earlier point-based check) only asks whether one of the
10k SAMPLED env surface points falls inside the placed shape.  That is a sampling
approximation: a thin obstacle feature passing between samples is missed.  This
script does the real thing -- triangles of the transformed shape against
triangles of the environment mesh -- so nothing depends on sample density.

Two closed solids overlap iff their surfaces intersect, or one wholly contains
the other.  We test all three:

    surface intersection : triangle-triangle, via 6 segment-triangle tests per
                           triangle pair (Moller-Trumbore with t in [0,1]).
                           Exact except for exactly-coplanar overlap, which is
                           measure-zero and implies penetration elsewhere anyway.
    shape inside body    : a shape vertex inside the body   (igl.signed_distance)
    body inside shape    : a body vertex inside the shape   (igl.signed_distance)

igl in this image ships no triangle-triangle routine (no fast_find_self_
intersections, no CGAL copyleft), so the surface test is implemented here and
igl is used for the containment half.

CONTAINER vs OBSTACLE.  env1.obj holds 5 closed boxes.  Four are real obstacles.
The fifth ('wall (2)') spans the whole workspace -- it is the container the robot
moves INSIDE, so overlapping its interior is required, not forbidden.  For that
body only a surface crossing counts (the shape poking out of the workspace), and
it is reported separately from true obstacle collisions.

Run inside the pytorch docker, from the nested ntrl-demo root.
"""

import sys
sys.path.append('.')

import os
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import igl

from dataprocessing.preprocess_obj import load_obj


def rotvec_to_matrix_t(rv):
    """(N,3) rotation vectors in radians -> (N,3,3) rotation matrices (torch)."""
    theta = rv.norm(dim=1, keepdim=True)
    k = rv / theta.clamp(min=1e-12)
    K = torch.zeros(len(rv), 3, 3, device=rv.device, dtype=rv.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    th = theta[:, :, None]
    I = torch.eye(3, device=rv.device, dtype=rv.dtype).expand(len(rv), 3, 3)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def split_bodies(F):
    """Group triangles into connected components by shared vertices."""
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for t in F:
        r = find(int(t[0]))
        for a in t[1:]:
            ra = find(int(a))
            if ra != r:
                par[ra] = r
    g = defaultdict(list)
    for i, t in enumerate(F):
        g[find(int(t[0]))].append(i)
    return [np.array(v) for v in g.values()]


def seg_tri_hit(P0, d, A, E1, E2, eps=1e-12):
    """Batched segment-triangle intersection (Moller-Trumbore, t in [0,1]).

    All args broadcast to (..., 3).  P0 + t*d is the segment; the triangle is
    A, A+E1, A+E2.  Returns a bool tensor of the broadcast shape[:-1].
    """
    p = torch.cross(d, E2, dim=-1)
    det = (E1 * p).sum(-1)
    ok = det.abs() > eps
    inv = torch.where(ok, 1.0 / torch.where(ok, det, torch.ones_like(det)),
                      torch.zeros_like(det))
    tv = P0 - A
    u = (tv * p).sum(-1) * inv
    ok &= (u >= 0) & (u <= 1)
    q = torch.cross(tv, E1, dim=-1)
    v = (d * q).sum(-1) * inv
    ok &= (v >= 0) & (u + v <= 1)
    t = (E2 * q).sum(-1) * inv
    ok &= (t >= 0) & (t <= 1)
    return ok


def surfaces_intersect(SV, ST, BV, BT):
    """Do the two triangle soups intersect?  SV (K,ns,3), ST (nt,3) indices;
    BV (nb,3) static verts, BT (mt,3).  Returns (K,) bool."""
    K = SV.shape[0]
    dev = SV.device
    s_tri = SV[:, ST]                                     # (K,nt,3,3)
    b_tri = BV[BT].to(dev)                                # (mt,3,3)

    def edges(tri):
        # (...,3,3) -> P0 (...,3,3), d (...,3,3) for the 3 edges
        P0 = tri
        P1 = tri[..., [1, 2, 0], :]
        return P0, P1 - P0

    # shape edges vs body triangles
    sP0, sd = edges(s_tri)                                # (K,nt,3,3)
    sP0 = sP0.reshape(K, -1, 1, 3)                        # (K,nt*3,1,3)
    sd = sd.reshape(K, -1, 1, 3)
    bA = b_tri[:, 0].reshape(1, 1, -1, 3)
    bE1 = (b_tri[:, 1] - b_tri[:, 0]).reshape(1, 1, -1, 3)
    bE2 = (b_tri[:, 2] - b_tri[:, 0]).reshape(1, 1, -1, 3)
    hit = seg_tri_hit(sP0, sd, bA, bE1, bE2).any(-1).any(-1)   # (K,)

    # body edges vs shape triangles
    bP0, bd = edges(b_tri)                                # (mt,3,3)
    bP0 = bP0.reshape(1, -1, 1, 3)
    bd = bd.reshape(1, -1, 1, 3)
    sA = s_tri[:, :, 0].reshape(K, 1, -1, 3)
    sE1 = (s_tri[:, :, 1] - s_tri[:, :, 0]).reshape(K, 1, -1, 3)
    sE2 = (s_tri[:, :, 2] - s_tri[:, :, 0]).reshape(K, 1, -1, 3)
    hit |= seg_tri_hit(bP0, bd, sA, sE1, sE2).any(-1).any(-1)
    return hit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1')
    p.add_argument('--shape', default=None)
    p.add_argument('--n', type=int, default=0, help='rows to check (0 = all)')
    p.add_argument('--chunk', type=int, default=200000)
    p.add_argument('--cand_chunk', type=int, default=20000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--container', default='wall',
                   help="substring naming the container body (surface-crossing "
                        "only); '' to treat every body as a solid obstacle")
    args = p.parse_args()
    dev = args.device

    meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))
    scale = float(meta['env_scale'])
    shape_scale = float(meta.get('shape_scale', 1.0))
    shape_obj = args.shape or os.path.join(
        'datasets/3dshape', os.path.basename(meta['shape_obj']))
    env_obj = os.path.join('datasets/3dshape', os.path.basename(meta['env_obj']))

    # ── meshes, normalized exactly as preprocess_obj.main does ──
    V_env, F_env, names = load_obj(env_obj)
    bb0, bb1 = V_env.min(axis=0), V_env.max(axis=0)
    Vn = (V_env - 0.5 * (bb0 + bb1)) / scale

    V_sh, F_sh, _ = load_obj(shape_obj)
    V_loc = ((V_sh - 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0)))
             / scale * shape_scale)

    bodies = split_bodies(F_env)
    body_name = []
    for tris in bodies:
        # name of the group that owns the first triangle
        body_name.append(str(names[tris[0]]) if names is not None else '?')

    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    if args.n and args.n < len(pairs):
        pairs = pairs[:args.n]

    print('shape : {}  ({} verts, {} tris)'.format(shape_obj, len(V_loc), len(F_sh)))
    print('env   : {}  ({} verts, {} tris, {} bodies)'.format(
        env_obj, len(Vn), len(F_env), len(bodies)))
    for i, tris in enumerate(bodies):
        role = 'CONTAINER' if (args.container and args.container in body_name[i].lower()) \
            else 'obstacle'
        print('   body {}: {:12s} {:2d} tris  [{}]'.format(i, body_name[i], len(tris), role))
    print('pairs : {}  -> {} placements\n'.format(len(pairs), 2 * len(pairs)))

    Vn_t = torch.tensor(Vn, dtype=torch.float64, device=dev)
    V_loc_t = torch.tensor(V_loc, dtype=torch.float64, device=dev)
    ST = torch.tensor(F_sh.astype(np.int64), device=dev)

    # static per-body data
    body_data = []
    for i, tris in enumerate(bodies):
        BT = torch.tensor(F_env[tris].astype(np.int64), device=dev)
        vs = np.unique(F_env[tris])
        aabb = (torch.tensor(Vn[vs].min(0), device=dev, dtype=torch.float64),
                torch.tensor(Vn[vs].max(0), device=dev, dtype=torch.float64))
        is_container = bool(args.container and args.container in body_name[i].lower())
        body_data.append(dict(
            BT=BT, aabb=aabb, container=is_container, name=body_name[i],
            V=np.ascontiguousarray(Vn), F=np.ascontiguousarray(F_env[tris].astype(np.int64)),
            verts_world=np.ascontiguousarray(Vn[vs])))

    V_loc_c = np.ascontiguousarray(V_loc)
    F_sh_c = np.ascontiguousarray(F_sh.astype(np.int64))
    WN = igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER

    summary = {}
    for tag, sl in (('x0', slice(0, 6)), ('x1', slice(6, 12))):
        N = len(pairs)
        collide = np.zeros(N, dtype=bool)          # overlaps a real obstacle
        escape = np.zeros(N, dtype=bool)           # crosses the container wall
        per_body = np.zeros(len(bodies), dtype=np.int64)
        t0 = time.time()

        for s in range(0, N, args.chunk):
            e = min(s + args.chunk, N)
            cfg = torch.tensor(pairs[s:e, sl], dtype=torch.float64, device=dev)
            t = cfg[:, 0:3]
            R = rotvec_to_matrix_t(cfg[:, 3:6] * (2.0 * np.pi))
            SV = torch.einsum('nij,vj->nvi', R, V_loc_t) + t[:, None, :]   # (C,8,3)
            lo, hi = SV.min(dim=1).values, SV.max(dim=1).values

            for bi, bd in enumerate(body_data):
                a0, a1 = bd['aabb']
                # broad phase: AABB overlap is necessary for surface intersection
                ov = ((lo <= a1).all(1) & (hi >= a0).all(1))
                idx = torch.nonzero(ov, as_tuple=False).squeeze(1)
                if idx.numel() == 0:
                    continue
                hits = torch.zeros(idx.numel(), dtype=torch.bool, device=dev)
                for cs in range(0, idx.numel(), args.cand_chunk):
                    ci = idx[cs:cs + args.cand_chunk]
                    hits[cs:cs + args.cand_chunk] = surfaces_intersect(
                        SV[ci], ST, Vn_t, bd['BT'])
                hit_np = hits.cpu().numpy()
                rows = idx.cpu().numpy() + s

                if bd['container']:
                    escape[rows[hit_np]] = True
                    continue

                bad = hit_np.copy()
                # containment, for candidates with no surface crossing
                rest = np.nonzero(~hit_np)[0]
                if len(rest):
                    ri = idx[torch.tensor(rest, device=dev)]
                    # (a) shape wholly inside body: test its vertices in world
                    q = SV[ri].reshape(-1, 3).cpu().numpy()
                    S1, _, _, _ = igl.signed_distance(
                        np.ascontiguousarray(q), bd['V'], bd['F'], WN)
                    inA = (S1.reshape(len(rest), -1) < 0).any(1)
                    # (b) body wholly inside shape: its verts in the shape's local frame
                    bw = torch.tensor(bd['verts_world'], dtype=torch.float64, device=dev)
                    rel = bw[None, :, :] - t[ri][:, None, :]
                    q2 = torch.einsum('nji,nvj->nvi', R[ri], rel).reshape(-1, 3).cpu().numpy()
                    S2, _, _, _ = igl.signed_distance(
                        np.ascontiguousarray(q2), V_loc_c, F_sh_c, WN)
                    inB = (S2.reshape(len(rest), -1) < 0).any(1)
                    bad[rest] = inA | inB
                collide[rows[bad]] = True
                per_body[bi] += int(bad.sum())

        free = ~collide
        summary[tag] = (free, escape)
        print('{}: collision-free (obstacles) {} / {} = {:.4f}%   [{:.1f}s]'.format(
            tag, int(free.sum()), N, 100.0 * free.mean(), time.time() - t0))
        print('    colliding {}   crossing container wall {}'.format(
            int(collide.sum()), int(escape.sum())))
        for bi, bd in enumerate(body_data):
            if not bd['container']:
                print('      body {} ({}): {} collisions'.format(bi, bd['name'], per_body[bi]))

    both = summary['x0'][0] & summary['x1'][0]
    print('\npairs with BOTH endpoints obstacle-free: {} / {} = {:.4f}%'.format(
        int(both.sum()), len(both), 100.0 * both.mean()))


if __name__ == '__main__':
    main()
