"""DXF based preprocessing for 2-D shape path planning.

This is the ntrl-demo equivalent of ``dataprocessing/preprocess.py`` for the
case where the "robot" is a rigid 2-D shape (e.g. the F-shape) that has to be
moved through a 2-D environment.  Instead of a 3-D mesh it consumes two
*normalized* DXF files (see ``normaldxf.py`` for the normalization step that is
done exactly like in ``2dshape_baseline``):

    * an environment DXF  -- the static obstacles / walls
    * a shape DXF         -- the moving body (drawn as a closed LWPOLYLINE)

The configuration space is SE(2): every sample is ``(x, y, theta)`` with
``theta`` stored normalized by ``2*pi`` so that all three coordinates live on a
comparable scale (the network treats them identically through its Fourier
features).

Sampling matches ``dataprocessing/speed_sampling_gpu_kdtree_normal.py`` (the
gibson sampler) in spirit: each training row is a correlated pair ``(x0, x1)``
where

    * ``x0`` is drawn in the **narrow band** ``offset < clearance < margin`` --
      only there does the eikonal / normal loss carry useful gradient signal
      (in pure free space ``speed=1`` and the eikonal PDE is trivial);
    * ``x1`` is a random SE(2) displacement away from ``x0`` (uniform direction,
      length ~ U[0, sqrt(3)] in the 2*pi-normalized SE(2) metric) and is kept
      iff it is collision-free with clearance ``> offset``.

For each endpoint we record

    * ``speed``  -- ``clip(clearance / margin, offset/margin, 1)``  (same formula
                    as the gibson sampler)
    * ``normal`` -- unit gradient of the clearance field in configuration
                    space.  Built by taking the closest point between the
                    placed shape and the environment and converting that
                    workspace contact normal into an SE(2) direction.  This is
                    exactly the target the ntrl-demo normal-loss term consumes.

Outputs (written to ``--out``):
    sampled_points.npy  (N, 6)  -- pair per row [x0, y0, th0, x1, y1, th1]
    speed.npy           (N, 2)  -- [speed0, speed1]
    normal.npy          (N, 6)  -- [n0(3) | n1(3)]
    env.npy             (M, 2)  -- environment boundary points
    meta.json                   -- shape_scale, margin, offset, ...

Usage:
    python dataprocessing/preprocess_dxf.py \
        --env   datasets/2dshape/Fmaze2_norm.dxf \
        --shape datasets/2dshape/Fshape_norm.dxf \
        --out   datasets/2dshape/Fmaze2_norm \
        --num_samples 400000 \
        --margin 0.1  --offset 0.005  --shape_scale 1.0
"""

import sys
sys.path.append('.')

import os
import json
import time
import argparse

import numpy as np
import torch
import ezdxf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from shapely.geometry import Polygon

from dataprocessing.parse_shape import dxf_to_shape, shape_to_points


# ---------------------------------------------------------------------------
# The F-shape, triangulated in its own local frame (k triangles, 3 verts, 2-D).
# Collision / distance queries run on this triangle soup; the matching closed
# polygon (from the shape DXF) is only used for plotting.
# ---------------------------------------------------------------------------
FSHAPE_TRIANGLES = [
    [[0.086, 0.171], [0.086, 0.157], [0.014, 0.157]],
    [[0.086, 0.171], [0.014, 0.157], [0.000, 0.171]],
    [[0.000, 0.171], [0.014, 0.157], [0.000, 0.000]],
    [[0.014, 0.000], [0.014, 0.086], [0.000, 0.000]],
    [[0.014, 0.100], [0.071, 0.100], [0.014, 0.086]],
    [[0.014, 0.086], [0.071, 0.100], [0.071, 0.086]],
]

# Number of boundary samples taken per unit length of every DXF entity.
DOTS_PER_M = 70


# ---------------------------------------------------------------------------
# DXF -> environment boundary point cloud
# ---------------------------------------------------------------------------
def sample_points(msp):
    """Densely sample points along every entity of a DXF modelspace."""
    points = []
    for e in msp:
        t = e.dxftype()

        if t == "LINE":
            start = np.array([e.dxf.start.x, e.dxf.start.y])
            end = np.array([e.dxf.end.x, e.dxf.end.y])
            length = np.linalg.norm(end - start)
            dot_cnt = max(int(length * DOTS_PER_M), 1)
            for i in range(dot_cnt):
                points.append(start + (end - start) * (i / dot_cnt))

        elif t in ["LWPOLYLINE", "POLYLINE"]:
            verts = np.array([[v[0], v[1]] for v in e.get_points()])
            closed = e.closed if hasattr(e, 'closed') else False
            for i in range(len(verts) - 1):
                start, end = verts[i], verts[i + 1]
                length = np.linalg.norm(end - start)
                dot_cnt = max(int(length * DOTS_PER_M), 1)
                for j in range(dot_cnt):
                    points.append(start + (end - start) * (j / dot_cnt))
            if closed:
                start, end = verts[-1], verts[0]
                length = np.linalg.norm(end - start)
                dot_cnt = max(int(length * DOTS_PER_M), 1)
                for j in range(dot_cnt):
                    points.append(start + (end - start) * (j / dot_cnt))

        elif t == "CIRCLE":
            center = np.array([e.dxf.center.x, e.dxf.center.y])
            radius = e.dxf.radius
            num_points = max(int(2 * np.pi * radius * DOTS_PER_M), 4)
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                points.append(center + radius * np.array([np.cos(angle), np.sin(angle)]))

    return np.array(points)


# ---------------------------------------------------------------------------
# Closest-point clearance distance + SE(2) clearance normal
# ---------------------------------------------------------------------------
def calculate_dist(shape_lines, shape_points, env_points, configs):
    """Clearance of a batch of shape placements and its configuration-space normal.

    Parameters
    ----------
    shape_lines  : (batch, 1, k, 3, 2, 1)  edge vectors  v_i -> v_{i+1}
    shape_points : (batch,    k, 3, 2, 1)  transformed triangle vertices (edge starts)
    env_points   : (E, 2)                  environment boundary point cloud
    configs      : (batch, 3)              sampled (x, y, theta) of each placement

    Returns
    -------
    min_dist : (batch,)     minimum distance between the placed shape and the env
    normal   : (batch, 3)   unit (d/dx, d/dy, d/dtheta) direction that increases
                            clearance -- theta chain-ruled to the 2*pi-normalized
                            coordinate used everywhere else in the pipeline.
    """
    AB = shape_lines.squeeze()        # (batch, k, 3, 2)  edge vectors
    A = shape_points.squeeze()        # (batch, k, 3, 2)  edge start vertices
    batch = A.shape[0]
    k3 = A.shape[1] * 3               # entries per environment point (k triangles x 3 edges)

    env = env_points.to(A.device).float()             # (E, 2)
    env_b = env.view(1, -1, 1, 1, 2)                   # (1, E, 1, 1, 2)
    A_b = A.unsqueeze(1)                               # (batch, 1, k, 3, 2)
    AB_b = AB.unsqueeze(1)                             # (batch, 1, k, 3, 2)

    # Project every env point onto every shape edge, clamped to the segment.
    AP = env_b - A_b                                   # (batch, E, k, 3, 2)
    AB_len2 = (AB_b * AB_b).sum(-1) + 1e-12            # (batch, 1, k, 3)
    proj = ((AP * AB_b).sum(-1) / AB_len2).clamp(0.0, 1.0)   # (batch, E, k, 3)
    del AP, AB_len2

    # diff = (closest point on edge) - (env point)
    diff = A_b + proj.unsqueeze(-1) * AB_b - env_b     # (batch, E, k, 3, 2)
    del proj
    dist = torch.linalg.norm(diff, dim=-1)             # (batch, E, k, 3)

    flat_dist = dist.reshape(batch, -1)                # (batch, E*k*3)
    min_dist, min_idx = flat_dist.min(dim=1)           # (batch,), (batch,)

    # Recover the closest pair: env point e* and shape point s*.
    flat_diff = diff.reshape(batch, -1, 2)
    gather_idx = min_idx.view(batch, 1, 1).expand(-1, 1, 2)
    sel_diff = torch.gather(flat_diff, 1, gather_idx).squeeze(1)        # (batch, 2)  s* - e*
    del diff, dist, flat_diff, flat_dist

    e_idx = torch.div(min_idx, k3, rounding_mode='floor')              # which env point
    sel_env = env[e_idx]                                               # (batch, 2)  e*
    sel_closest = sel_diff + sel_env                                   # (batch, 2)  s*

    # Workspace contact normal: unit vector from the closest obstacle point
    # towards the shape -- the direction the shape moves to gain clearance.
    n_ws = sel_diff / (min_dist.unsqueeze(1) + 1e-12)                   # (batch, 2)

    # Configuration-space gradient of the clearance field:
    #   d(clearance)/dx, d(clearance)/dy = n_ws
    #   d(clearance)/dtheta              = (s* - center) x n_ws
    center = configs[:, :2].to(A.device)
    r = sel_closest - center                                           # (batch, 2)
    d_theta = r[:, 0] * n_ws[:, 1] - r[:, 1] * n_ws[:, 0]               # cross(r, n_ws)

    # theta is stored normalized by 2*pi, so d/dtheta_norm = 2*pi * d/dtheta.
    normal = torch.stack([n_ws[:, 0], n_ws[:, 1], 2.0 * np.pi * d_theta], dim=1)
    normal = normal / (torch.linalg.norm(normal, dim=1, keepdim=True) + 1e-12)

    return min_dist, normal


# ---------------------------------------------------------------------------
# Per-batch placement evaluation: collision-free mask + clearance + normal
# ---------------------------------------------------------------------------
def evaluate_placements(configs, shape_tensor_points_h, environment_boundary_points,
                        env_points_subtract, device):
    """Evaluate a batch of SE(2) placements.

    Parameters
    ----------
    configs              : (B, 3)  raw (x, y, theta) on CPU
    shape_tensor_points_h: (k, 3, 3) homogeneous shape verts (on CPU)
    environment_boundary_points : (E, 2) env points on CPU (for distance query)
    env_points_subtract  : pre-reshaped env tensor (on GPU after first call)
    device               : torch device

    Returns
    -------
    is_free : (B,) bool, CPU   -- no env point lies inside any shape triangle
    dist    : (B,) float, CPU  -- clearance distance (shape edge -> nearest env pt)
    normal  : (B, 3) float, CPU -- unit SE(2) clearance normal (theta in /2pi units)
    """
    B = configs.shape[0]
    tx, ty = configs[:, 0], configs[:, 1]
    cos, sin = torch.cos(configs[:, 2]), torch.sin(configs[:, 2])

    T = torch.zeros(B, 3, 3, device=device)
    T[:, 0, 0] = cos.to(device);  T[:, 0, 1] = -sin.to(device)
    T[:, 1, 0] = sin.to(device);  T[:, 1, 1] = cos.to(device)
    T[:, 0, 2] = tx.to(device);   T[:, 1, 2] = ty.to(device)
    T[:, 2, 2] = 1.0
    T = T.unsqueeze(1).unsqueeze(1)                                 # (B,1,1,3,3)

    shape_h = shape_tensor_points_h.to(device).unsqueeze(0).unsqueeze(-1)  # (1,k,3,3,1)
    transformed = torch.matmul(T, shape_h)                          # (B,k,3,3,1)
    transformed2d = transformed[:, :, :, 0:2, :].unsqueeze(1)       # (B,1,k,3,2,1)

    center_to_vertex = env_points_subtract - transformed2d

    vertex_to_vertex = transformed2d.clone()
    vertex_to_vertex[:, :, :, 0, :, :] = transformed2d[:, :, :, 1, :, :] - transformed2d[:, :, :, 0, :, :]
    vertex_to_vertex[:, :, :, 1, :, :] = transformed2d[:, :, :, 2, :, :] - transformed2d[:, :, :, 1, :, :]
    vertex_to_vertex[:, :, :, 2, :, :] = transformed2d[:, :, :, 0, :, :] - transformed2d[:, :, :, 2, :, :]

    a = vertex_to_vertex[..., 0, 0];  b = center_to_vertex[..., 0, 0]
    c = vertex_to_vertex[..., 1, 0];  d = center_to_vertex[..., 1, 0]
    cross_sign = torch.sign(a * d - b * c)
    inside = torch.where(abs(cross_sign.sum(dim=3)) == 3, 1, 0)     # (B,1,k)
    inside_any = inside.sum(dim=2).sum(dim=1)                       # (B,)
    is_free = (inside_any == 0)

    transformed2d_squeezed = transformed2d.squeeze(1)               # (B,k,3,2,1)
    dist, normal = calculate_dist(vertex_to_vertex, transformed2d_squeezed,
                                  environment_boundary_points, configs)
    return is_free.cpu(), dist.cpu(), normal.cpu()


# ---------------------------------------------------------------------------
# Correlated-pair rejection sampling -- mirrors speed_sampling_gpu_kdtree_normal
#
#   x0 : drawn in the narrow band  offset < dist(x0) < margin
#   x1 : x0 + random SE(2) displacement, kept iff collision-free and dist > offset
#
# This matches the gibson sampler in `dataprocessing/speed_sampling_gpu_kdtree_normal.py`:
# only narrow-band starts carry useful eikonal / normal gradient signal, while
# the paired goal x1 just has to be reachable / collision-free.
# ---------------------------------------------------------------------------
def generate_valid_pairs(number_pairs, shape_tensor_points, msp,
                         margin, offset,
                         batch_size=1000, device='cuda', testing=False):
    """Sample ``number_pairs`` (x0, x1) SE(2) placements.

    Training mode (default): correlated pairs -- ``x0`` is drawn in the narrow
    band ``offset < clearance < margin`` and ``x1`` is a random SE(2)
    displacement away from it, kept iff collision-free with clearance > offset.

    Testing mode (``testing=True``): ``x0`` and ``x1`` are *independent*,
    uniformly-random placements, kept iff *both* are simply collision-free (no
    clearance band, no correlation).  Use this to generate start/goal test pairs.

    Returns
    -------
    pairs       : (number_pairs, 6)  raw [x0,y0,th0, x1,y1,th1] (theta in radians)
    env_points  : (E, 2)             environment boundary points
    dists       : (number_pairs, 2)  raw clearance distances [d0, d1]
    normals     : (number_pairs, 6)  SE(2) clearance normals  [n0(3) | n1(3)]
    """
    triangle_cnt = shape_tensor_points.shape[0]
    shape_h = torch.cat(
        [shape_tensor_points, torch.ones(triangle_cnt, 3, 1)], 2)   # (k, 3, 3)

    env_boundary = sample_points(msp=msp)
    bb_min = np.min(env_boundary, axis=0)
    bb_max = np.max(env_boundary, axis=0)
    xrange = bb_max[0] - bb_min[0]
    yrange = bb_max[1] - bb_min[1]
    scale  = max(xrange, yrange)
    half_x = (xrange / scale - 0.02) * 0.5
    half_y = (yrange / scale - 0.02) * 0.5

    env_boundary = torch.tensor(env_boundary, dtype=torch.float32)

    env_subtract = env_boundary.clone() \
        .unsqueeze(-1).unsqueeze(0).unsqueeze(2).unsqueeze(2).to(device)   # (1,E,1,1,2,1)

    pairs   = torch.zeros(number_pairs, 6)
    dists   = torch.zeros(number_pairs, 2)
    normals = torch.zeros(number_pairs, 6)
    count   = 0
    sqrt3   = float(np.sqrt(3.0))
    two_pi  = 2.0 * np.pi

    while count < number_pairs:
        # ── Sample x0 uniformly inside the env bbox; theta in [-pi, pi] ──
        x0 = torch.empty(batch_size, 3)
        x0[:, 0].uniform_(-half_x, half_x)
        x0[:, 1].uniform_(-half_y, half_y)
        x0[:, 2].uniform_(-np.pi, np.pi)

        if testing:
            # ── x1: a second, *independent* uniformly-random placement ──
            x1 = torch.empty(batch_size, 3)
            x1[:, 0].uniform_(-half_x, half_x)
            x1[:, 1].uniform_(-half_y, half_y)
            x1[:, 2].uniform_(-np.pi, np.pi)
        else:
            # ── Sample correlated displacement to x1 in 2pi-normalized SE(2) ──
            d = torch.rand(batch_size, 3) - 0.5
            d = d / (torch.linalg.norm(d, dim=1, keepdim=True) + 1e-12)
            rL = torch.rand(batch_size, 1) * sqrt3
            delta = d * rL                                          # (B, 3) normalized

            x1 = x0.clone()
            x1[:, 0] = x0[:, 0] + delta[:, 0]
            x1[:, 1] = x0[:, 1] + delta[:, 1]
            x1[:, 2] = x0[:, 2] + delta[:, 2] * two_pi             # raw theta delta
            # wrap theta back into [-pi, pi]
            x1[:, 2] = ((x1[:, 2] + np.pi) % two_pi) - np.pi

        # ── Both endpoints must be inside the xy bbox (theta wraps) ──
        in_bbox = (x1[:, 0].abs() <= half_x) & (x1[:, 1].abs() <= half_y)
        if int(in_bbox.sum()) == 0:
            continue
        x0 = x0[in_bbox]
        x1 = x1[in_bbox]

        # ── Evaluate both endpoints ──
        free0, dist0, normal0 = evaluate_placements(
            x0, shape_h, env_boundary, env_subtract, device)
        free1, dist1, normal1 = evaluate_placements(
            x1, shape_h, env_boundary, env_subtract, device)

        if testing:
            # ── Testing data: independent points, only require no collision ──
            keep = free0 & free1 & (dist0 > offset) & (dist1 > offset)
        else:
            # ── Asymmetric filtering (matches gibson sampler) ──
            keep_x0 = free0 & (dist0 > offset) & (dist0 < margin)   # narrow band
            keep_x1 = free1 & (dist1 > offset)                      # loose: just reachable
            keep    = keep_x0 & keep_x1

        nk = int(keep.sum())
        if nk == 0:
            continue

        take = min(nk, number_pairs - count)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)[:take]

        pairs[count:count + take, 0:3]   = x0[idx]
        pairs[count:count + take, 3:6]   = x1[idx]
        dists[count:count + take, 0]     = dist0[idx]
        dists[count:count + take, 1]     = dist1[idx]
        normals[count:count + take, 0:3] = normal0[idx]
        normals[count:count + take, 3:6] = normal1[idx]
        count += take
        print(f'generated: {count} / {number_pairs}')

    return pairs, env_boundary, dists, normals


# ---------------------------------------------------------------------------
# Optional debug visualization of the sampled data
# ---------------------------------------------------------------------------
def rotate_points(points, x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    pts = np.array(points, dtype=np.float32)
    rot = np.array([[c, -s], [s, c]])
    return [(p[0], p[1]) for p in (rot @ pts.T).T + np.array([x, y])]


def visual_training(start, shape_points, env_points, cnt, speed, save_path):
    """Draw a subset of sampled placements colored by clearance speed."""
    cmap = get_cmap('viridis')
    norm = Normalize(vmin=0.0, vmax=float(np.max(speed)))
    fig, ax = plt.subplots()
    for i in range(min(cnt, len(start))):
        rotated = rotate_points(shape_points, start[i][0], start[i][1], start[i][2])
        if len(rotated) < 3:
            continue
        poly = Polygon(rotated)
        if not poly.is_valid:
            continue
        ax.add_patch(plt.Polygon(list(poly.exterior.coords),
                                 facecolor=cmap(norm(speed[i])),
                                 edgecolor='black', alpha=0.7))
    if len(env_points) > 0:
        ax.scatter(env_points[:, 0], env_points[:, 1], color='grey', s=4)
    ax.set_aspect('equal')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='clearance speed')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print('Saved sample visualization: ' + save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='DXF-based preprocessing for 2-D shape path planning.')
    parser.add_argument('--env', default='datasets/2dshape/FmazeEasy_norm.dxf',
                        help='Normalized environment DXF.')
    parser.add_argument('--shape', default='datasets/2dshape/Fshape_norm.dxf',
                        help='Normalized shape DXF (closed LWPOLYLINE, used for plots).')
    parser.add_argument('--out', default='datasets/2dshape/Fshape_FmazeEasy',
                        help='Output directory for the .npy training data.')
    parser.add_argument('--num_samples', type=int, default=400000,
                        help='Total number of sampled configs (split into pairs).')
    parser.add_argument('--shape_scale', type=float, default=1,
                        help='Uniform scale applied to the F-shape (1.0 = baseline size, '
                             '<1 shrinks).')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Upper band: x0 must have clearance < margin; maps to speed=1.')
    parser.add_argument('--offset', type=float, default=0.01,
                        help='Lower band: x0 must have clearance > offset, '
                             'and x1 (paired goal) must also have clearance > offset. '
                             'Maps to the minimum speed value offset/margin.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Sampling batch size (each batch evaluates 2x configs).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--visualize', action='store_true',
                        help='Also save a debug plot of the sampled placements.')
    parser.add_argument('--testing_data', action='store_true',
                        help='Generate TEST pairs: x0 and x1 are independent, '
                             'uniformly-random placements kept only if both are '
                             'collision-free (no narrow-band / clearance filtering).')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    doc = ezdxf.readfile(args.env)
    msp = doc.modelspace()

    shape_polygon = dxf_to_shape(args.shape)
    shape_points = shape_to_points(shape_polygon, args.shape_scale)   # for plotting only

    triangles = torch.tensor(FSHAPE_TRIANGLES, dtype=torch.float32) * args.shape_scale

    # --num_samples is interpreted as the number of (x0, x1) training pairs.
    num_pairs = int(args.num_samples)

    mode = 'TESTING (independent collision-free pairs)' if args.testing_data \
        else 'TRAINING (narrow-band correlated pairs)'
    print('Sampling {} (x0, x1) pairs [{}]   margin={}  offset={}  ...'.format(
        num_pairs, mode, args.margin, args.offset))
    t0 = time.time()
    pairs, env_points, dists, normals = generate_valid_pairs(
        num_pairs, triangles, msp,
        margin=args.margin, offset=args.offset,
        batch_size=args.batch_size, device=args.device,
        testing=args.testing_data)
    print('Sampling done in {:.1f}s'.format(time.time() - t0))

    pairs = pairs.cpu().numpy()
    env_points = env_points.cpu().numpy()
    dists = dists.cpu().numpy()           # (N, 2)
    normals = normals.cpu().numpy()       # (N, 6)

    # Eikonal speed term: clearance clipped into [offset/margin, 1]
    # (same formula as `speed_sampling_gpu_kdtree_normal.py`).
    speed_pairs = np.clip(dists / args.margin,
                          a_min=args.offset / args.margin, a_max=1.0)

    if args.visualize:
        # Visualize x0 placements (the narrow-band starts).
        start_x0 = pairs[:, 0:3].copy()
        visual_training(start_x0, shape_points, env_points, cnt=400,
                        speed=speed_pairs[:, 0],
                        save_path=os.path.join(args.out, 'sampled_placements.png'))

    # theta -> normalized by 2*pi so all three coords share a comparable scale.
    pairs[:, 2] /= (2 * np.pi)
    pairs[:, 5] /= (2 * np.pi)

    print('x0 range  : x=[{:.3f},{:.3f}]  y=[{:.3f},{:.3f}]  th_norm=[{:.3f},{:.3f}]'.format(
        pairs[:, 0].min(), pairs[:, 0].max(),
        pairs[:, 1].min(), pairs[:, 1].max(),
        pairs[:, 2].min(), pairs[:, 2].max()))
    print('x1 range  : x=[{:.3f},{:.3f}]  y=[{:.3f},{:.3f}]  th_norm=[{:.3f},{:.3f}]'.format(
        pairs[:, 3].min(), pairs[:, 3].max(),
        pairs[:, 4].min(), pairs[:, 4].max(),
        pairs[:, 5].min(), pairs[:, 5].max()))
    print('speed x0  : [{:.3f}, {:.3f}]   (band [{:.3f}, 1.0])'.format(
        speed_pairs[:, 0].min(), speed_pairs[:, 0].max(),
        args.offset / args.margin))
    print('speed x1  : [{:.3f}, {:.3f}]'.format(
        speed_pairs[:, 1].min(), speed_pairs[:, 1].max()))

    np.save(os.path.join(args.out, 'sampled_points'), pairs)
    np.save(os.path.join(args.out, 'speed'), speed_pairs)
    np.save(os.path.join(args.out, 'normal'), normals)
    np.save(os.path.join(args.out, 'env'), env_points)

    meta = {
        'shape_scale': float(args.shape_scale),
        'shape_dxf':   os.path.abspath(args.shape),
        'env_dxf':     os.path.abspath(args.env),
        'margin':      float(args.margin),
        'offset':      float(args.offset),
        'theta_norm':  float(2 * np.pi),
        'testing_data': bool(args.testing_data),
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('Wrote {} training pairs to {}'.format(pairs.shape[0], args.out))
    print('  sampled_points.npy', pairs.shape)
    print('  speed.npy         ', speed_pairs.shape)
    print('  normal.npy        ', normals.shape)
    print('  env.npy           ', env_points.shape)


if __name__ == '__main__':
    main()
