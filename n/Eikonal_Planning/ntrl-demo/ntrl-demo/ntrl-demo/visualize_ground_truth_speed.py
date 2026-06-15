"""Interactively probe the GROUND-TRUTH clearance / angle that feed speed_*.

This is the data-side counterpart of ``visual_pose.py``.  Where ``visual_pose.py``
drags a pose and reports what the *trained network* predicts, this script drags a
pose and reports the *ground-truth* geometric quantities that ``preprocess_obj.py``
computes for every sampled placement -- i.e. exactly what is reduced into
``speed_dists.npy`` and ``speed_angles.npy``.

Everything geometric is imported straight from ``dataprocessing/preprocess_obj``
(``load_obj``, ``tetrahedralize_shape``, ``generate_radius_surface_points`` and
the batched ``evaluate_placements``) so the numbers here are computed by the SAME
code as the dataset -- the only difference is we evaluate a single pose (batch of
1) that you place yourself, rather than the random batch sampler.

The viewer shows the two RAW quantities with nothing applied (no ``-0.03``, no
``/margin``, no clip, no ``/pi``):

    * raw clearance distance  -- ``dist`` from evaluate_placements (env-normalized
                                 units).  This is the source of ``speed_dist``,
                                 which preprocess later turns into
                                 ``clip((dist-0.03)/margin, offset/margin, 1)``.
    * raw min angle (radians) -- ``min_angle`` from evaluate_placements.  This is
                                 the source of ``speed_angle`` (= angle/pi).

For reference (clearly labelled FYI) it also shows the collision flag and the
post-transform speed_dist / speed_angle as they would land in the .npy files.

Run from the nested ntrl-demo root:

    python visualize_ground_truth_speed.py --dataPath datasets/3dshape/rectangle_env1

Requires ``viser`` and the same torch/igl stack as preprocess_obj.py.
"""

import sys
sys.path.append('.')

import os
import json
import time
import argparse

import numpy as np
import torch
import viser

from dataprocessing.preprocess_obj import (
    load_obj,
    tetrahedralize_shape,
    generate_radius_surface_points,
    evaluate_placements,
    sample_surface_points,
    DEFAULT_TET_SWITCHES,
    DEFAULT_ENV_POINTS,
)
# Reuse visual_pose's gizmo / scene helpers so the viewer behaves identically.
from visual_pose import (
    _resolve_obj,
    quat_to_rotvec,
    add_environment,
    _speed_color,
)

HTTP_PORT = 8080
DIM = 6
TWO_PI = 2.0 * np.pi


def _segment_quad(p0, p1, width=0.004):
    """Thin quad (two triangles) approximating the segment p0->p1.

    Built only from add_mesh_simple-able geometry (the one viser scene primitive
    the rest of this codebase relies on) so it works on any viser version.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None, None
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0] / n) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(d, ref)
    perp = perp / (np.linalg.norm(perp) + 1e-9) * width
    V = np.array([p0 - perp, p0 + perp, p1 + perp, p1 - perp], dtype=np.float64)
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return V, F


def _octahedron(center, r=0.008):
    """Small octahedron marker mesh centered at ``center`` (add_mesh_simple-able)."""
    c = np.asarray(center, float)
    V = c + r * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                          [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [0, 4, 2], [0, 5, 2], [2, 5, 1], [1, 5, 3], [3, 5, 0]], dtype=np.int64)
    return V, F


def gizmo_to_config_radians(giz):
    """Read a transform-control handle into a raw 6-D config (rotvec in RADIANS).

    ``evaluate_placements`` expects raw radians for the rotvec part (it is the
    code that, in preprocess, runs *before* the ``/2pi`` normalization), so unlike
    ``visual_pose.gizmo_to_cfg`` we do NOT divide by 2*pi here.
    """
    cfg = np.zeros(DIM)
    cfg[0:3] = np.asarray(giz.position, dtype=np.float64)
    cfg[3:6] = quat_to_rotvec(giz.wxyz)          # radians (axis-angle)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Drag a pose in viser and read the ground-truth clearance / '
                    'angle that feed speed_dist / speed_angle.')
    parser.add_argument('--dataPath', default='datasets/3dshape/rectangle_env1',
                        help='Dataset dir with meta.json (shape/env meshes, scale, '
                             'margin, offset) and env.npy (env surface points).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=HTTP_PORT)
    # These are not stored in meta.json; defaults mirror preprocess_obj.py so the
    # shape tetrahedra and radial shells match what the dataset was built with.
    parser.add_argument('--num_radius_points', type=int, default=1000,
                        help='Shape-surface points binned by radius (matches preprocess).')
    parser.add_argument('--radius_bins', type=int, default=10,
                        help='Number of radial bins for the shape-surface points.')
    parser.add_argument('--tet_switches', default=DEFAULT_TET_SWITCHES,
                        help='tetgen switches used to tetrahedralize the shape.')
    parser.add_argument('--num_env_points', type=int, default=DEFAULT_ENV_POINTS,
                        help='Only used if env.npy is absent (re-samples the env surface).')
    args = parser.parse_args()

    device = args.device

    # ── Read dataset metadata (same fields visual_pose / preprocess use) ──
    with open(os.path.join(args.dataPath, 'meta.json')) as f:
        meta = json.load(f)
    env_scale = float(meta['env_scale'])
    env_center = np.asarray(meta['env_center'], dtype=np.float64)
    shape_scale = float(meta['shape_scale'])
    margin = float(meta['margin'])
    offset = float(meta['offset'])

    # ── Environment mesh (for display) + the exact surface points used as data ──
    V_env, F_env, names_env = load_obj(_resolve_obj(meta['env_obj'], args.dataPath))
    V_env_n = np.ascontiguousarray((V_env - env_center) / env_scale, dtype=np.float64)
    wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
    wall_F = np.ascontiguousarray(F_env[wall_mask], dtype=np.int64)
    obst_F = np.ascontiguousarray(F_env[~wall_mask], dtype=np.int64)

    # Prefer the env point cloud the dataset was actually built from (env.npy) so
    # the clearance / angle queries see exactly the same env points.  Fall back to
    # re-sampling with the same routine preprocess uses if env.npy is missing.
    env_npy = os.path.join(args.dataPath, 'env.npy')
    if os.path.exists(env_npy):
        env_points = np.load(env_npy).astype(np.float32)
        print('Loaded env points from {}: {}'.format(env_npy, env_points.shape))
    else:
        sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
        env_points = sample_surface_points(
            V_env_n, F_env[sample_mask], args.num_env_points).astype(np.float32)
        print('env.npy absent; re-sampled {} env points'.format(env_points.shape[0]))
    env_t = torch.tensor(env_points, dtype=torch.float32)

    # ── Shape: normalize into the env frame and tetrahedralize (== preprocess) ──
    V_sh, F_sh, _ = load_obj(_resolve_obj(meta['shape_obj'], args.dataPath))
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    V_sh_local = (V_sh - shape_center) / env_scale * shape_scale
    shape_V = np.ascontiguousarray(V_sh_local, dtype=np.float64)
    shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)

    TV, TT, TF = tetrahedralize_shape(V_sh_local, F_sh, switches=args.tet_switches)
    tet_verts_local = torch.tensor(TV[TT], dtype=torch.float32)     # (K,4,3)
    face_verts_local = torch.tensor(TV[TF], dtype=torch.float32)    # (F,3,3)
    print('Tetrahedralized shape: {} tets, {} boundary triangles'.format(
        TT.shape[0], TF.shape[0]))

    radius_points, radius_bins = generate_radius_surface_points(
        V_sh_local, F_sh, args.num_radius_points, args.radius_bins)

    # ── Ground-truth query for a single placement (batch of 1) ──
    def probe(cfg):
        """Run the batched evaluator on one pose; return raw quantities.

        Returns (is_free, dist, min_angle, normal6, shape_pt, env_pt) for the
        single config -- identical computation to the dataset, but B = 1.
        """
        configs = torch.tensor(cfg, dtype=torch.float32).unsqueeze(0)   # (1,6) radians
        free, dist, angle, normal, shp, envp = evaluate_placements(
            configs, tet_verts_local, face_verts_local, env_t,
            radius_points, radius_bins, device, return_angle_pts=True)
        return (bool(free[0].item()), float(dist[0].item()), float(angle[0].item()),
                normal[0].numpy(), shp[0].numpy(), envp[0].numpy())

    # ── viser scene (mirrors visual_pose) ──
    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    server.scene.set_up_direction('+y')
    add_environment(server, V_env_n, obst_F, wall_F)

    pose_ctrl = server.scene.add_transform_controls(
        '/pose', scale=0.25, position=(0.0, 0.0, 0.0))
    server.scene.add_mesh_simple(
        '/pose/shape', vertices=shape_V, faces=shape_F,
        color=(80, 140, 240), opacity=0.95, flat_shading=True, side='double')

    # GUI readouts: RAW quantities first (the ask), then FYI transformed values.
    txt_dist = server.gui.add_text('RAW clearance dist (->speed_dist)',
                                   initial_value='', disabled=True)
    txt_angle = server.gui.add_text('RAW min angle rad (->speed_angle)',
                                    initial_value='', disabled=True)
    txt_collision = server.gui.add_text('collision-free?', initial_value='', disabled=True)
    txt_sd_fyi = server.gui.add_text('FYI speed_dist (clip)', initial_value='', disabled=True)
    txt_sa_fyi = server.gui.add_text('FYI speed_angle (/pi)', initial_value='', disabled=True)
    txt_normal = server.gui.add_text('clearance normal (6)', initial_value='', disabled=True)
    txt_cfg = server.gui.add_text('config (xyz | rotvec rad)', initial_value='', disabled=True)

    def update(_=None):
        cfg = gizmo_to_config_radians(pose_ctrl)
        free, dist, angle, normal, shp, envp = probe(cfg)



        #angle = (angle/np.pi)
    

        #angle = np.sin(angle)
       
        # RAW: nothing applied.
        txt_dist.value = '{:.5f}'.format(dist)
        txt_angle.value = '{:.5f}'.format(angle)
        txt_collision.value = 'FREE' if free else 'COLLISION'

        # FYI: exactly how preprocess_obj turns these into the saved npy values.
        speed_dist = dist
        speed_angle = angle


        speed_dist = np.clip(speed_dist / 0.05,
                          a_min=0.001 / 0.05, a_max=1.0)


        angle_max = 1*torch.pi/18
        angle_min = 0.001
        speed_angle = np.clip(speed_angle/angle_max, a_min=angle_min / angle_max, a_max=1)
        speed_angle = speed_angle**2 *(2-speed_angle)**2

        txt_sd_fyi.value = '{:.5f}'.format(speed_dist)
        txt_sa_fyi.value = '{:.5f}'.format(speed_angle)

        txt_normal.value = '[' + ', '.join('{:+.3f}'.format(v) for v in normal) + ']'
        txt_cfg.value = '({:.3f}, {:.3f}, {:.3f} | {:+.3f}, {:+.3f}, {:+.3f})'.format(
            cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5])

        # Recolor shape by raw clearance distance, and red collision when overlapping.
        color = (220, 40, 40) if not free else _speed_color(dist / max(margin, 1e-6))
        server.scene.add_mesh_simple(
            '/pose/shape', vertices=shape_V, faces=shape_F,
            color=color, opacity=0.95, flat_shading=True, side='double')

        # Draw the min-angle pair (shape-surface point -> env point): a red segment
        # with a red marker at the shape point and an orange marker at the env point.
        # Built from add_mesh_simple geometry only (works on any viser version).
        # NOTE: shp/envp are already in the world (env-normalized) frame, so these
        # nodes live at the scene root -- NOT under /pose, which carries the gizmo
        # transform (parenting them there would apply the pose twice).
        segV, segF = _segment_quad(shp, envp)
        if segV is not None:
            server.scene.add_mesh_simple(
                '/min_angle', vertices=segV, faces=segF,
                color=(255, 0, 0), opacity=1.0, flat_shading=True, side='double')
        mV, mF = _octahedron(shp)
        server.scene.add_mesh_simple('/min_angle_shape_pt', vertices=mV, faces=mF,
                                     color=(255, 0, 0), flat_shading=True, side='double')
        mV, mF = _octahedron(envp)
        server.scene.add_mesh_simple('/min_angle_env_pt', vertices=mV, faces=mF,
                                     color=(255, 160, 0), flat_shading=True, side='double')

    pose_ctrl.on_update(update)
    update()

    print('\nServing viser at http://0.0.0.0:{}  —  open this on your host PC'.format(
        args.port))
    print('Drag the gizmo to place the shape. RAW clearance / angle are the '
          'speed_dist / speed_angle sources. Ctrl-C to stop.\n')
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
