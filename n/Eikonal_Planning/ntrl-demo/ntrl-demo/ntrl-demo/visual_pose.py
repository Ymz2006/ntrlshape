"""Interactively probe the learned speed/metric field at a pose you drag in viser.

Load a trained 3-D shape-planner checkpoint (``.pt``) and drag the shape around
the environment with a viser transform gizmo.  For whatever pose you place it at,
the viewer reports what the model thinks about that configuration:

    * tau          -- the network travel time tau(query, goal)
    * speed        -- 1 / ||d tau / d config||  (the eikonal speed at the query;
                      same quantity as model_function_metric.Function.Speed)
    * d tau/d cfg  -- the full 6-vector gradient of tau w.r.t. the query config
                      (x, y, z, rx, ry, rz), rotvec in /2pi units like training
    * |pos part|   -- magnitude of the first 3 gradient elements (translation)
    * |rot part|   -- magnitude of the last 3 gradient elements (rotation)

The gradient is the same computation you flagged in model_function_metric.py:

    tau, w, Xp = network.out(points)
    dtau = gradient(tau, Xp)        # autograd.grad(tau, Xp)

evaluated at a single (query, goal) pair, with dtau[:, :6] = d tau / d query.

tau is a function of BOTH the query pose and a goal pose, so a goal gizmo is also
provided (green, translucent).  By the eikonal equation ||grad_x tau(x, goal)||
is the slowness at x independent of the goal, so the speed reading barely depends
on where you put the goal -- move it to convince yourself, or just leave it.

Run from the nested ntrl-demo root (defaults to the latest checkpoint, like
evaluate_training_3d.py -- pass --pt to probe a specific one):

    python visual_pose.py --dataPath datasets/3dshape/rectangle_env1
    python visual_pose.py --pt ./Experiments/3dshape/.../Model_Epoch_XXXX.pt \
                          --dataPath datasets/3dshape/rectangle_env1

Requires ``viser`` and the same torch/igl stack as evaluate_training_3d.py.
"""

import sys
sys.path.append('.')

import os
import json
import time
import argparse
from glob import glob

import numpy as np
import torch
import viser

from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import load_obj

HTTP_PORT = 8080
DIM = 6
TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────────────
# Geometry / config helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_obj(meta_path, data_path):
    """Resolve an OBJ path from meta.json, falling back to a local copy next to
    the dataset directory (meta records absolute paths from another machine)."""
    if os.path.exists(meta_path):
        return meta_path
    cand = os.path.join(os.path.dirname(os.path.normpath(data_path)),
                        os.path.basename(meta_path))
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        "OBJ '{}' not found (meta '{}' missing, no local copy at '{}')".format(
            os.path.basename(meta_path), meta_path, cand))


def quat_to_rotvec(wxyz):
    """viser orientation quaternion (w, x, y, z) -> rotation vector (radians).

    Returns the minimal axis-angle representation (magnitude wrapped to [0, pi]),
    so that preprocess_obj.rotvec_to_matrix(rotvec) reproduces the same rotation
    the gizmo applies to the mesh.
    """
    w, x, y, z = np.asarray(wxyz, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.zeros(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))      # in [0, 2pi)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros(3)                              # no rotation
    axis = np.array([x, y, z]) / s
    if angle > np.pi:                                   # take the short way round
        angle -= TWO_PI
    return axis * angle


def gizmo_to_cfg(giz):
    """Read a transform-control handle into a 6-D config (rotvec stored /2pi)."""
    cfg = np.zeros(DIM)
    cfg[0:3] = np.asarray(giz.position, dtype=np.float64)
    cfg[3:6] = quat_to_rotvec(giz.wxyz) / TWO_PI
    return cfg


def _speed_color(s):
    """Map a speed value s in [0, 1] to an RGB tuple of ints (viridis)."""
    try:
        import matplotlib.cm as cm
        r, g, b, _ = cm.get_cmap('viridis')(float(np.clip(s, 0.0, 1.0)))
    except Exception:
        s = float(np.clip(s, 0.0, 1.0))
        r, g, b = s, s, 1.0 - s
    return (int(r * 255), int(g * 255), int(b * 255))


def add_environment(server, env_V, obst_F, wall_F):
    """Draw the environment mesh (obstacles solid grey, walls translucent)."""
    if len(obst_F) > 0:
        server.scene.add_mesh_simple(
            '/env/obstacles', vertices=env_V, faces=obst_F,
            color=(150, 150, 150), opacity=1.0, flat_shading=True, side='double')
    if len(wall_F) > 0:
        server.scene.add_mesh_simple(
            '/env/walls', vertices=env_V, faces=wall_F,
            color=(173, 216, 230), opacity=0.15, flat_shading=True, side='double')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Drag a pose in viser and read the model speed + d tau/d config.')
    parser.add_argument('--pt', default=None,
                        help='Path to the trained checkpoint (.pt) to probe. '
                             'Defaults to latest.pt under --modelPath (then the '
                             'newest Model_Epoch_*.pt), like evaluate_training_3d.py.')
    parser.add_argument('--dataPath', default='datasets/3dshape/rectangle_env1',
                        help='Dataset dir with meta.json (shape/env meshes, scale).')
    parser.add_argument('--modelPath', default='./Experiments/3dshape',
                        help='Model root (only used to seed the Model object).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--port', type=int, default=HTTP_PORT)
    args = parser.parse_args()

    device = args.device

    # ── Load the model ──
    womodel = md.Model(args.modelPath, args.dataPath, DIM, [0.0] * DIM, device=device)

    # Prefer an explicit --pt; otherwise default to the always-current latest.pt
    # written every epoch by training, falling back to the most recent
    # timestamped Model_Epoch_*.pt checkpoint (mirrors evaluate_training_3d.py).
    pt = args.pt
    if pt is None:
        latest = os.path.join(args.modelPath, 'latest.pt')
        if os.path.exists(latest):
            pt = latest
        else:
            ckpts = sorted(glob(os.path.join(args.modelPath, '*', 'Model_Epoch_*.pt')))
            if not ckpts:
                raise FileNotFoundError(
                    'No --pt given, no latest.pt and no checkpoints under '
                    '{}/*/Model_Epoch_*.pt'.format(args.modelPath))
            pt = ckpts[-1]

    print('Loading checkpoint: {}'.format(pt))
    womodel.load(pt)
    womodel.network.eval()

    # ── Reconstruct shape + environment meshes in the normalized frame ──
    with open(os.path.join(args.dataPath, 'meta.json')) as f:
        meta = json.load(f)
    env_scale = float(meta['env_scale'])
    env_center = np.asarray(meta['env_center'], dtype=np.float64)
    shape_scale = float(meta['shape_scale'])

    V_sh, F_sh, _ = load_obj(_resolve_obj(meta['shape_obj'], args.dataPath))
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                                   dtype=np.float64)
    shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)

    V_env, F_env, names_env = load_obj(_resolve_obj(meta['env_obj'], args.dataPath))
    V_env_n = np.ascontiguousarray((V_env - env_center) / env_scale, dtype=np.float64)
    wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
    wall_F = np.ascontiguousarray(F_env[wall_mask], dtype=np.int64)
    obst_F = np.ascontiguousarray(F_env[~wall_mask], dtype=np.int64)

    # ── Model query: tau, speed, and the full d tau/d config at (query, goal) ──
    def probe(cfg_q, cfg_g):
        """Return (tau, speed, grad6, pos_mag, rot_mag, pos_speed, rot_speed).

        grad6 = d tau / d query_config (the first DIM coords), exactly the
        Speed()/Loss() computation: out() then autograd grad of tau w.r.t. input.
        speed     = 1 / ||grad6||         (full 6-D eikonal speed)
        pos_speed = 1 / ||grad6[0:3]||    (translation-only speed)
        rot_speed = 1 / ||grad6[3:6]||    (rotation-only speed)
        """
        x = np.concatenate([cfg_q, cfg_g]).astype(np.float32)[None]   # (1, 2*DIM)
        Xp = torch.from_numpy(x).to(device)
        tau, _w, coords = womodel.network.out(Xp)        # coords carries requires_grad
        g_full = torch.autograd.grad(tau.sum(), coords)[0][0]         # (2*DIM,)
        g = g_full[:DIM].detach().cpu().numpy().astype(np.float64)    # d tau/d query
        speed = 1.0 / (float(np.linalg.norm(g)) + 1e-12)
        pos_mag = float(np.linalg.norm(g[0:3]))
        rot_mag = float(np.linalg.norm(g[3:6]))
        pos_speed = 1.0 / (pos_mag + 1e-12)
        rot_speed = 1.0 / (rot_mag + 1e-12)
        return (float(tau.item()), speed, g, pos_mag, rot_mag, pos_speed, rot_speed)

    # ── viser scene ──
    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    server.scene.set_up_direction('+y')
    add_environment(server, V_env_n, obst_F, wall_F)

    # Query gizmo (drag this) + goal gizmo (translucent green, optional).
    query_ctrl = server.scene.add_transform_controls(
        '/query', scale=0.25, position=(0.0, 0.0, 0.0))
    server.scene.add_mesh_simple(
        '/query/shape', vertices=shape_V, faces=shape_F,
        color=(80, 140, 240), opacity=0.95, flat_shading=True, side='double')

    goal_ctrl = server.scene.add_transform_controls(
        '/goal', scale=0.2, position=(0.25, 0.0, 0.0))
    server.scene.add_mesh_simple(
        '/goal/shape', vertices=shape_V, faces=shape_F,
        color=(30, 180, 30), opacity=0.4, flat_shading=True, side='double')

    # GUI readouts.
    txt_speed = server.gui.add_text('speed (1/|grad|)', initial_value='', disabled=True)
    txt_pos_speed = server.gui.add_text('pos speed (1/|grad xyz|)', initial_value='', disabled=True)
    txt_rot_speed = server.gui.add_text('rot speed (1/|grad rxyz|)', initial_value='', disabled=True)
    txt_tau = server.gui.add_text('tau(query, goal)', initial_value='', disabled=True)
    txt_pos_mag = server.gui.add_text('|grad pos (xyz)|', initial_value='', disabled=True)
    txt_rot_mag = server.gui.add_text('|grad rot (rxyz)|', initial_value='', disabled=True)
    txt_grad = server.gui.add_text('d tau/d config (6)', initial_value='', disabled=True)
    txt_cfg = server.gui.add_text('query (xyz | rot/2pi)', initial_value='', disabled=True)

    def update(_=None):
        cfg_q = gizmo_to_cfg(query_ctrl)
        cfg_g = gizmo_to_cfg(goal_ctrl)
        tau, speed, g, pmag, rmag, pspeed, rspeed = probe(cfg_q, cfg_g)

        txt_speed.value = '{:.4f}'.format(speed)
        txt_pos_speed.value = '{:.4f}'.format(pspeed)
        txt_rot_speed.value = '{:.4f}'.format(rspeed)
        txt_tau.value = '{:.4f}'.format(tau)
        txt_pos_mag.value = '{:.4f}'.format(pmag)
        txt_rot_mag.value = '{:.4f}'.format(rmag)
        txt_grad.value = '[' + ', '.join('{:+.3f}'.format(v) for v in g) + ']'
        txt_cfg.value = '({:.3f}, {:.3f}, {:.3f} | {:+.3f}, {:+.3f}, {:+.3f})'.format(
            cfg_q[0], cfg_q[1], cfg_q[2], cfg_q[3], cfg_q[4], cfg_q[5])

        # Recolor the query shape by its speed (viridis: blue slow, yellow fast).
        server.scene.add_mesh_simple(
            '/query/shape', vertices=shape_V, faces=shape_F,
            color=_speed_color(speed), opacity=0.95, flat_shading=True, side='double')

    query_ctrl.on_update(update)
    goal_ctrl.on_update(update)
    update()

    print('\nServing viser at http://0.0.0.0:{}  —  open this on your host PC'.format(
        args.port))
    print("Drag the BLUE gizmo (query pose). GREEN is the goal. Ctrl-C to stop.\n")
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
