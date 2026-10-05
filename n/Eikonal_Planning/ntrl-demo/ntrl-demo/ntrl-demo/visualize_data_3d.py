"""Browse the sampled 3-D training configurations one at a time in viser.

Companion to ``evaluate_training_3d.py``.  Where that script renders MPPI
*episodes* (a start/goal trajectory sweep), this one steps through the raw
*sampled configurations* written by ``dataprocessing/preprocess_obj.py`` and,
for each single SE(3) placement, shows its labelled ``speed_angles`` and
``speed_dists`` (and the combined ``speed``).

Data layout (all under ``--dataPath``):

    sampled_points.npy  (N, 12)  pair per row [x0(6), x1(6)]   rotvec stored /2pi
    speed_angles.npy    (N, 2)   per-endpoint angle speed   [a0, a1]  in [0,1]
    speed_dists.npy     (N, 2)   per-endpoint clearance speed [d0, d1] in [0,1]
    speed.npy           (N, 2)   per-endpoint combined speed (mean of the two)
    env.npy             (M, 3)   environment surface points
    meta.json                    shape_obj, env_obj, env_scale, env_center, ...

Each *configuration* is one 6-D endpoint, so the N pairs flatten into 2N single
placements: config ``2*i``   = x0 of pair i, config ``2*i+1`` = x1 of pair i.
The viewer draws that one placement as its actual transformed shape mesh inside
the environment mesh (obstacles solid grey, walls translucent), exactly like
``evaluate_training_3d.py``, and reports the three speed values in the GUI.

Navigation (viser GUI panel, browse to http://<server-ip>:8080):
    * "Next" / "Prev" buttons step one configuration at a time
    * an index slider jumps anywhere
    * the speed_angles / speed_dists / speed readouts update with each config

Run from the nested ntrl-demo root:

    python visualize_data_3d.py --dataPath datasets/3dshape/rectangle_env1

Requires ``viser`` (``pip install viser``).
"""

import sys
sys.path.append('.')

import os
import json
import time
import argparse

import numpy as np
import viser

from dataprocessing.preprocess_obj import load_obj, _rotvec_to_matrix_np

HTTP_PORT = 8080
DIM = 6
TWO_PI = 2.0 * np.pi


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_obj(meta_path, data_path):
    """Resolve an OBJ path from meta.json, falling back to a local copy.

    ``meta.json`` records absolute paths from the machine the data was generated
    on (e.g. ``/workspace/...``), which need not exist here.  If the recorded
    path is missing, look for a file of the same basename next to the dataset
    directory (``datasets/3dshape/<name>.obj``).
    """
    if os.path.exists(meta_path):
        return meta_path
    base = os.path.basename(meta_path)
    cand = os.path.join(os.path.dirname(os.path.normpath(data_path)), base)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        "OBJ '{}' not found (meta path '{}' missing, no local copy at '{}')".format(
            base, meta_path, cand))


def _placed_mesh(shape_V, cfg_rad):
    """Transform the shape's local mesh by a config (x,y,z, rotvec in radians)."""
    R = _rotvec_to_matrix_np(cfg_rad[3:6])
    return shape_V @ R.T + cfg_rad[0:3]


def _speed_color(s):
    """Map a speed value s in [0, 1] to an RGB tuple of ints (viridis)."""
    try:
        import matplotlib.cm as cm
        r, g, b, _ = cm.get_cmap('viridis')(float(np.clip(s, 0.0, 1.0)))
    except Exception:
        # Fallback: blue (slow) -> yellow (fast) ramp if matplotlib is missing.
        s = float(np.clip(s, 0.0, 1.0))
        r, g, b = s, s, 1.0 - s
    return (int(r * 255), int(g * 255), int(b * 255))


def add_environment(server, env_V, obst_F, wall_F):
    """Draw the environment as its actual triangle mesh (static scene)."""
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
        description='Browse sampled 3-D configurations one at a time in viser, '
                    'showing each placement\'s speed_angles and speed_dists.')
    parser.add_argument('--dataPath', default='datasets/3dshape/rectangle_env1',
                        help='Directory holding the sampled data (sampled_points.npy, '
                             'speed_angles.npy, speed_dists.npy, env.npy, meta.json).')
    parser.add_argument('--max_speed', type=float, default=0.05,
                        help='Only load configurations whose (combined) speed is '
                             'below this threshold.')
    parser.add_argument('--num', type=int, default=500,
                        help='How many configurations to load for browsing '
                             '(from the start; <=0 loads all).')
    parser.add_argument('--stride', type=int, default=1,
                        help='Take every Nth configuration (1 = consecutive endpoints).')
    parser.add_argument('--port', type=int, default=HTTP_PORT,
                        help='HTTP port for the viser server.')
    args = parser.parse_args()

    data_path = args.dataPath

    # ── Load the sampled configurations and their per-endpoint speed labels ──
    pairs = np.load(os.path.join(data_path, 'sampled_points.npy'))     # (N, 12)
    speed_angles = np.load(os.path.join(data_path, 'speed_angles.npy'))  # (N, 2)
    speed_dists = np.load(os.path.join(data_path, 'speed_dists.npy'))   # (N, 2)
    speed_path = os.path.join(data_path, 'speed.npy')
    speed = np.load(speed_path) if os.path.exists(speed_path) else None  # (N, 2) or None

    # Flatten the N pairs into 2N single endpoints: config 2*i = x0, 2*i+1 = x1.
    # rotvec is de-normalized back to radians here (stored /2pi by preprocess_obj).
    cfgs = pairs.reshape(-1, 2, DIM).reshape(-1, DIM).astype(np.float64).copy()  # (2N, 6)
    cfgs[:, 3:6] *= TWO_PI
    s_ang = speed_angles.reshape(-1).astype(np.float64)                # (2N,)
    s_dist = speed_dists.reshape(-1).astype(np.float64)               # (2N,)
    s_all = speed.reshape(-1).astype(np.float64) if speed is not None else None

    # Keep only the slow configurations (combined speed below the threshold).
    # Fall back to the mean of the angle/dist speeds if speed.npy is absent.
    speed_for_filter = s_all if s_all is not None else 0.5 * (s_ang + s_dist)
    keep = np.nonzero(speed_for_filter < args.max_speed)[0]              # flat indices

    # Subsample for browsing (the matching set can still be large).
    sel = keep[::max(int(args.stride), 1)]
    if args.num > 0:
        sel = sel[:args.num]
    cfgs, s_ang, s_dist = cfgs[sel], s_ang[sel], s_dist[sel]
    s_all = s_all[sel] if s_all is not None else None
    n_cfg = len(cfgs)
    print('Loaded {} configurations from {} (speed < {}, stride={}, of {} '
          'total endpoints, {} below threshold)'.format(
              n_cfg, data_path, args.max_speed, args.stride,
              len(pairs) * 2, len(keep)))

    # ── Reconstruct the shape + environment meshes in the normalized frame ──
    with open(os.path.join(data_path, 'meta.json')) as f:
        meta = json.load(f)
    env_scale = float(meta['env_scale'])
    env_center = np.asarray(meta['env_center'], dtype=np.float64)
    shape_scale = float(meta['shape_scale'])

    shape_obj = _resolve_obj(meta['shape_obj'], data_path)
    env_obj = _resolve_obj(meta['env_obj'], data_path)

    V_sh, F_sh, _ = load_obj(shape_obj)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                                   dtype=np.float64)
    shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)

    V_env, F_env, names_env = load_obj(env_obj)
    V_env_n = np.ascontiguousarray((V_env - env_center) / env_scale, dtype=np.float64)
    wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
    wall_F = np.ascontiguousarray(F_env[wall_mask], dtype=np.int64)
    obst_F = np.ascontiguousarray(F_env[~wall_mask], dtype=np.int64)

    # ── Launch the viser scene ──
    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    server.scene.set_up_direction('+y')
    add_environment(server, V_env_n, obst_F, wall_F)

    # GUI: navigation + speed readouts.
    gui_idx = server.gui.add_slider('Config index', min=0, max=max(n_cfg - 1, 0),
                                    step=1, initial_value=0)
    with server.gui.add_folder('Navigate'):
        btn_prev = server.gui.add_button('◀ Prev')
        btn_next = server.gui.add_button('Next ▶')
    txt_which = server.gui.add_text('Endpoint', initial_value='', disabled=True)
    txt_ang = server.gui.add_text('speed_angles', initial_value='', disabled=True)
    txt_dist = server.gui.add_text('speed_dists', initial_value='', disabled=True)
    txt_speed = server.gui.add_text('speed', initial_value='', disabled=True)
    txt_cfg = server.gui.add_text('config (xyz|rot°)', initial_value='', disabled=True)

    current = []   # scene handles for the currently shown shape mesh

    def show(i):
        i = int(np.clip(i, 0, n_cfg - 1))
        for h in current:
            h.remove()
        current.clear()

        cfg = cfgs[i]
        Vp = _placed_mesh(shape_V, cfg)
        # Colour the placement by its clearance speed (viridis: blue slow, yellow fast).
        current.append(server.scene.add_mesh_simple(
            '/config/shape', vertices=Vp, faces=shape_F,
            color=_speed_color(s_dist[i]), opacity=0.95,
            flat_shading=True, side='double'))

        # Update the GUI readouts.  Each endpoint came from pair (i//2), x0 or x1.
        endpoint = 'x0' if (sel[i] % 2 == 0) else 'x1'
        txt_which.value = '{} of pair {}'.format(endpoint, sel[i] // 2)
        txt_ang.value = '{:.4f}'.format(s_ang[i])
        txt_dist.value = '{:.4f}'.format(s_dist[i])
        txt_speed.value = '{:.4f}'.format(s_all[i]) if s_all is not None else 'n/a'
        deg = np.degrees(cfg[3:6])
        txt_cfg.value = '({:.3f}, {:.3f}, {:.3f}) | ({:.1f}, {:.1f}, {:.1f})'.format(
            cfg[0], cfg[1], cfg[2], deg[0], deg[1], deg[2])

    @gui_idx.on_update
    def _(_):
        show(gui_idx.value)

    @btn_next.on_click
    def _(_):
        gui_idx.value = int(np.clip(gui_idx.value + 1, 0, n_cfg - 1))

    @btn_prev.on_click
    def _(_):
        gui_idx.value = int(np.clip(gui_idx.value - 1, 0, n_cfg - 1))

    if n_cfg:
        show(0)

    print('\nServing viser at http://0.0.0.0:{}  —  open this on your host PC'.format(
        args.port))
    print("Use 'Next'/'Prev' or the slider to browse configurations. Ctrl-C to stop.\n")
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
