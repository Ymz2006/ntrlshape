"""Single-episode (non-batched) evaluation for the 3-D shape planner.

3-D (SE(3)) counterpart of ``evaluate_training.py``.  Same MPPI rollout, but the
configuration space is ``(x, y, z, rx, ry, rz)`` (dim = 6, rotvec stored
normalized by 2*pi -- exactly what ``dataprocessing/preprocess_obj.py`` writes).

Each episode is rendered with plotly using the same visualization choices as
``preprocess_obj.visual_training``: the environment is drawn as sampled grey
points, the walls as a translucent mesh, and the moving shape is drawn as its
actual transformed ``Mesh3d`` at every trajectory waypoint (colored by progress
along the path), with the start pose in red and the goal pose in green.  The
view is oriented y-up.

All plots are saved as HTML files under ./output_3d/ and served over HTTP on
port 8080 so they can be viewed from a remote host:

    http://<server-ip>:8080

Run from the nested ntrl-demo root:

    python evaluate_training_3d.py
"""

import sys
sys.path.append('.')

import os
import json
import argparse
import http.server
import socketserver
from glob import glob
from timeit import default_timer as timer

import numpy as np
import torch
import plotly.graph_objects as go

from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import load_obj, _rotvec_to_matrix_np

COLLISION_THRESH = 0.001
HTTP_PORT = 8080
DIM = 6

parser = argparse.ArgumentParser(
    description='Single-episode evaluation for the 3-D shape planner.')
parser.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1',
                    help='Directory holding the test data (sampled_points.npy, '
                         'speed.npy, env.npy, meta.json) used as start/goal pairs.')
parser.add_argument('--out', default='./output_3d',
                    help='Output directory for the rendered HTML plots (also the '
                         'web root served over HTTP).')
args = parser.parse_args()

OUTPUT_DIR = args.out

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clear stale artifacts from any previous run so old episode_*.html (with a
# different success/fail suffix) don't linger alongside this run's output.
for _stale in (glob(os.path.join(OUTPUT_DIR, 'episode_*.html'))
               + glob(os.path.join(OUTPUT_DIR, 'summary_*.html'))
               + glob(os.path.join(OUTPUT_DIR, 'success_rate.txt'))):
    os.remove(_stale)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def check_trajectory_collision(traj, env_pts):
    """Return True if ANY waypoint position comes within COLLISION_THRESH of ANY
    environment point.  Only x, y, z (first 3 dims) are compared."""
    waypoints = torch.cat(traj, dim=0)[:, :3].cuda()
    diff = waypoints.unsqueeze(1) - env_pts.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)
    return dists.min().item() < COLLISION_THRESH


def MPPI(womodel, XP, dim):
    """Run MPPI for a single start/goal pair.

    Returns the recorded waypoints, the iteration count, and a success flag.
    """
    found_path = False
    steps = 200
    sample_num = 50
    horizon = 5

    dP_prior = torch.zeros((1, dim)).cuda()

    point0 = []
    point0.append(XP[:, 0:dim].clone())

    iter = 0
    for iter in range(steps):
        XP_tmp = XP.clone()
        XP_tmp = XP_tmp.unsqueeze(0).repeat(sample_num, horizon, 1)

        dP = 0.015 * torch.normal(0, 1, size=(sample_num, 1, dim), dtype=torch.float32, device='cuda') \
            + 0.015 * torch.normal(0, 1, size=(sample_num, horizon, dim), dtype=torch.float32, device='cuda')
        dP = dP + 2 * dP_prior
        dP_norm = torch.norm(dP, dim=2, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=0.015) / 0.015)
        dP_cumsum = torch.cumsum(dP, dim=1)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + dP_cumsum

        indices = [0, -1]
        cost = womodel.function.TravelTimes(XP_tmp[:, indices, :].reshape(-1, dim * 2))
        cost = cost.reshape(-1, 2)
        cost = 10 * cost[:, 0] + cost[:, 1]

        weight = torch.softmax(-50 * cost, dim=0)
        dP_prior = (weight @ dP[:, 0, :])

        XP[:, 0:dim] = dP_prior + XP[:, 0:dim]

        dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim])
        point0.append(XP[:, 0:dim].clone())

        if dis < 0.01:
            found_path = True
            break

    point0.append(XP[:, dim:dim * 2].clone())
    return point0, iter, found_path


def _placed_mesh(shape_V, cfg):
    """Transform the shape's local mesh by a config (x,y,z, rotvec in radians)."""
    R = _rotvec_to_matrix_np(cfg[3:6])
    return shape_V @ R.T + cfg[0:3]


def visual_trajectory(waypoints, shape_V, shape_F, env_points, wall_V, wall_F,
                      begin_cfg, end_cfg, save_path):
    """Render one MPPI trajectory of shape meshes, plotly, y-up.

    Mirrors ``preprocess_obj.visual_training``: translucent walls, grey env
    points, and the moving shape as a combined ``Mesh3d`` colored (here) by
    progress along the path; start pose red, goal pose green.

    Parameters
    ----------
    waypoints : (T, 6)  poses [x, y, z, rx, ry, rz] (rotvec in radians)
    """
    traces = []

    # ── Walls: translucent mesh ──
    if wall_V is not None and wall_F is not None and len(wall_F) > 0:
        traces.append(go.Mesh3d(
            x=wall_V[:, 0], y=wall_V[:, 1], z=wall_V[:, 2],
            i=wall_F[:, 0], j=wall_F[:, 1], k=wall_F[:, 2],
            color='lightblue', opacity=0.15, name='walls', flatshading=True))

    # ── Environment: sampled surface points ──
    if len(env_points) > 0:
        sub = env_points[np.random.choice(
            len(env_points), size=min(len(env_points), 8000), replace=False)]
        traces.append(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2], mode='markers',
            name='environment', marker=dict(size=1.5, color='grey', opacity=0.3)))

    # ── Trajectory: shape mesh at every waypoint, colored by progress ──
    T = len(waypoints)
    nv = shape_V.shape[0]
    verts_all, faces_all, inten_all = [], [], []
    for t in range(T):
        verts_all.append(_placed_mesh(shape_V, waypoints[t]))
        faces_all.append(shape_F + t * nv)
        inten_all.append(np.full(nv, t / max(T - 1, 1)))
    if verts_all:
        V = np.concatenate(verts_all, 0)
        Fc = np.concatenate(faces_all, 0)
        inten = np.concatenate(inten_all, 0)
        traces.append(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=Fc[:, 0], j=Fc[:, 1], k=Fc[:, 2],
            intensity=inten, colorscale='Viridis', cmin=0.0, cmax=1.0,
            opacity=0.5, name='trajectory', flatshading=True,
            colorbar=dict(title='path progress')))

    # ── Start (red) and goal (green) poses ──
    for cfg, col, nm in [(begin_cfg, 'red', 'start'), (end_cfg, 'green', 'goal')]:
        if cfg is None:
            continue
        Vp = _placed_mesh(shape_V, cfg)
        traces.append(go.Mesh3d(
            x=Vp[:, 0], y=Vp[:, 1], z=Vp[:, 2],
            i=shape_F[:, 0], j=shape_F[:, 1], k=shape_F[:, 2],
            color=col, opacity=0.9, name=nm, flatshading=True))

    fig = go.Figure(traces)
    fig.update_layout(
        title='Trajectory (shape sweep, start=red goal=green)',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z',
                   aspectmode='data', camera=dict(up=dict(x=0, y=1, z=0))))
    fig.write_html(save_path, include_plotlyjs='cdn')


# ──────────────────────────────────────────────────────────────────────────────
# Model & data setup
# ──────────────────────────────────────────────────────────────────────────────
modelPath = './Experiments/3dshape'
dataPath = args.dataPath

womodel = md.Model(modelPath, dataPath, DIM, [0.0] * DIM, device='cuda')

# Prefer the always-current latest.pt written every epoch by training; fall back
# to the most recent timestamped Model_Epoch_*.pt checkpoint.
latest = os.path.join(modelPath, 'latest.pt')
if os.path.exists(latest):
    pt = latest
else:
    ckpts = sorted(glob(os.path.join(modelPath, '*', 'Model_Epoch_*.pt')))
    if not ckpts:
        raise FileNotFoundError(
            f'No latest.pt and no checkpoints under {modelPath}/*/Model_Epoch_*.pt')
    pt = ckpts[-1]
print(f'Loading checkpoint: {pt}')


womodel.load(pt)
womodel.network.eval()

arr = np.load(os.path.join(dataPath, 'sampled_points.npy'))       # (N, 12)
arr_speeds = np.load(os.path.join(dataPath, 'speed.npy'))         # (N, 2)
environment_boundary_points = np.load(os.path.join(dataPath, 'env.npy'))   # (M, 3)
env_pts_cuda = torch.tensor(environment_boundary_points[:, :3],
                            dtype=torch.float32, device='cuda')

# Reconstruct the shape + wall meshes in the normalized frame from meta.json
# (written by preprocess_obj.py) so the visualization matches the sampling frame.
with open(os.path.join(dataPath, 'meta.json')) as f:
    meta = json.load(f)
env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)
shape_scale = float(meta['shape_scale'])

V_sh, F_sh, _ = load_obj(meta['shape_obj'])
shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
shape_V = ((V_sh - shape_center) / env_scale * shape_scale).astype(np.float64)
shape_F = F_sh

V_env, F_env, names_env = load_obj(meta['env_obj'])
V_env_n = (V_env - env_center) / env_scale
wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
wall_F = F_env[wall_mask]

test_list = []
test_list_speed = []
for i in range(100):
    curr = torch.tensor(arr[i]).cuda()
    test_list.append(curr)
    test_list_speed.append(min(arr_speeds[i]))

BASE = torch.zeros((1, 2 * DIM)).cuda()

# ──────────────────────────────────────────────────────────────────────────────
# Per-episode evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
total = 0
n_collision = 0
n_no_conv = 0
success_list = []
fail_list = []
test_list_speed_failed = []

two_pi = 2 * np.pi
cnt2 = 0
for XP in test_list:
    # displacement goal - start over all 6 DOF (position part used for scatter)
    diff = (XP[DIM:2 * DIM] - XP[0:DIM]).detach().cpu().numpy().tolist()
    XP = (XP + BASE).reshape(1, 2 * DIM)

    start = timer()
    with torch.no_grad():
        point, iter, success = MPPI(womodel, XP.clone(), dim=DIM)
    end = timer()

    # Build (T, 6) waypoints; rotvec de-normalized to radians (stored / 2*pi).
    waypoints = np.zeros((len(point), DIM))
    for c, p in enumerate(point):
        waypoints[c, 0:3] = [p[0][0].item(), p[0][1].item(), p[0][2].item()]
        waypoints[c, 3:6] = [p[0][3].item() * two_pi,
                             p[0][4].item() * two_pi,
                             p[0][5].item() * two_pi]

    begin_cfg = waypoints[0].copy()
    end_cfg = np.array([XP[0][DIM + 0].item(), XP[0][DIM + 1].item(), XP[0][DIM + 2].item(),
                        XP[0][DIM + 3].item() * two_pi, XP[0][DIM + 4].item() * two_pi,
                        XP[0][DIM + 5].item() * two_pi])

    collision = check_trajectory_collision(point, env_pts_cuda)
    ok = success and not collision

    status = 'success' if ok else 'fail'
    episode_path = os.path.join(OUTPUT_DIR, f'episode_{cnt2:03d}_{status}.html')
    visual_trajectory(waypoints, shape_V, shape_F, environment_boundary_points,
                      V_env_n, wall_F, begin_cfg, end_cfg, save_path=episode_path)

    if ok:
        success_list.append(diff[:3])
        print("success")
        total += 1
    else:
        fail_list.append(diff[:3])
        reason = "collision" if collision else "no convergence"
        if collision:
            n_collision += 1
        else:
            n_no_conv += 1
        print(f"fail ({reason})")
        test_list_speed_failed.append(test_list_speed[cnt2])

    cnt2 += 1

n_total = len(test_list)
success_rate = total / n_total if n_total else 0.0
print(f"total: {total} / {n_total}  ({success_rate:.1%})")

# Success-rate summary written into the output directory.
summary_lines = [
    f"data_path     : {dataPath}",
    f"model_path    : {modelPath}",
    f"checkpoint    : {pt}",
    f"episodes      : {n_total}",
    f"successes     : {total}",
    f"failures      : {n_total - total}",
    f"  collision   : {n_collision}",
    f"  no_converge : {n_no_conv}",
    f"success_rate  : {success_rate:.4f}  ({success_rate:.1%})",
]
with open(os.path.join(OUTPUT_DIR, 'success_rate.txt'), 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
print('Wrote ' + os.path.join(OUTPUT_DIR, 'success_rate.txt'))

# ──────────────────────────────────────────────────────────────────────────────
# Summary plots
# ──────────────────────────────────────────────────────────────────────────────
success_list = np.array(success_list)
fail_list = np.array(fail_list)

if success_list.size:
    fig = go.Figure(go.Scatter3d(
        x=success_list[:, 0], y=success_list[:, 1], z=success_list[:, 2],
        mode='markers', marker=dict(size=4, color='green', opacity=0.8)))
    fig.update_layout(title='Successes', scene=dict(
        xaxis_title='dx', yaxis_title='dy', zaxis_title='dz',
        camera=dict(up=dict(x=0, y=1, z=0))))
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_success.html'), include_plotlyjs='cdn')

if fail_list.size:
    fig = go.Figure(go.Scatter3d(
        x=fail_list[:, 0], y=fail_list[:, 1], z=fail_list[:, 2],
        mode='markers', marker=dict(size=4, color='red', opacity=0.8)))
    fig.update_layout(title='Failures', scene=dict(
        xaxis_title='dx', yaxis_title='dy', zaxis_title='dz',
        camera=dict(up=dict(x=0, y=1, z=0))))
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_failures.html'), include_plotlyjs='cdn')

if test_list_speed_failed:
    fig = go.Figure(go.Histogram(
        x=test_list_speed_failed, nbinsx=10,
        marker=dict(color='skyblue', line=dict(color='black', width=1)), opacity=0.7))
    fig.update_layout(title='Distribution of Speed (Failed Cases)',
                      xaxis_title='Speed Value', yaxis_title='Frequency', bargap=0.05)
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_speed_failed.html'), include_plotlyjs='cdn')

# ──────────────────────────────────────────────────────────────────────────────
# HTTP server — browse to http://<server-ip>:8080 from your host PC
# ──────────────────────────────────────────────────────────────────────────────
os.chdir(OUTPUT_DIR)
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", HTTP_PORT), handler) as httpd:
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    print(f"Serving at http://0.0.0.0:{HTTP_PORT}  —  open this on your host PC")
    print("Press Ctrl-C to stop.\n")
    httpd.serve_forever()
