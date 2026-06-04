"""Single-episode (non-batched) evaluation for the 2-D shape planner.

Same MPPI rollout as ``evaluate_training_batched.py`` but solves one start/goal
pair at a time so each trajectory can be rendered with ``visual_training`` (the
F-shape sweeping through the maze, start/goal poses marked in red).  Finishes
with success/failure scatter plots and a histogram of failed-case speeds.

All plots are saved as HTML files under ./output/ and served over HTTP on
port 8080 so they can be viewed from a remote host:

    http://<server-ip>:8080

Run from the nested ntrl-demo root:

    python evaluate_training.py
"""

import sys
sys.path.append('.')

import os
import argparse
import http.server
import socketserver
import threading
from glob import glob
from timeit import default_timer as timer

import numpy as np
import torch
import plotly.graph_objects as go

from models.metric import model_train_metric as md
from dataprocessing.parse_shape import dxf_to_shape, shape_to_points
from visualfunctions import visual_training

COLLISION_THRESH = 0.001
HTTP_PORT = 8080

parser = argparse.ArgumentParser(
    description='Single-episode evaluation for the 2-D shape planner.')
parser.add_argument('--dataPath', default='./testing_data/2dshape/Fmaze2_norm',
                    help='Directory holding the test data (sampled_points.npy, '
                         'speed.npy, env.npy) used as start/goal pairs.')
parser.add_argument('--out', default='./output',
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


def check_trajectory_collision(traj: list, env_pts: torch.Tensor) -> bool:
    """Return True if ANY waypoint in `traj` comes within COLLISION_THRESH of ANY
    environment boundary point.  Only x, y (first 2 dims) are compared."""
    waypoints = torch.cat(traj, dim=0)[:, :2].cuda()
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


# ──────────────────────────────────────────────────────────────────────────────
# Model & data setup
# ──────────────────────────────────────────────────────────────────────────────
modelPath = './Experiments/2dshape'
dataPath = args.dataPath

womodel = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0], device='cuda')

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

arr = np.load(os.path.join(dataPath, 'sampled_points.npy'))
arr_speeds = np.load(os.path.join(dataPath, 'speed.npy'))
environment_boundary_points = np.load(os.path.join(dataPath, 'env.npy'))
env_pts_cuda = torch.tensor(
    environment_boundary_points[:, :2], dtype=torch.float32, device='cuda'
)
Fshape_norm = dxf_to_shape('./datasets/2dshape/Fshape_norm.dxf')
Fshape_points = shape_to_points(Fshape_norm)

test_list = []
test_list_speed = []
for i in range(100):
    curr = torch.tensor(arr[i]).cuda()
    test_list.append(curr)
    test_list_speed.append(min(arr_speeds[i]))

BASE = torch.tensor([[0, 0, 0, 0, 0, 0]]).cuda()

# ──────────────────────────────────────────────────────────────────────────────
# Per-episode evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
total = 0
n_collision = 0
n_no_conv = 0
success_list = []
fail_list = []
test_list_speed_failed = []

cnt2 = 0
for XP in test_list:
    print(XP)
    diff = [
        (XP[3] - XP[0]).item(),
        (XP[4] - XP[1]).item(),
        (XP[5] - XP[3]).item()
    ]
    XP = XP + BASE

    start = timer()
    with torch.no_grad():
        point, iter, success = MPPI(womodel, XP.clone(), dim=3)
    end = timer()

    cnt = 0
    start_list = np.zeros((len(point), 3))
    for p in point:
        start_list[cnt, 0] = p[0][0].item()
        start_list[cnt, 1] = p[0][1].item()
        start_list[cnt, 2] = p[0][2].item() * 2 * np.pi
        cnt += 1

    begin_point = [XP[0][0].item(), XP[0][1].item(), XP[0][2].item() * 2 * np.pi]
    end_point   = [XP[0][3].item(), XP[0][4].item(), XP[0][5].item() * 2 * np.pi]
    start_list[0, 0] = XP[0][0].item()
    start_list[0, 1] = XP[0][1].item()
    start_list[0, 2] = XP[0][2].item() * 2 * np.pi

    speed_temp = np.full((len(point), 1), 0.5)

    collision = check_trajectory_collision(point, env_pts_cuda)
    ok = success and not collision

    status = 'success' if ok else 'fail'
    episode_path = os.path.join(OUTPUT_DIR, f'episode_{cnt2:03d}_{status}.html')
    visual_training(start=start_list, shape_points=Fshape_points,
                    env_points=environment_boundary_points,
                    cnt=len(point), speed=speed_temp, vmin=0.2,
                    begin_point=begin_point, end_point=end_point,
                    save_path=episode_path)

    if ok:
        success_list.append(diff)
        print("success")
        total += 1
    else:
        fail_list.append(diff)
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
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.8),
    ))
    fig.update_layout(title='Successes', scene=dict(
        xaxis_title='dx', yaxis_title='dy', zaxis_title='dtheta'))
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_success.html'), include_plotlyjs='cdn')

if fail_list.size:
    fig = go.Figure(go.Scatter3d(
        x=fail_list[:, 0], y=fail_list[:, 1], z=fail_list[:, 2],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.8),
    ))
    fig.update_layout(title='Failures', scene=dict(
        xaxis_title='dx', yaxis_title='dy', zaxis_title='dtheta'))
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_failures.html'), include_plotlyjs='cdn')

if test_list_speed_failed:
    fig = go.Figure(go.Histogram(
        x=test_list_speed_failed, nbinsx=10,
        marker=dict(color='skyblue', line=dict(color='black', width=1)),
        opacity=0.7,
    ))
    fig.update_layout(
        title='Distribution of Speed (Failed Cases)',
        xaxis_title='Speed Value',
        yaxis_title='Frequency',
        bargap=0.05,
    )
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_speed_failed.html'), include_plotlyjs='cdn')



os.chdir(OUTPUT_DIR)
handler = http.server.SimpleHTTPRequestHandler
http.server.ThreadingHTTPServer.allow_reuse_address = True
with http.server.ThreadingHTTPServer(("", HTTP_PORT), handler) as httpd:
    print(f"\nPlots saved to {OUTPUT_DIR}/")
    print(f"Serving at http://0.0.0.0:{HTTP_PORT}  —  open this on your host PC")
    print("Press Ctrl-C to stop.\n")
    httpd.serve_forever()
