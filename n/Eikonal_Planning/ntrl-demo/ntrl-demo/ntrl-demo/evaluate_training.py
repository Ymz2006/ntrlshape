"""Single-episode (non-batched) evaluation for the 2-D shape planner.

Same MPPI rollout as ``evaluate_training_batched.py`` but solves one start/goal
pair at a time so each trajectory can be rendered with ``visual_training`` (the
F-shape sweeping through the maze, start/goal poses marked in red).  Finishes
with success/failure scatter plots and a histogram of failed-case speeds.

Run from the nested ntrl-demo root:

    python evaluate_training.py
"""

import sys
sys.path.append('.')

import os
from glob import glob
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3-D projection)

from models.metric import model_train_metric as md
from dataprocessing.parse_shape import dxf_to_shape, shape_to_points
from visualfunctions import visual_training

COLLISION_THRESH = 0.001  # min distance from env boundary point to count as collision


def check_trajectory_collision(traj: list, env_pts: torch.Tensor) -> bool:
    """Return True if ANY waypoint in `traj` comes within COLLISION_THRESH of ANY
    environment boundary point.  Only x, y (first 2 dims) are compared.

    Args:
        traj    : list of (1, dim) tensors — the recorded waypoints
        env_pts : (N, 2) float tensor on CUDA — boundary point cloud
    """
    waypoints = torch.cat(traj, dim=0)[:, :2].cuda()   # (T, 2)
    diff = waypoints.unsqueeze(1) - env_pts.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)                    # (T, N)
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
dataPath = './datasets/2dshape/Fmaze2_norm'

womodel = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0], device='cuda')

# Pick the most recent run folder and its highest-epoch checkpoint
ckpts = sorted(glob(os.path.join(modelPath, '*', 'Model_Epoch_*.pt')))
if not ckpts:
    raise FileNotFoundError(f'No checkpoints found under {modelPath}/*/Model_Epoch_*.pt')
pt = ckpts[-1]
print(f'Loading checkpoint: {pt}')
womodel.load(pt)
womodel.network.eval()

arr = np.load(os.path.join(dataPath, 'sampled_points.npy'))
arr_speeds = np.load(os.path.join(dataPath, 'speed.npy'))
environment_boundary_points = np.load(os.path.join(dataPath, 'env.npy'))
env_pts_cuda = torch.tensor(
    environment_boundary_points[:, :2], dtype=torch.float32, device='cuda'
)   # (N, 2) — only x, y needed for 2-D collision check
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
# Per-episode evaluation loop (with trajectory visualization)
# ──────────────────────────────────────────────────────────────────────────────
total = 0
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
    end_point = [XP[0][3].item(), XP[0][4].item(), XP[0][5].item() * 2 * np.pi]
    start_list[0, 0] = XP[0][0].item()
    start_list[0, 1] = XP[0][1].item()
    start_list[0, 2] = XP[0][2].item() * 2 * np.pi

    speed_temp = np.full((len(point), 1), 0.5)

    # Fail if the planner did not converge OR the trajectory clips the environment
    collision = check_trajectory_collision(point, env_pts_cuda)
    visual_training(start=start_list, shape_points=Fshape_points,
                    env_points=environment_boundary_points,
                    cnt=len(point), speed=speed_temp, vmin=0.2,
                    begin_point=begin_point, end_point=end_point)
    if success and not collision:
        success_list.append(diff)
        print("success")
        total += 1
    else:
        fail_list.append(diff)
        reason = "collision" if collision else "no convergence"
        print("fail (" + reason + ")")
        test_list_speed_failed.append(test_list_speed[cnt2])
        # visual_training(start=start_list, shape_points=Fshape_points,
        #                 env_points=environment_boundary_points,
        #                 cnt=len(point), speed=speed_temp, vmin=0.2,
        #                 begin_point=begin_point, end_point=end_point)

    cnt2 += 1

print("total:" + str(total))

# ──────────────────────────────────────────────────────────────────────────────
# Summary scatter / histogram plots
# ──────────────────────────────────────────────────────────────────────────────
success_list = np.array(success_list)
fail_list = np.array(fail_list)

if success_list.size:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(success_list[:, 0], success_list[:, 1], success_list[:, 2])
    ax.set_title('Successes')
    plt.show()

if fail_list.size:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fail_list[:, 0], fail_list[:, 1], fail_list[:, 2])
    ax.set_title('Failures')
    plt.show()

if test_list_speed_failed:
    plt.hist(test_list_speed_failed, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Speed Failed')
    plt.xlabel('Speed Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
