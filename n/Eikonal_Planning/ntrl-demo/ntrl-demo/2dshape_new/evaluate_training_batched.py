import sys
sys.path.append('.')
import model2d as md
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
import math

from parse_shape import dxf_to_shape, shape_to_points
import ezdxf

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 16    # number of start/goal pairs to solve in parallel
TOTAL       = 100   # total pairs to evaluate (will stop early if dataset is smaller)
DIM         = 3     # state dimension (x, y, theta)

STEPS       = 200   # max MPPI iterations per episode
SAMPLE_NUM  = 50    # rollout samples per iteration
HORIZON     = 5     # rollout horizon
NOISE_STD   = 0.015 # base perturbation magnitude
CONV_THRESH = 0.01  # distance threshold for declaring success
SOFTMAX_T   = 50.0  # temperature for softmax weighting (higher = greedier)
GOAL_WEIGHT = 10.0  # relative cost weight on final vs intermediate horizon step
COLLISION_THRESH = 0.02  # min distance from env boundary point to count as collision

# ──────────────────────────────────────────────────────────────────────────────
# Batched MPPI
# ──────────────────────────────────────────────────────────────────────────────
def MPPI_batched(womodel, XP: torch.Tensor, dim: int):
    """
    Run MPPI for a full batch of start/goal pairs in parallel on GPU.

    Args:
        womodel : loaded neural network model with a .function.TravelTimes method
        XP      : (B, dim*2) tensor on CUDA — first `dim` cols are start,
                  last `dim` cols are goal
        dim     : dimensionality of the configuration space

    Returns:
        trajectories : list[list[Tensor]]  — per-batch list of (1, dim) waypoints
        final_iters  : (B,) int tensor     — iteration at which each episode ended
        found_path   : (B,) bool tensor    — True if the episode converged
    """
    B = XP.shape[0]
    device = XP.device

    dP_prior = torch.zeros(B, dim, device=device)

    # One trajectory list per batch element; start with the initial position
    trajectories = [[XP[b, 0:dim].unsqueeze(0).clone()] for b in range(B)]

    found_path  = torch.zeros(B, dtype=torch.bool,  device=device)
    active      = torch.ones (B, dtype=torch.bool,  device=device)
    final_iters = torch.full ((B,), STEPS, dtype=torch.int32)

    for it in range(STEPS):
        if not active.any():
            break

        # ── 1. Expand current state to (B, SAMPLE_NUM, HORIZON, dim*2) ────────
        # Goal columns remain fixed; only the start columns are perturbed.
        XP_tmp = (XP                            # (B, dim*2)
                  .unsqueeze(1).unsqueeze(1)    # (B, 1, 1, dim*2)
                  .expand(B, SAMPLE_NUM, HORIZON, dim * 2)
                  .clone())                     # contiguous copy for in-place write

        # ── 2. Sample perturbations ────────────────────────────────────────────
        # Shared noise across the horizon (correlated) + per-step noise
        noise_shared = NOISE_STD * torch.randn(B, SAMPLE_NUM, 1,       dim, device=device)
        noise_steps  = NOISE_STD * torch.randn(B, SAMPLE_NUM, HORIZON, dim, device=device)
        dP = noise_shared + noise_steps                       # (B, S, H, dim)

        # Bias towards the previous best direction
        dP = dP + 2.0 * dP_prior.view(B, 1, 1, dim)

        # Clip each perturbation to a ball of radius NOISE_STD
        dP_norm = torch.norm(dP, dim=-1, keepdim=True)       # (B, S, H, 1)
        dP = dP / (torch.clamp(dP_norm, min=NOISE_STD) / NOISE_STD)

        # ── 3. Roll out perturbations along the horizon ────────────────────────
        dP_cumsum = torch.cumsum(dP, dim=2)                  # (B, S, H, dim)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + dP_cumsum

        # ── 4. Evaluate cost via the neural travel-time model ──────────────────
        # Select first and last horizon steps; flatten for the network call
        selected = XP_tmp[:, :, [0, -1], :]                  # (B, S, 2, dim*2)
        flat     = selected.reshape(B * SAMPLE_NUM * 2, dim * 2)

        cost_raw = womodel.function.TravelTimes(flat)        # (B*S*2,)
        cost_raw = cost_raw.reshape(B, SAMPLE_NUM, 2)        # (B, S, 2)

        # Weighted combination: penalise distance-to-goal more than intermediate
        cost = GOAL_WEIGHT * cost_raw[:, :, 0] + cost_raw[:, :, 1]  # (B, S)

        # ── 5. Compute importance weights and update prior ─────────────────────
        weight   = torch.softmax(-SOFTMAX_T * cost, dim=1)            # (B, S)
        dP_prior = torch.einsum('bs,bsd->bd', weight, dP[:, :, 0, :])  # (B, dim)

        # ── 6. Step current position forward ──────────────────────────────────
        XP[:, 0:dim] = XP[:, 0:dim] + dP_prior

        # ── 7. Check convergence for each active element ──────────────────────
        dist = torch.norm(XP[:, dim:dim*2] - XP[:, 0:dim], dim=1)   # (B,)
        newly_done = active & (dist < CONV_THRESH)

        found_path  |= newly_done
        active[newly_done] = False
        final_iters[newly_done] = it

        # Store current position for every batch element
        for b in range(B):
            trajectories[b].append(XP[b, 0:dim].unsqueeze(0).clone())

    # Append the true goal as the final waypoint
    for b in range(B):
        trajectories[b].append(XP[b, dim:dim*2].unsqueeze(0).clone())

    return trajectories, final_iters, found_path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_diff(XP_row: torch.Tensor, dim: int):
    """Compute a simple displacement descriptor for success/fail scatter plots."""
    cpu = XP_row.cpu()
    return [
        (cpu[dim    ] - cpu[0]).item(),
        (cpu[dim + 1] - cpu[1]).item(),
        (cpu[dim + 2] - cpu[dim]).item(),
    ]


def build_start_list(traj, XP_row, dim):
    """Convert a trajectory list into a numpy array for optional visualisation."""
    start_list = np.zeros((len(traj), dim))
    for i, p in enumerate(traj):
        start_list[i, 0] = p[0, 0].item()
        start_list[i, 1] = p[0, 1].item()
        start_list[i, 2] = p[0, 2].item() * 2 * np.pi
    start_list[0, 0] = XP_row[0].item()
    start_list[0, 1] = XP_row[1].item()
    start_list[0, 2] = XP_row[2].item() * 2 * np.pi
    return start_list

def check_trajectory_collision(traj: list, env_pts: torch.Tensor) -> bool:
    """
    Return True if ANY waypoint in `traj` is within COLLISION_THRESH of ANY
    environment boundary point.  Only x, y (first 2 dims) are compared.

    Args:
        traj    : list of (1, dim) tensors — the recorded waypoints
        env_pts : (N, 2) float tensor on CUDA — boundary point cloud

    Returns:
        True if a collision is detected, False otherwise.
    """
    # Stack all waypoints into (T, 2), keeping only x and y
    waypoints = torch.cat(traj, dim=0)[:, :2].cuda()   # (T, 2)

    # Pairwise distances: (T, 1, 2) vs (1, N, 2)  →  (T, N)
    diff  = waypoints.unsqueeze(1) - env_pts.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)                    # (T, N)

    return dists.min().item() < COLLISION_THRESH
# ──────────────────────────────────────────────────────────────────────────────
# Model & data setup
# ──────────────────────────────────────────────────────────────────────────────
modelPath = './Experiments/UR5'
meshname  = 'Auburn'
dataPath  = './datasets/arm/' + meshname

womodel = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0, 0.0], device='cuda')
pt = './Experiments/latest/Model_latest.pt'
print(f'Loading checkpoint: {pt}')
womodel.load(pt)
womodel.network.eval()

arr        = np.load('./testing_data2d/Fshape_FmazeEasy/sampled_points.npy')
arr_speeds = np.load('./testing_data2d/Fshape_FmazeEasy/speed.npy')

Fshape_norm   = dxf_to_shape('./datasets/Fshape_norm.dxf')
Fshape_points = shape_to_points(Fshape_norm)
environment_boundary_points = np.load('./testing_data2d/Fshape_FmazeEasy/Fmaze3env.npy')
env_pts_cuda = torch.tensor(
    environment_boundary_points[:, :2], dtype=torch.float32, device='cuda'
)   # (N, 2) — only x, y needed for 2-D collision check


# Build test tensors on CPU; we'll move batches to GPU below
test_tensors = [torch.tensor(arr[i], dtype=torch.float32) for i in range(len(arr))]
test_speeds  = [float(min(arr_speeds[i])) for i in range(len(arr_speeds))]

BASE = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device='cuda')

# ──────────────────────────────────────────────────────────────────────────────
# Batched evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
n_eval = min(TOTAL, len(test_tensors))
print(f'Evaluating {n_eval} samples  |  batch_size={BATCH_SIZE}  |  DIM={DIM}')

success_list : list[list[float]] = []
fail_list    : list[list[float]] = []
failed_speeds: list[float]       = []

total_success = 0
timer_start   = timer()

with torch.no_grad():
    idx = 0
    while idx < n_eval:
        batch_end = min(idx + BATCH_SIZE, n_eval)
        B_actual  = batch_end - idx

        # Stack raw tensors into (B, dim*2), send to GPU, add offset
        batch_raw = torch.stack(test_tensors[idx:batch_end], dim=0).cuda()  # (B, dim*2)
        batch_XP  = batch_raw + BASE.unsqueeze(0)                           # broadcast

        # --- run batched MPPI -------------------------------------------------
        trajs, iters, success = MPPI_batched(womodel, batch_XP.clone(), DIM)

        # --- collect results --------------------------------------------------
        for b in range(B_actual):
            global_i = idx + b
            diff = extract_diff(batch_raw[b], DIM)

            # Collision check: fail if any waypoint touches the environment
            collision = check_trajectory_collision(trajs[b], env_pts_cuda)

            if success[b].item() and not collision:
                success_list.append(diff)
                total_success += 1
                print(f'[{global_i:4d}] ✓ success  iter={iters[b].item()}')
            else:
                fail_list.append(diff)
                failed_speeds.append(test_speeds[global_i])
                reason = 'collision' if collision else 'no convergence'
                print(f'[{global_i:4d}] ✗ fail     iter={iters[b].item()}  ({reason})')
        idx = batch_end

elapsed = timer() - timer_start
print(f'\n─────────────────────────────────────────')
print(f'Total evaluated : {n_eval}')
print(f'Successes       : {total_success}  ({100*total_success/n_eval:.1f}%)')
print(f'Failures        : {n_eval - total_success}')
print(f'Wall time       : {elapsed:.2f} s  ({elapsed/n_eval*1000:.1f} ms/sample)')
print(f'─────────────────────────────────────────')

# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def scatter3d(data, title, color):
    arr = np.array(data)
    if arr.size == 0:
        print(f'No data for "{title}"')
        return
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=color, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Δx'); ax.set_ylabel('Δy'); ax.set_zlabel('Δθ')
    plt.tight_layout()
    plt.show()

scatter3d(success_list, 'Successes', 'steelblue')
scatter3d(fail_list,    'Failures',  'tomato')

if failed_speeds:
    plt.figure()
    plt.hist(failed_speeds, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Speed distribution of failed cases')
    plt.xlabel('Speed value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
