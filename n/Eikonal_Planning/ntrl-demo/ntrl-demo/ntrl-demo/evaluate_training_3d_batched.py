"""Batched (GPU) evaluation for the 3-D shape planner.

Functionally identical to ``evaluate_training_3d.py`` -- same MPPI rollout, same
collision / convergence / linear-interp accounting, same success_rate.txt, same
plotly summaries, and the same interactive viser viewer -- but the MPPI rollouts
for many start/goal pairs are run *simultaneously* on the GPU instead of one
episode at a time.  This is the dominant cost (repeated ``TravelTimes`` network
evaluations), so batching gives a large speedup.

The per-waypoint collision test still runs on the CPU through ``igl`` (it is the
exact same point-in-solid check used by the non-batched script, so the pass/fail
labels are bit-for-bit the same); only the network-driven rollout is batched.

Episodes are processed in chunks of ``--batch`` at a time to bound GPU memory.

Before any rollout, every test case is screened with the same mesh collision
check at its START and GOAL pose.  A case whose start or goal already overlaps
the environment is unsolvable by construction, so it is discarded outright and
excluded from the denominator -- the reported success rate is therefore the
actual rate over solvable cases, not diluted by degenerate test data.

Run from the nested ntrl-demo root:

    python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_env1 --out ./results/output_3d/rectangle_env1 --batch 250
"""


import sys
sys.path.append('.')


import os
import json
import time
import argparse
from glob import glob
from timeit import default_timer as timer


import numpy as np
import torch
import igl
import plotly.graph_objects as go
# ``viser`` is imported lazily inside launch_viser() so headless runs
# (``--no-viser``) work on machines where viser is not installed.


from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import (
    load_obj, _rotvec_to_matrix_np, sample_surface_points)


# Sign convention for igl.signed_distance: the DEFAULT/WINDING_NUMBER types are
# unreliable in this binding, but FAST_WINDING_NUMBER returns negative-inside and
# is robust to per-face orientation (only requires a closed/watertight mesh).
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
# Number of surface points sampled over the full environment mesh for collision.
ENV_COLLISION_POINTS = 50000
HTTP_PORT = 8080
DIM = 6


parser = argparse.ArgumentParser(
    description='Batched (GPU) evaluation for the 3-D shape planner.')
parser.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1',
                    help='Directory holding the test data (sampled_points.npy, '
                         'speed.npy, env.npy, meta.json) used as start/goal pairs.')
parser.add_argument('--out', default='./output_3d',
                    help='Output directory for the rendered HTML plots (also the '
                         'web root served over HTTP).')
parser.add_argument('--batch', type=int, default=250,
                    help='Number of episodes whose MPPI rollouts are run together '
                         'on the GPU per chunk (bounds GPU memory).')
parser.add_argument('--device', default='cuda',
                    help='Torch device the network and the batched rollouts run '
                         'on (e.g. cuda, cuda:2, cpu).')
parser.add_argument('--nbins_dist', type=int, default=8,
                    help='speed_dist bins (uniform over [0,1]) for the end-of-run '
                         'endpoint frequency tables.')
parser.add_argument('--nbins_angle', type=int, default=8,
                    help='speed_angle bins (uniform over [0,1]) for the end-of-run '
                         'endpoint frequency tables.')
parser.add_argument('--no-viser', action='store_true',
                    help='Skip launching the interactive viser viewer (still '
                         'writes success_rate.txt and the summary HTML plots).')
args = parser.parse_args()


OUTPUT_DIR = args.out
EPISODE_BATCH = args.batch
DEVICE = args.device


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
def check_trajectory_collision(traj, env_pts, shape_V, shape_F, shape_radius):
    """Return True if the placed shape mesh overlaps the environment at ANY waypoint.


    Collision uses the same point-in-solid test that labels the training data
    (preprocess_obj.points_inside_tets): an environment surface point lying
    *inside* the watertight shape mesh.  The full 6-DOF pose (position + rotvec)
    is applied at each waypoint, and ``env_pts`` densely covers the whole
    environment surface (obstacles *and* walls), so everything counts as an
    obstacle.  No interpolation between waypoints -- each is checked on its own.


    ``traj`` is a list of (1, DIM) configs with the rotvec stored normalized by
    2*pi (as written by preprocess_obj); ``shape_V``/``shape_F`` is the shape
    mesh in its local frame; ``shape_radius`` is max ||vertex|| about that frame.
    """
    two_pi = 2 * np.pi
    for cfg_t in traj:
        cfg = cfg_t.detach().cpu().numpy().reshape(-1)
        t = cfg[0:3]
        # Broad phase: a point inside the shape must lie within its bounding radius.
        near = env_pts[np.linalg.norm(env_pts - t, axis=1) <= shape_radius]
        if near.shape[0] == 0:
            continue
        # Map env points into the shape's local frame (inverse of _placed_mesh,
        # which does  world = local @ R.T + t  =>  local = (world - t) @ R).
        R = _rotvec_to_matrix_np(cfg[3:6] * two_pi)
        near_local = np.ascontiguousarray((near - t) @ R)
        S = igl.signed_distance(near_local, shape_V, shape_F, _SDF_SIGN)[0]
        if S.size and S.min() < 0.0:
            return True
    return False




def check_config_collision(cfg, env_pts, shape_V, shape_F, shape_radius):
    """Return True if the shape placed at a single config overlaps the environment.


    Thin wrapper around ``check_trajectory_collision`` for a one-waypoint
    "trajectory"; ``cfg`` is any tensor holding DIM normalized coordinates.
    """
    return check_trajectory_collision(
        [cfg.reshape(1, -1)], env_pts, shape_V, shape_F, shape_radius)




def linear_interp_traj(start_cfg, goal_cfg, n_steps=50):
    """Build a list of (1, DIM) configs linearly interpolating start -> goal.


    Configs are returned in the same normalized space (rotvec stored / 2*pi)
    that ``check_trajectory_collision`` consumes, so the straight-line path can
    be collision-checked exactly like an MPPI trajectory.  Used to measure
    whether the trivial linear interpolation from start to goal is itself a
    valid (collision-free) path.
    """
    start = start_cfg.reshape(1, -1)
    goal = goal_cfg.reshape(1, -1)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=start.device).reshape(-1, 1)
    interp = (1.0 - alphas) * start + alphas * goal
    return [interp[i:i + 1] for i in range(n_steps)]




def MPPI_batched(womodel, XP, dim):
    """Batched MPPI: run ``B`` start/goal pairs' rollouts together on the GPU.


    ``XP`` is a (B, 2*dim) tensor on the run device (each row ``[start(dim) | goal(dim)]``,
    rotvec stored normalized by 2*pi).  This is the vectorized-over-episodes form
    of the single-episode ``MPPI`` in evaluate_training_3d.py: identical sampling,
    cost, softmax-weighting and convergence rule, just with a leading batch axis.


    Once an episode's current config reaches the goal (``dis < 0.01``) it is
    *frozen* (its config stops updating) exactly like the single-episode version
    breaks out of the loop; other episodes in the batch keep stepping.  Episodes
    are fully independent within the batch, so freezing one does not affect any
    other.


    Returns, per episode, the same objects the single-episode loop expects:
        points_list : list (len B) of lists of (1, dim) configs -- the recorded
                      trajectory up to convergence with the goal config appended
                      last (matches ``point0`` from the single-episode MPPI).
        iters       : list (len B) of the convergence iteration index.
        success     : list (len B) of bools.
    """
    B = XP.shape[0]
    steps = 200
    sample_num = 50
    horizon = 5
    dev = XP.device


    dP_prior = torch.zeros((B, dim), device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)


    # recorded[k] is the (B, dim) current config after k updates (recorded[0] = start).
    recorded = [XP[:, 0:dim].clone()]


    for it in range(steps):
        XP_tmp = XP.clone()
        # (B, sample_num, horizon, 2*dim)
        XP_tmp = XP_tmp[:, None, None, :].repeat(1, sample_num, horizon, 1)


        dP = 0.015 * torch.normal(0, 1, size=(B, sample_num, 1, dim),
                                  dtype=torch.float32, device=dev) \
            + 0.015 * torch.normal(0, 1, size=(B, sample_num, horizon, dim),
                                   dtype=torch.float32, device=dev)
        dP = dP + 2 * dP_prior[:, None, None, :]
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=0.015) / 0.015)
        dP_cumsum = torch.cumsum(dP, dim=2)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + dP_cumsum


        # First and last horizon step of every sample -> (B, sample_num, 2, 2*dim).
        endpoints = XP_tmp[:, :, [0, -1], :]
        cost = womodel.function.TravelTimes(endpoints.reshape(-1, dim * 2))
        cost = cost.reshape(B, sample_num, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]           # (B, sample_num)


        weight = torch.softmax(-50 * cost, dim=1)           # (B, sample_num)
        # Weighted mean of the first-step displacement over samples -> (B, dim).
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        dP_prior = step_prior


        # Freeze converged episodes: only advance those not yet done.
        XP[:, 0:dim] = XP[:, 0:dim] + step_prior * (~done).unsqueeze(1)


        dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)   # (B,)
        recorded.append(XP[:, 0:dim].clone())


        newly = (dis < 0.01) & (~done)
        conv_step[newly] = it
        done = done | (dis < 0.01)
        if bool(done.all()):
            break


    success = done.detach().cpu().numpy()
    conv_step_cpu = conv_step.detach().cpu().numpy()
    n_recorded = len(recorded)
    goal = XP[:, dim:dim * 2]


    points_list = []
    iters = []
    for b in range(B):
        if success[b]:
            # start + (conv_step+1) updates  ->  recorded[0 .. conv_step+1].
            end = int(conv_step_cpu[b]) + 2
        else:
            end = n_recorded
        cfgs = [recorded[k][b:b + 1, :] for k in range(end)]
        cfgs.append(goal[b:b + 1, :])          # append goal config, like point0
        points_list.append(cfgs)
        iters.append(int(conv_step_cpu[b]))


    return points_list, iters, success.tolist()




def _placed_mesh(shape_V, cfg):
    """Transform the shape's local mesh by a config (x,y,z, rotvec in radians)."""
    R = _rotvec_to_matrix_np(cfg[3:6])
    return shape_V @ R.T + cfg[0:3]




def _progress_color(t):
    """Map a progress value t in [0, 1] to an RGB tuple of ints (viridis)."""
    try:
        import matplotlib.cm as cm
        r, g, b, _ = cm.get_cmap('viridis')(float(t))
    except Exception:
        # Fallback: simple blue -> yellow ramp if matplotlib is unavailable.
        r, g, b = float(t), float(t), 1.0 - float(t)
    return (int(r * 255), int(g * 255), int(b * 255))




def _speed_color(s):
    """Map a model-predicted speed s in [0, 1] to an RGB tuple (viridis)."""
    return _progress_color(float(np.clip(s, 0.0, 1.0)))




def add_environment(server, env_V, obst_F, wall_F):
    """Draw the environment as its actual triangle mesh (static scene).


    Obstacles are solid grey; walls are translucent light-blue.  This replaces
    the old sampled-point cloud rendering.
    """
    if len(obst_F) > 0:
        server.scene.add_mesh_simple(
            '/env/obstacles', vertices=env_V, faces=obst_F,
            color=(150, 150, 150), opacity=1.0, flat_shading=True, side='double')
    if len(wall_F) > 0:
        server.scene.add_mesh_simple(
            '/env/walls', vertices=env_V, faces=wall_F,
            color=(173, 216, 230), opacity=0.15, flat_shading=True, side='double')




def render_episode(server, ep, shape_V, shape_F, show_speed_labels=True):
    """Add the moving-shape sweep + start/goal poses for one episode.


    The shape mesh is drawn at every waypoint, colored by the model's PREDICTED
    SPEED at that config (viridis, 0=slow .. 1=fast) when ``ep['speeds']`` is
    present, otherwise by progress along the path.  When ``show_speed_labels`` is
    set, a small floating text label at each waypoint reports the numeric speed.
    The start pose is red and the goal pose is green.  Returns the list of scene
    handles so the caller can remove them before rendering the next episode.


    ``ep['waypoints']`` is a (T, 6) array of poses [x, y, z, rx, ry, rz] (rotvec
    in radians); ``ep['speeds']`` is a (T,) array of model-predicted speeds.
    """
    handles = []
    waypoints = ep['waypoints']
    speeds = ep.get('speeds')
    T = len(waypoints)
    for t in range(T):
        Vp = _placed_mesh(shape_V, waypoints[t])
        color = (_speed_color(speeds[t]) if speeds is not None
                 else _progress_color(t / max(T - 1, 1)))
        handles.append(server.scene.add_mesh_simple(
            f'/episode/traj/{t:04d}', vertices=Vp, faces=shape_F,
            color=color, opacity=0.5,
            flat_shading=True, side='double'))
        if speeds is not None and show_speed_labels:
            handles.append(server.scene.add_label(
                f'/episode/spd/{t:04d}', text=f'{speeds[t]:.3f}',
                position=tuple(Vp.mean(axis=0))))


    for cfg, col, nm in [(ep['begin_cfg'], (220, 30, 30), 'start'),
                         (ep['end_cfg'], (30, 180, 30), 'goal')]:
        if cfg is None:
            continue
        Vp = _placed_mesh(shape_V, cfg)
        handles.append(server.scene.add_mesh_simple(
            f'/episode/{nm}', vertices=Vp, faces=shape_F,
            color=col, opacity=0.9, flat_shading=True, side='double'))
    return handles




def launch_viser(episodes, shape_V, shape_F, env_V, obst_F, wall_F):
    """Serve an interactive viser scene with a dropdown to browse episodes."""
    import viser
    server = viser.ViserServer(host='0.0.0.0', port=HTTP_PORT)
    server.scene.set_up_direction('+y')


    add_environment(server, env_V, obst_F, wall_F)


    options = [f"{ep['idx']:03d}_{ep['status']}" for ep in episodes]
    dropdown = server.gui.add_dropdown('Episode', options=options)
    speed_labels = server.gui.add_checkbox('Show speed labels', initial_value=True)
    speed_summary = server.gui.add_text('Predicted speed', initial_value='', disabled=True)


    current = []


    def show(i):
        for h in current:
            h.remove()
        current.clear()
        ep = episodes[i]
        current.extend(render_episode(
            server, ep, shape_V, shape_F, show_speed_labels=speed_labels.value))
        sp = ep.get('speeds')
        speed_summary.value = (
            'min {:.3f}  mean {:.3f}  max {:.3f}'.format(
                float(np.min(sp)), float(np.mean(sp)), float(np.max(sp)))
            if sp is not None and len(sp) else 'n/a')


    @dropdown.on_update
    def _(_):
        show(options.index(dropdown.value))


    @speed_labels.on_update
    def _(_):
        show(options.index(dropdown.value))


    if episodes:
        show(0)


    print(f"\nServing viser at http://0.0.0.0:{HTTP_PORT}  —  open this on your host PC")
    print("Use the 'Episode' dropdown to browse. Press Ctrl-C to stop.\n")
    while True:
        time.sleep(10)




# ──────────────────────────────────────────────────────────────────────────────
# Model & data setup
# ──────────────────────────────────────────────────────────────────────────────
modelPath = './Experiments/3dshape'
dataPath = args.dataPath


womodel = md.Model(modelPath, dataPath, DIM, [0.0] * DIM, device=DEVICE)


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


pt = './Experiments/3dshape/3dshape_08_08_22_26/latest.pt'

print(f'Loading checkpoint: {pt}')


womodel.load(pt)
womodel.network.eval()


arr = np.load(os.path.join(dataPath, 'sampled_points.npy'))       # (N, 12)
arr_speeds = np.load(os.path.join(dataPath, 'speed.npy'))         # (N, 2)


# Per-endpoint clearance proxies, used only for the end-of-run frequency tables.
# Column 0 is the START (x0), column 1 the GOAL (x1).  Optional: older datasets
# may predate them, in which case the tables are skipped.
def _try_load(name):
    p = os.path.join(dataPath, name)
    return np.load(p) if os.path.exists(p) else None


arr_speed_dists = _try_load('speed_dists.npy')                    # (N, 2) or None
arr_speed_angles = _try_load('speed_angles.npy')                  # (N, 2) or None


# Reconstruct the shape + wall meshes in the normalized frame from meta.json
# (written by preprocess_obj.py) so the visualization matches the sampling frame.
with open(os.path.join(dataPath, 'meta.json')) as f:
    meta = json.load(f)
env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)
shape_scale = float(meta['shape_scale'])


V_sh, F_sh, _ = load_obj(meta['shape_obj'])
shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                               dtype=np.float64)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
# Bounding radius of the shape about its local origin (used for collision broad phase).
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())


V_env, F_env, names_env = load_obj(meta['env_obj'])
V_env_n = (V_env - env_center) / env_scale
wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
wall_F = F_env[wall_mask]
obst_F = F_env[~wall_mask]


# Collision point cloud: env.npy is obstacles-only, so resample densely over the
# FULL environment mesh (obstacles + walls) -- everything counts as an obstacle.
env_collision_pts = np.ascontiguousarray(
    sample_surface_points(V_env_n, F_env, ENV_COLLISION_POINTS), dtype=np.float64)


test_list = []
test_list_speed = []
for i in range(500):
    curr = torch.tensor(arr[i]).to(DEVICE)
    test_list.append(curr)
    test_list_speed.append(min(arr_speeds[i]))


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint validity filter
# ──────────────────────────────────────────────────────────────────────────────
# A test case is only meaningful if the shape is collision-free at BOTH the start
# and the goal pose -- if either endpoint is already inside the environment the
# case is unsolvable by construction and would otherwise be charged to the
# planner as a failure.  Those cases are dropped here (before the rollouts, so no
# GPU time is spent on them) and excluded from every statistic below, making the
# reported success rate the actual rate over solvable cases.
n_cases = len(test_list)
eval_list = []
eval_idx = []          # original index in test_list, for logging / speed lookup
n_invalid_endpoints = 0
n_invalid_start = 0
n_invalid_goal = 0

filter_start = timer()
for i, curr in enumerate(test_list):
    XP = curr.reshape(1, 2 * DIM)
    start_bad = check_config_collision(
        XP[0, 0:DIM], env_collision_pts, shape_V, shape_F, shape_radius)
    goal_bad = check_config_collision(
        XP[0, DIM:2 * DIM], env_collision_pts, shape_V, shape_F, shape_radius)
    if start_bad or goal_bad:
        n_invalid_endpoints += 1
        n_invalid_start += int(start_bad)
        n_invalid_goal += int(goal_bad)
        print(f"[{i:03d}] SKIP  start_collision={start_bad}  goal_collision={goal_bad}")
        continue
    eval_list.append(curr)
    eval_idx.append(i)

print(f"\nEndpoint check took {timer() - filter_start:.1f}s: "
      f"{n_invalid_endpoints} / {n_cases} cases discarded "
      f"(start in collision: {n_invalid_start}, goal in collision: {n_invalid_goal}); "
      f"{len(eval_list)} remain.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Batched evaluation loop (MPPI rollouts run EPISODE_BATCH at a time on the GPU)
# ──────────────────────────────────────────────────────────────────────────────
total = 0
total_flip = 0         # successes of the flipped (goal -> start) attempt alone
total_either = 0       # successes of EITHER attempt -- the 2-path planner
n_rescued = 0          # regular failed but flipped succeeded
n_collision = 0
n_no_conv = 0
n_collision_flip = 0
n_no_conv_flip = 0
n_lin_valid = 0       # cases where the straight-line start->goal path is collision-free
n_succ_lin_valid = 0  # planner successes among cases where linear interp is valid
n_succ_lin_invalid = 0  # planner successes among cases where linear interp is invalid
success_list = []
fail_list = []
failed_idx = []        # original test-case index of every FAILED episode
failed_both_idx = []   # index of every case where BOTH regular AND flipped failed
rescued_idx = []       # index of every case the flip rescued (regular fail, flip pass)
test_list_speed_failed = []
episodes = []          # per-episode data for the interactive viser viewer


two_pi = 2 * np.pi
n_total = len(eval_list)          # solvable cases only (invalid endpoints dropped)
eval_start = timer()

for chunk_start in range(0, n_total, EPISODE_BATCH):
    chunk = eval_list[chunk_start:chunk_start + EPISODE_BATCH]
    XPb = torch.stack([c.reshape(2 * DIM) for c in chunk], dim=0)   # (B, 2*DIM)
    # Second attempt at the same case, planned BACKWARDS (goal -> start).  Rows are
    # [start | goal], so flipping is a half-swap.  tau is symmetric by construction
    # but the MPPI rollout is not: it is stochastic and descends the field from a
    # different end, so a case whose forward rollout gets trapped can still be
    # solvable in reverse.  Two paths are generated per case and the case counts as
    # solved if EITHER succeeds.
    XPb_flip = torch.cat([XPb[:, DIM:2 * DIM], XPb[:, 0:DIM]], dim=1)


    # Batched MPPI rollout for the whole chunk (network-heavy part, on GPU).
    with torch.no_grad():
        points_list, iters, successes = MPPI_batched(womodel, XPb.clone(), dim=DIM)
        points_list_f, iters_f, successes_f = MPPI_batched(
            womodel, XPb_flip.clone(), dim=DIM)


    # Model-predicted speed at every waypoint of every episode, computed in one
    # batched Speed() call for the chunk.  Speed() runs network.out + gradient of
    # tau w.r.t. the current config and returns 1/||grad tau||; it needs autograd
    # so it is called OUTSIDE the no_grad block.  point holds the normalized
    # configs the network was trained on (rotvec stored / 2*pi), fed directly.
    cur_per_ep = []
    lengths = []
    for b, point in enumerate(points_list):
        cur = torch.cat(point, dim=0)                              # (T, DIM)
        goal = XPb[b, DIM:2 * DIM].unsqueeze(0).expand(cur.shape[0], -1)
        cur_per_ep.append(torch.cat([cur, goal], dim=1))           # (T, 2*DIM)
        lengths.append(cur.shape[0])
    big = torch.cat(cur_per_ep, dim=0)
    speeds_all = womodel.function.Speed(big).detach().cpu().numpy()
    # Split the flat speeds back into per-episode arrays.
    speeds_split = np.split(speeds_all, np.cumsum(lengths)[:-1])


    for b, (point, iter, success) in enumerate(zip(points_list, iters, successes)):
        cnt2 = eval_idx[chunk_start + b]     # original test-case index
        XP = chunk[b].reshape(1, 2 * DIM)
        # displacement goal - start over all 6 DOF (position part used for scatter)
        diff = (XP[0, DIM:2 * DIM] - XP[0, 0:DIM]).detach().cpu().numpy().tolist()


        speeds = speeds_split[b]                                    # (T,)


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


        collision = check_trajectory_collision(
            point, env_collision_pts, shape_V, shape_F, shape_radius)
        did_not_converge = not success


        # A case can independently collide AND fail to converge -- count both.
        if collision:
            n_collision += 1
        if did_not_converge:
            n_no_conv += 1


        # ── Flipped attempt (goal -> start), scored by the same rules ──
        collision_f = check_trajectory_collision(
            points_list_f[b], env_collision_pts, shape_V, shape_F, shape_radius)
        did_not_converge_f = not successes_f[b]
        if collision_f:
            n_collision_flip += 1
        if did_not_converge_f:
            n_no_conv_flip += 1
        ok_flip = successes_f[b] and not collision_f


        # Linear-interpolation baseline: is the straight line from start to goal a
        # collision-free path on its own?  (Independent of what the planner found.)
        lin_traj = linear_interp_traj(XP[0, 0:DIM], XP[0, DIM:2 * DIM])
        lin_collision = check_trajectory_collision(
            lin_traj, env_collision_pts, shape_V, shape_F, shape_radius)
        lin_valid = not lin_collision
        if lin_valid:
            n_lin_valid += 1


        ok = success and not collision
        ok_either = ok or ok_flip
        if ok_flip:
            total_flip += 1
        if ok_either:
            total_either += 1
        if ok_either and not ok:
            n_rescued += 1
            rescued_idx.append(cnt2)
        if not ok_either:
            # Hard failure: the flip did not rescue it either.
            failed_both_idx.append(cnt2)
        status = 'success' if ok else 'fail'
        episodes.append({
            'idx': cnt2,
            'status': status,
            'status_flip': 'success' if ok_flip else 'fail',
            'waypoints': waypoints,
            'begin_cfg': begin_cfg,
            'end_cfg': end_cfg,
            'speeds': speeds,
        })


        print(
            f"[{cnt2:03d}] {'PASS' if ok else 'FAIL'}  "
            f"flipped={'PASS' if ok_flip else 'FAIL'}  "
            f"either={'PASS' if ok_either else 'FAIL'}  "
            f"did_not_converge={did_not_converge}  "
            f"collision={collision}  "
            f"linear_interp_valid={lin_valid}")


        if ok:
            success_list.append(diff[:3])
            total += 1
            if lin_valid:
                n_succ_lin_valid += 1
            else:
                n_succ_lin_invalid += 1
        else:
            fail_list.append(diff[:3])
            failed_idx.append(cnt2)
            test_list_speed_failed.append(test_list_speed[cnt2])


print(f"\nEvaluation (rollouts) took {timer() - eval_start:.1f}s "
      f"for {n_total} episodes in chunks of {EPISODE_BATCH}.")


success_rate = total / n_total if n_total else 0.0
# Conditional planner success rates: how often the planner succeeds among cases
# where the trivial straight-line (linear interp) path is valid vs. invalid.
n_lin_invalid = n_total - n_lin_valid
lin_valid_success_rate = n_succ_lin_valid / n_lin_valid if n_lin_valid else 0.0
lin_invalid_success_rate = n_succ_lin_invalid / n_lin_invalid if n_lin_invalid else 0.0
flip_rate = total_flip / n_total if n_total else 0.0
either_rate = total_either / n_total if n_total else 0.0
print(f"discarded (start/goal in collision): {n_invalid_endpoints} / {n_cases}")
print(f"total: {total} / {n_total}  ({success_rate:.1%})")
print(f"\n--- two-path planner (regular + flipped) ---")
print(f"  step 1  regular (start->goal): {total} / {n_total}  ({success_rate:.1%})")
print(f"          flipped (goal->start): {total_flip} / {n_total}  ({flip_rate:.1%})")
print(f"  step 2  EITHER succeeded     : {total_either} / {n_total}  ({either_rate:.1%})")
print(f"          rescued by the flip  : {n_rescued}  "
      f"(+{either_rate - success_rate:.1%} over regular alone)")
print(f"          failed BOTH ways     : {len(failed_both_idx)} / {n_total}  "
      f"({(1 - either_rate):.1%})")
print(f"success | linear interp valid  : {n_succ_lin_valid} / {n_lin_valid}  "
      f"({lin_valid_success_rate:.1%})")
print(f"success | linear interp invalid: {n_succ_lin_invalid} / {n_lin_invalid}  "
      f"({lin_invalid_success_rate:.1%})")


# Success-rate summary written into the output directory.
summary_lines = [
    f"data_path                    : {dataPath}",
    f"model_path                   : {modelPath}",
    f"checkpoint                   : {pt}",
    f"test_cases                   : {n_cases}",
    f"discarded_invalid_endpoints  : {n_invalid_endpoints}"
    f"  (start: {n_invalid_start}, goal: {n_invalid_goal})",
    f"episodes                     : {n_total}   [evaluated = test_cases - discarded]",
    f"successes                    : {total}",
    f"failures                     : {n_total - total}",
    f"  collision                  : {n_collision}",
    f"  no_converge                : {n_no_conv}",
    f"success_rate                 : {success_rate:.4f}  ({success_rate:.1%})",
    f"successes_flipped            : {total_flip}"
    f"  (collision: {n_collision_flip}, no_converge: {n_no_conv_flip})",
    f"success_rate_flipped         : {flip_rate:.4f}  ({flip_rate:.1%})",
    f"successes_either             : {total_either}   [regular OR flipped]",
    f"success_rate_either          : {either_rate:.4f}  ({either_rate:.1%})",
    f"rescued_by_flip              : {n_rescued}",
    f"failures_both                : {len(failed_both_idx)}"
    f"   [regular AND flipped both failed]",
    f"failure_rate_both            : {1 - either_rate:.4f}  ({1 - either_rate:.1%})",
    f"linear_interp_valid          : {n_lin_valid} / {n_total}",
    f"success_rate|lin_interp_valid: {lin_valid_success_rate:.4f}  "
    f"({lin_valid_success_rate:.1%})  [{n_succ_lin_valid}/{n_lin_valid}]",
    f"success_rate|lin_interp_fail : {lin_invalid_success_rate:.4f}  "
    f"({lin_invalid_success_rate:.1%})  [{n_succ_lin_invalid}/{n_lin_invalid}]",
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
# Failure frequency tables over the (speed_dist x speed_angle) grid
# ──────────────────────────────────────────────────────────────────────────────
# Where the FAILED episodes sit in clearance space -- one table binned by the
# start's clearance, one by the goal's.  Both cover the same set of failed cases
# (an episode fails as a whole), so the two grand totals are equal; what differs
# is which endpoint's clearance is used as the coordinate.  Bins are uniform over
# [0,1], the range both proxies are already normalized to.  Cases discarded by
# the endpoint filter never ran, so they are absent from both.
def print_speed_freq_table(title, rows, sd_col, sa_col, html_name=None):
    """Frequency of `rows` over the speed_dist x speed_angle grid, with margins.

    If `html_name` is given, the same grid is also written to OUTPUT_DIR as a
    plotly heatmap (rows = angular-speed bin, cols = translational-speed bin).
    """
    if arr_speed_dists is None or arr_speed_angles is None:
        print(f'\n{title}: skipped (speed_dists.npy / speed_angles.npy not in {dataPath})')
        return
    if not rows:
        print(f'\n{title}: none')
        return
    idx = np.asarray(rows)
    sd = np.asarray(arr_speed_dists)[idx, sd_col].astype(np.float64)
    sa = np.asarray(arr_speed_angles)[idx, sa_col].astype(np.float64)

    nb_d, nb_a = args.nbins_dist, args.nbins_angle
    di = np.clip((sd * nb_d).astype(np.int64), 0, nb_d - 1)
    ai = np.clip((sa * nb_a).astype(np.int64), 0, nb_a - 1)
    grid = np.zeros((nb_a, nb_d), dtype=np.int64)
    np.add.at(grid, (ai, di), 1)

    print('\n' + '=' * (13 + 8 * nb_d + 9))
    print(f'{title}   N = {len(idx)}')
    print('rows = speed_angle bin, cols = speed_dist bin  (0 = tight -> 1 = open)')
    print('=' * (13 + 8 * nb_d + 9))
    hdr = ' angle\\dist '
    for d in range(nb_d):
        hdr += '%8s' % ('%.2f' % ((d + 0.5) / nb_d))
    print(hdr + '  |   total')
    for a in range(nb_a - 1, -1, -1):                 # high angle on top
        row = '%11s ' % ('%.2f' % ((a + 0.5) / nb_a))
        for d in range(nb_d):
            row += '%8d' % grid[a, d]
        print(row + '  |%8d' % grid[a].sum())
    print(' ' * 12 + '-' * (8 * nb_d + 11))
    row = '%11s ' % 'total'
    for d in range(nb_d):
        row += '%8d' % grid[:, d].sum()
    print(row + '  |%8d' % grid.sum())

    if html_name is not None:
        d_centers = [(d + 0.5) / nb_d for d in range(nb_d)]
        a_centers = [(a + 0.5) / nb_a for a in range(nb_a)]
        fig = go.Figure(go.Heatmap(
            z=grid, x=d_centers, y=a_centers, colorscale='Reds',
            text=grid, texttemplate='%{text}', hovertemplate=(
                'trans speed %{x:.2f}<br>angular speed %{y:.2f}'
                '<br>count %{z}<extra></extra>'),
            colorbar=dict(title='count')))
        fig.update_layout(title=f'{title}   (N = {len(idx)})',
                          xaxis_title='speed_dist bin (0 = tight -> 1 = open)',
                          yaxis_title='speed_angle bin (0 = tight -> 1 = open)')
        path = os.path.join(OUTPUT_DIR, html_name)
        fig.write_html(path, include_plotlyjs='cdn')
        print('Wrote ' + path)


print_speed_freq_table('FREQUENCY: FAILED episodes, binned by START clearance',
                       failed_idx, 0, 0)
print_speed_freq_table('FREQUENCY: FAILED episodes, binned by END POINT clearance',
                       failed_idx, 1, 1)


# ── Hard failures: neither the regular nor the flipped rollout solved the case ──
# These are the cases the flip could not rescue, so they isolate genuinely hard
# geometry rather than an unlucky rollout direction.  Same grid as above, one
# table/heatmap per endpoint.
print_speed_freq_table(
    'FREQUENCY: FAILED BOTH WAYS (regular AND flipped), binned by START clearance',
    failed_both_idx, 0, 0, html_name='summary_failed_both_start.html')
print_speed_freq_table(
    'FREQUENCY: FAILED BOTH WAYS (regular AND flipped), binned by END POINT clearance',
    failed_both_idx, 1, 1, html_name='summary_failed_both_end.html')


# ── Rescued: the regular run failed but the flipped run solved it ──
# These isolate the cases where the planning DIRECTION is what mattered.  The
# test pairs are symmetric by construction (preprocess_obj --testing_data draws
# both endpoints as independent uniform collision-free placements), so a global
# directional preference cannot show up in the marginal rates -- it can only show
# up here, in WHERE the rescued cases sit in clearance space.
#
# Read the two tables against each other: if the flip helps because entering a
# tight goal is harder than leaving a tight start, the rescued mass concentrates
# in LOW bins of the END table and HIGH bins of the START table.  If instead the
# two tables look alike (and alike to the overall clearance distribution), the
# rescues are just MPPI's per-rollout randomness and carry no directional signal.
#
# The baseline below is what makes the rescued tables readable: it is the same
# grid over EVERY evaluated case, so "the rescues sit in the low-clearance bins"
# can be told apart from "most cases sit in the low-clearance bins".  Compare
# rescued-cell / baseline-cell, not the raw rescued counts.
print_speed_freq_table(
    'BASELINE: ALL evaluated episodes, binned by START clearance',
    eval_idx, 0, 0)
print_speed_freq_table(
    'BASELINE: ALL evaluated episodes, binned by END POINT clearance',
    eval_idx, 1, 1)


print_speed_freq_table(
    'FREQUENCY: RESCUED BY FLIP (regular fail, flipped pass), binned by START clearance',
    rescued_idx, 0, 0, html_name='summary_rescued_start.html')
print_speed_freq_table(
    'FREQUENCY: RESCUED BY FLIP (regular fail, flipped pass), binned by END POINT clearance',
    rescued_idx, 1, 1, html_name='summary_rescued_end.html')


# ──────────────────────────────────────────────────────────────────────────────
# Interactive viser viewer — browse to http://<server-ip>:8080 from your host PC
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nSummary plots saved to {OUTPUT_DIR}/")
if args.no_viser:
    print('--no-viser set; skipping the interactive viewer.')
else:
    launch_viser(episodes, shape_V, shape_F, V_env_n, obst_F, wall_F)
