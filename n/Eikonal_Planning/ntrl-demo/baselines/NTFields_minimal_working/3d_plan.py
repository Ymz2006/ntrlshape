#!/usr/bin/env python3
"""Path inference for the NTFields 3-D shape (SE(3)) models.

Takes a checkpoint trained by ``train_3dshape.py`` and plans the 1000 start/goal
pairs of the matching test set, reporting per-case wall-clock planning time and
path length so the numbers line up with the main repo's
``ntrl-demo/RRT_experiments.md`` case for case.

The controller is the MPPI rollout from ``ntrl-demo/evaluate_training_3d.py``,
adapted to NTFields:

* the cost-to-go comes from ``models.model_3d.Model.TravelTimes`` instead of
  ``models.metric.model_train_metric.Model.function.TravelTimes`` -- the two have
  the same ``(N, 2*dim) -> (N,)`` signature, so the sampler is otherwise
  untouched (50 samples, horizon 5, 0.015 stride cap, momentum 2.0, softmax
  temperature -50, cost ``10*tau(first) + tau(last)``, 0.01 convergence ball);
* the rollout is vectorized over a batch of episodes, so ``--batch`` > 1 runs
  several pairs' rollouts in one set of GPU launches.  Timing is per episode
  only when ``--batch 1`` (the default); larger batches report the amortized
  chunk time and are flagged as such in the summary;
* the device is a parameter rather than a hardcoded ``.cuda()``.

NTFields plans in the same normalized frame the datasets are stored in
(``(x, y, z, rx, ry, rz)`` with the rotvec divided by 2*pi and the translation in
units of ``meta.env_scale``), so ``--data-scale`` is not needed here: unlike the
bundled Aloha example, nothing is divided by 10.

Success is the same two-part test ``evaluate_training_3d.py`` applies: the
rollout has to reach the 0.01 goal ball, and the placed shape must be
collision-free at every recorded waypoint.  Collision uses the point-in-tet test
that labels the training data (``preprocess_obj.points_inside_tets``) over a
dense surface sampling of the whole environment mesh, walls included, with a
KD-tree broad phase -- the checker of ``baseline_ompl/rrt_connect_eval.py``.

Two path lengths are recorded per case:

* ``path_length`` -- OMPL's SE(3) metric, ``sum ||dt|| + acos(|q_i . q_i+1|)``
  over consecutive waypoints in the normalized frame.  This is exactly what
  ``rrt_connect_eval.py`` reports, so the two tables are comparable.
* ``cfg_length``  -- the plain Euclidean length in the normalized 6-D config
  space the network actually plans in.  Recorded in the CSV only.

Run from this directory (the repository root must be visible so that the sibling
``ntrl-demo/`` datasets and ``dataprocessing`` package can be found):

    python 3d_plan.py --env rectangle_env1 --device cuda:0
    python 3d_plan.py --env rectangle_env1 --cases 20 --no-collision   # smoke test
"""

import argparse
import json
import os
import sys
import time
from glob import glob
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.abspath(__file__))
# The 3-D shape datasets, meshes and the ``dataprocessing`` package all live in
# the sibling ntrl-demo checkout; train_3dshape.py reaches for it the same way.
NTRL_ROOT = os.path.abspath(os.path.join(ROOT, os.pardir, os.pardir,
                                         'ntrl-demo', 'ntrl-demo'))
for _p in (ROOT, NTRL_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)

from models.model_3d import Model                                    # noqa: E402
from dataprocessing.preprocess_obj import (                          # noqa: E402
    DEFAULT_TET_SWITCHES, load_obj, sample_surface_points,
    tetrahedralize_shape)

DIM = 6
TWO_PI = 2.0 * np.pi
# Surface points sampled over the FULL environment mesh for collision checking
# (the count evaluate_training_3d.py and rrt_connect_eval.py both use).
ENV_COLLISION_POINTS = 50000
# preprocess_obj.py ``--2d`` draws placements with z == 0 and a rotvec of
# (0, 0, rz), so the only free coordinates of the normalized 6-D config are
# x, y and rz.  A planar rollout must sample in that same sub-space.
PLANAR_FREE_DIMS = (0, 1, 5)
DEFAULT_TEST_ROOT = os.path.join(NTRL_ROOT, 'testing_data', '3dshape')
DEFAULT_MESH_ROOT = os.path.join(NTRL_ROOT, 'datasets', '3dshape')


# ──────────────────────────────────────────────────────────────────────────────
# Collision checking  (ported from baseline_ompl/rrt_connect_eval.py)
# ──────────────────────────────────────────────────────────────────────────────
class ShapeCollisionChecker:
    """Point-in-tet collision test for a rigid shape moving in a point-cloud env.

    ``tets_local`` is (K, 4, 3) in the shape's local frame.  A world point is
    mapped into that frame with ``local = (world - t) @ R`` (the inverse of the
    placement ``world = local @ R.T + t``) and tested against every tet via
    barycentric coordinates -- exactly ``preprocess_obj.points_inside_tets``.
    """

    def __init__(self, tets_local, env_points, radius, tol=-1e-6):
        self.env_points = np.ascontiguousarray(env_points, dtype=np.float64)
        self.tree = cKDTree(self.env_points)
        self.radius = float(radius)
        self.tol = tol

        tets = np.asarray(tets_local, dtype=np.float64)
        self.v0 = tets[:, 0, :]                                  # (K,3)
        M = np.stack([tets[:, 1, :] - self.v0,
                      tets[:, 2, :] - self.v0,
                      tets[:, 3, :] - self.v0], axis=-1)         # (K,3,3) columns
        self.Minv = np.linalg.inv(M)                             # (K,3,3)
        self.n_checks = 0

    def in_collision(self, t, R):
        """True iff any env point lies inside the shape placed at (t, R)."""
        self.n_checks += 1
        idx = self.tree.query_ball_point(t, self.radius)
        if not idx:
            return False
        near_local = (self.env_points[idx] - t) @ R              # (M,3)
        rhs = near_local[None, :, :] - self.v0[:, None, :]       # (K,M,3)
        bary = np.einsum('kij,kmj->kmi', self.Minv, rhs)         # (K,M,3)
        l0 = 1.0 - bary.sum(axis=-1)
        inside = ((bary >= self.tol).all(axis=-1)) & (l0 >= self.tol)
        return bool(inside.any())

    def path_in_collision(self, waypoints):
        """True if ANY waypoint of a (T, 6) normalized-config path collides.

        No interpolation between waypoints -- each is checked on its own, as in
        ``evaluate_training_3d.check_trajectory_collision``.  MPPI's stride is
        capped at 0.015, so consecutive waypoints are already dense.
        """
        rots = Rotation.from_rotvec(waypoints[:, 3:6] * TWO_PI).as_matrix()
        for i in range(len(waypoints)):
            if self.in_collision(waypoints[i, 0:3].astype(np.float64), rots[i]):
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Path metrics
# ──────────────────────────────────────────────────────────────────────────────
def se3_path_length(waypoints):
    """OMPL's SE(3) path length of a (T, 6) normalized-config polyline.

    ``ompl::base::SE3StateSpace`` is a compound space whose R^3 and SO(3)
    subspaces both carry weight 1, and ``SO3StateSpace::distance`` returns
    ``acos(|q1 . q2|)``.  Summing that over consecutive waypoints is exactly what
    ``ompl::geometric::PathGeometric::length()`` -- and therefore
    ``rrt_connect_eval.py``'s ``path_length`` -- measures.
    """
    if len(waypoints) < 2:
        return 0.0
    trans = np.diff(waypoints[:, 0:3], axis=0)
    d_trans = float(np.linalg.norm(trans, axis=1).sum())
    quats = Rotation.from_rotvec(waypoints[:, 3:6] * TWO_PI).as_quat()
    dots = np.abs(np.einsum('ij,ij->i', quats[:-1], quats[1:]))
    d_rot = float(np.arccos(np.clip(dots, 0.0, 1.0)).sum())
    return d_trans + d_rot


def cfg_path_length(waypoints):
    """Euclidean length in the normalized 6-D config space the network plans in."""
    if len(waypoints) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())


# ──────────────────────────────────────────────────────────────────────────────
# MPPI controller  (adapted from ntrl-demo/evaluate_training_3d.py)
# ──────────────────────────────────────────────────────────────────────────────
def mppi(model, XP, dim, steps=200, sample_num=50, horizon=5,
         step=0.015, momentum=2.0, goal_tol=0.01, planar=False):
    """MPPI rollout for ``B`` start/goal pairs at once.

    ``XP`` is a (B, 2*dim) tensor on the run device, each row
    ``[start(dim) | goal(dim)]`` in the normalized frame (rotvec / 2*pi).  With
    ``B = 1`` this is the single-episode controller of ``evaluate_training_3d.py``
    step for step; the leading batch axis is the only change.

    Every iteration draws ``sample_num`` displacement sequences of length
    ``horizon`` (a per-sequence offset plus per-step noise, both at scale
    ``step``), biases them by ``momentum`` times the last accepted step, clamps
    each to ``step`` in norm, and scores the first and last horizon config of
    each sample by cost-to-go under the learned field:

        cost = 10 * tau(cand_first -> goal) + tau(cand_last -> goal)

    The executed step is the softmax(-50 * cost)-weighted mean of the samples'
    first displacements.  An episode whose config reaches within ``goal_tol`` of
    the goal is frozen (the batched equivalent of breaking out of the loop);
    other episodes keep stepping.

    ``planar`` zeroes the displacement outside PLANAR_FREE_DIMS before the
    magnitude clamp, so z / rx / ry keep their start values for the whole
    rollout -- required on test sets generated with ``preprocess_obj.py --2d``.

    Returns:
        paths   : list (len B) of (T_b, dim) numpy arrays, the recorded configs
                  with the goal config appended last.
        iters   : list (len B) of the convergence iteration index.
        success : list (len B) of bools (reached the goal ball).
        min_dis : (B,) numpy array, the closest the rollout ever came to the goal.
    """
    B = XP.shape[0]
    dev = XP.device
    if planar and dim != 6:
        raise ValueError('planar mode assumes the SE(3) layout (x,y,z,rx,ry,rz); '
                         'got dim={}'.format(dim))
    free_mask = None
    if planar:
        free_mask = torch.zeros(dim, device=dev)
        free_mask[list(PLANAR_FREE_DIMS)] = 1.0

    dP_prior = torch.zeros((B, dim), device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    min_dis = torch.norm(XP[:, dim:2 * dim] - XP[:, 0:dim], dim=1)      # (B,)

    # recorded[k] is the (B, dim) current config after k updates.
    recorded = [XP[:, 0:dim].clone()]

    for it in range(steps):
        # (B, sample_num, horizon, 2*dim)
        XP_tmp = XP[:, None, None, :].repeat(1, sample_num, horizon, 1)

        dP = step * torch.normal(0, 1, size=(B, sample_num, 1, dim),
                                 dtype=torch.float32, device=dev) \
            + step * torch.normal(0, 1, size=(B, sample_num, horizon, dim),
                                  dtype=torch.float32, device=dev)
        if momentum:
            dP = dP + momentum * dP_prior[:, None, None, :]
        if free_mask is not None:
            # Project onto the planar sub-space BEFORE the magnitude clamp, so
            # the norm below is taken over the free coordinates alone and every
            # downstream quantity inherits the zeros.
            dP = dP * free_mask
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=step) / step)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + torch.cumsum(dP, dim=2)

        # First and last horizon config of every sample -> (B, sample_num, 2, 2*dim).
        endpoints = XP_tmp[:, :, [0, -1], :]
        cost = model.TravelTimes(endpoints.reshape(-1, 2 * dim))
        cost = cost.reshape(B, sample_num, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]                       # (B, S)

        weight = torch.softmax(-50 * cost, dim=1)                       # (B, S)
        # Weighted mean of the first-step displacement over samples -> (B, dim).
        dP_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)

        # Freeze converged episodes: only advance those not yet done.
        XP[:, 0:dim] = XP[:, 0:dim] + dP_prior * (~done).unsqueeze(1)

        dis = torch.norm(XP[:, dim:2 * dim] - XP[:, 0:dim], dim=1)      # (B,)
        min_dis = torch.minimum(min_dis, dis)
        recorded.append(XP[:, 0:dim].clone())

        newly = (dis < goal_tol) & (~done)
        conv_step[newly] = it
        done = done | newly
        if bool(done.all()):
            break

    success = done.detach().cpu().numpy()
    conv = conv_step.detach().cpu().numpy()
    traj = torch.stack(recorded, dim=1).detach().cpu().numpy()          # (B,K,dim)
    goal = XP[:, dim:2 * dim].detach().cpu().numpy()                    # (B,dim)

    paths = []
    for b in range(B):
        # start + (conv_step+1) updates -> recorded[0 .. conv_step+1].
        end = int(conv[b]) + 2 if success[b] else traj.shape[1]
        paths.append(np.concatenate([traj[b, :end], goal[b:b + 1]], axis=0))
    return paths, conv.tolist(), success.tolist(), min_dis.detach().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
def resolve_device(requested):
    if requested == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is false')
    return requested


def resolve_mesh(path, mesh_root):
    """Locate a mesh recorded in meta.json.

    ``meta.json`` stores the absolute path the preprocessing container saw
    (``/workspace/ntrl-demo/datasets/3dshape/<name>.obj``), which does not exist
    outside it, so fall back to the basename under ``mesh_root``.
    """
    if os.path.exists(path):
        return path
    alt = os.path.join(mesh_root, os.path.basename(path))
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(
        '{} not found, and no {} under {}'.format(path, os.path.basename(path), mesh_root))


def find_checkpoint(args):
    """The checkpoint to plan with: --checkpoint, else <model-root>/<env>/latest.pt."""
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        return args.checkpoint
    latest = os.path.join(args.model_root, args.env, 'latest.pt')
    if os.path.exists(latest):
        return latest
    hits = sorted(glob(os.path.join(args.model_root, args.env, 'Model_Epoch_*.pt')))
    if not hits:
        raise FileNotFoundError(
            'no latest.pt and no Model_Epoch_*.pt under {}'.format(
                os.path.join(args.model_root, args.env)))
    return hits[-1]


def build_scene(data_path, mesh_root, seed):
    """Load the meshes named by meta.json and build the collision checker."""
    with open(os.path.join(data_path, 'meta.json')) as fh:
        meta = json.load(fh)
    env_scale = float(meta['env_scale'])
    env_center = np.asarray(meta['env_center'], dtype=np.float64)
    shape_scale = float(meta.get('shape_scale', 1.0))

    # Shape: centre on its bbox and normalize, as preprocess_obj / evaluate do.
    shape_obj = resolve_mesh(meta['shape_obj'], mesh_root)
    V_sh, F_sh, _ = load_obj(shape_obj)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                                   dtype=np.float64)
    shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
    shape_radius = float(np.linalg.norm(shape_V, axis=1).max())

    # 'Q' = quiet, so tetgen's page of statistics stays out of the log.
    TV, TT, _ = tetrahedralize_shape(shape_V, shape_F,
                                     switches=DEFAULT_TET_SWITCHES + 'Q')
    tets_local = np.asarray(TV, dtype=np.float64)[TT]                  # (K,4,3)

    # Environment: obstacles *and* walls all count as obstacles.  env.npy holds
    # obstacle points only, so resample the full mesh, seeded for repeatability.
    env_obj = resolve_mesh(meta['env_obj'], mesh_root)
    V_env, F_env, _ = load_obj(env_obj)
    V_env_n = (V_env - env_center) / env_scale
    np.random.seed(seed)
    env_pts = np.ascontiguousarray(
        sample_surface_points(V_env_n, F_env, ENV_COLLISION_POINTS), dtype=np.float64)

    checker = ShapeCollisionChecker(tets_local, env_pts, shape_radius)
    print('shape       : {}  ({} verts, {} tets, radius {:.4f})'.format(
        shape_obj, len(shape_V), len(TT), shape_radius))
    print('environment : {}  ({} tris, {} collision points)'.format(
        env_obj, len(F_env), len(env_pts)))
    return meta, checker


def mean_std(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return float('nan'), float('nan')
    return float(a.mean()), float(a.std())


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env', default='rectangle_env1',
                        help='Environment name; selects both the checkpoint '
                             '(<model-root>/<env>/latest.pt) and the test set '
                             '(<test-root>/<env>).')
    parser.add_argument('--model-root', default=os.path.join(ROOT, 'outputs', '3dshape'))
    parser.add_argument('--checkpoint', default=None,
                        help='Explicit .pt to plan with; overrides --model-root/--env.')
    parser.add_argument('--test-root', default=DEFAULT_TEST_ROOT)
    parser.add_argument('--data', default=None,
                        help='Full test-set directory; overrides --test-root/--env.')
    parser.add_argument('--mesh-root', default=DEFAULT_MESH_ROOT,
                        help='Where to look for the meshes meta.json names, when '
                             'the absolute path it recorded does not exist here.')
    parser.add_argument('--out', default=None,
                        help='Defaults to outputs/3dplan/<env>.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--model-size', type=int, choices=(0, 1, 2), default=2)
    parser.add_argument('--cases', type=int, default=0,
                        help='How many start/goal pairs to plan; 0 = all 1000.')
    parser.add_argument('--batch', type=int, default=1,
                        help='Episodes whose rollouts run together on the GPU. '
                             '1 (the default) is the only setting whose per-case '
                             'times are true per-case wall clock; larger batches '
                             'report the amortized chunk time.')
    parser.add_argument('--steps', type=int, default=200,
                        help='Cap on MPPI iterations per episode.')
    parser.add_argument('--samples', type=int, default=50, help='MPPI samples per step.')
    parser.add_argument('--horizon', type=int, default=5, help='MPPI rollout horizon.')
    parser.add_argument('--step', type=float, default=0.015,
                        help='Per-sample displacement cap in the normalized 6-D '
                             'config space (the convergence ball is 0.01).')
    parser.add_argument('--momentum', type=float, default=2.0,
                        help='Gain on the previous accepted step, added to every '
                             'sample before the magnitude clamp; 0 disables it.')
    parser.add_argument('--goal-tol', type=float, default=0.01,
                        help='Convergence ball radius in the normalized 6-D space.')
    parser.add_argument('--2d', dest='two_d', default=None, action='store_true',
                        help='Force planar rollouts (x, y, rz only). Read from '
                             'meta.json ("two_d") when not given.')
    parser.add_argument('--no-collision', dest='collision', action='store_false',
                        help='Skip collision checking; success is convergence only. '
                             'Useful for a quick smoke test, not for a real number.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4,
                        help='Torch intra-op threads; keeps concurrent runs from '
                             'thrashing the CPU.')
    args = parser.parse_args()

    data_path = args.data if args.data is not None else os.path.join(args.test_root, args.env)
    out_dir = args.out if args.out is not None else os.path.join(ROOT, 'outputs', '3dplan', args.env)
    os.makedirs(out_dir, exist_ok=True)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = find_checkpoint(args)
    meta, checker = build_scene(data_path, args.mesh_root, args.seed)
    planar = meta.get('two_d', False) if args.two_d is None else args.two_d

    arr = np.load(os.path.join(data_path, 'sampled_points.npy')).astype(np.float32)
    if arr.shape[1] != 2 * DIM:
        raise ValueError('expected an (N, {}) sampled_points.npy, got {}'.format(
            2 * DIM, arr.shape))
    n_cases = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
    arr = arr[:n_cases]

    model = Model(out_dir, data_path, DIM, [0.0] * DIM, device=device,
                  model_size=args.model_size)
    model.load(checkpoint)
    model.network.eval()

    print('checkpoint  : {}'.format(checkpoint))
    print('test set    : {}  ({} cases)'.format(data_path, n_cases))
    print('device      : {}   planar: {}   batch: {}'.format(device, planar, args.batch))
    print('collision   : {}'.format('on' if args.collision else 'OFF (convergence only)'))

    cuda = device.startswith('cuda')

    # Warm-up: the first CUDA launch of a process pays context creation, kernel
    # autotuning and lazy module loading, which would otherwise land entirely on
    # case 0 (an order of magnitude over its true cost) and skew the mean/sd.
    with torch.no_grad():
        mppi(model, torch.from_numpy(arr[0:1]).to(device), DIM, steps=5,
             sample_num=args.samples, horizon=args.horizon, step=args.step,
             momentum=args.momentum, goal_tol=args.goal_tol, planar=planar)
    if cuda:
        torch.cuda.synchronize(device)

    rows = []            # (idx, success, plan_s, se3_len, cfg_len, waypoints,
                         #  converged, collision, min_dis)
    n_collision = 0
    n_no_conv = 0
    run_start = timer()

    for lo in range(0, n_cases, args.batch):
        hi = min(lo + args.batch, n_cases)
        XP = torch.from_numpy(arr[lo:hi]).to(device)

        if cuda:
            torch.cuda.synchronize(device)
        t0 = timer()
        with torch.no_grad():
            paths, _, converged, min_dis = mppi(
                model, XP.clone(), DIM, steps=args.steps, sample_num=args.samples,
                horizon=args.horizon, step=args.step, momentum=args.momentum,
                goal_tol=args.goal_tol, planar=planar)
        if cuda:
            torch.cuda.synchronize(device)
        # With --batch 1 this is the case's own wall clock; with a larger batch
        # the chunk time is shared out evenly over its episodes.
        per_case = (timer() - t0) / (hi - lo)

        for k, idx in enumerate(range(lo, hi)):
            wp = paths[k]
            collision = bool(checker.path_in_collision(wp)) if args.collision else False
            ok = bool(converged[k]) and not collision
            if collision:
                n_collision += 1
            if not converged[k]:
                n_no_conv += 1
            rows.append((idx, ok, per_case, se3_path_length(wp), cfg_path_length(wp),
                         len(wp), bool(converged[k]), collision, float(min_dis[k])))
            print('[{:04d}] {}  t={:7.3f}s  len={:7.3f}  waypoints={:4d}  '
                  'converged={}  collision={}'.format(
                      idx, 'PASS' if ok else 'FAIL', per_case, rows[-1][3],
                      len(wp), bool(converged[k]), collision))

    total_wall = timer() - run_start
    times = np.array([r[2] for r in rows], dtype=np.float64)
    succ_times = np.array([r[2] for r in rows if r[1]], dtype=np.float64)
    lengths = np.array([r[3] for r in rows if r[1]], dtype=np.float64)
    cfg_lengths = np.array([r[4] for r in rows if r[1]], dtype=np.float64)
    n_succ = int(sum(1 for r in rows if r[1]))
    rate = n_succ / len(rows) if rows else 0.0

    t_mean, t_std = mean_std(times)
    ts_mean, ts_std = mean_std(succ_times)
    l_mean, l_std = mean_std(lengths)
    c_mean, c_std = mean_std(cfg_lengths)

    lines = [
        'env                       : {}'.format(args.env),
        'checkpoint                : {}'.format(checkpoint),
        'data_path                 : {}'.format(data_path),
        'planner                   : NTFields + MPPI (SE(3))',
        'device                    : {}'.format(device),
        'episode_batch             : {}{}'.format(
            args.batch,
            '' if args.batch == 1 else '   [times are amortized over the chunk]'),
        'mppi                      : steps {}  samples {}  horizon {}  step {}  '
        'momentum {}  goal_tol {}'.format(args.steps, args.samples, args.horizon,
                                          args.step, args.momentum, args.goal_tol),
        'planar_rollout            : {}'.format(planar),
        'collision_checking        : {}'.format('on' if args.collision else 'OFF'),
        '',
        'test_cases_scored         : {}'.format(len(rows)),
        'successes                 : {}'.format(n_succ),
        'failures                  : {}'.format(len(rows) - n_succ),
        '  did_not_converge        : {}'.format(n_no_conv),
        '  path_in_collision       : {}'.format(n_collision),
        'success_rate              : {:.4f}  ({:.1%})  [{}/{}]'.format(
            rate, rate, n_succ, len(rows)),
        '',
        'time_mean                 : {:.4f} s   [all {} scored cases]'.format(
            t_mean, len(rows)),
        'time_std                  : {:.4f} s'.format(t_std),
        'time_mean_successful      : {:.4f} s   [{} successful cases]'.format(
            ts_mean, succ_times.size),
        'time_std_successful       : {:.4f} s'.format(ts_std),
        'time_median               : {:.4f} s'.format(
            float(np.median(times)) if times.size else float('nan')),
        '',
        'path_length_mean          : {:.4f}   [{} successful cases, OMPL SE(3) '
        'metric]'.format(l_mean, lengths.size),
        'path_length_std           : {:.4f}'.format(l_std),
        'path_length_total         : {:.4f}'.format(float(lengths.sum())),
        'cfg_length_mean           : {:.4f}   [same cases, Euclidean in the '
        'normalized 6-D config space]'.format(c_mean),
        'cfg_length_std            : {:.4f}'.format(c_std),
        '',
        'total_time                : {:.2f} s   [rollouts + collision checks]'.format(
            total_wall),
        'collision_checks          : {}'.format(checker.n_checks),
    ]
    print()
    print('\n'.join(lines))

    summary = os.path.join(out_dir, 'plan_summary.txt')
    with open(summary, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    csv = os.path.join(out_dir, 'plan_cases.csv')
    with open(csv, 'w') as fh:
        fh.write('idx,success,time_s,path_length,cfg_length,waypoints,'
                 'converged,collision,min_dis\n')
        for idx, ok, t, ln, cl, npts, conv, col, md in rows:
            fh.write('{},{},{:.6f},{:.6f},{:.6f},{},{},{},{:.6f}\n'.format(
                idx, int(ok), t, ln, cl, npts, int(conv), int(col), md))
    print('\nWrote {}\nWrote {}'.format(summary, csv))


if __name__ == '__main__':
    main()
