"""Diagnose WHY the hlB / 'Alternate Bellman Horizon' planner fails.

Re-runs exactly the hlB planner of evaluate_training_3d_batched.py (local_step +
all_horizon, alternating bidirectional) on a test set and then, for every
waypoint of every rollout, measures

    clearance   : signed distance of the closest env surface point to the placed
                  shape  (>0 free, <0 penetrating; -clearance = penetration depth)
    hit_wall    : the penetrating env points belong to the 'wall' group
    speed_full  : network 1/|grad_x0 tau(wp -> goal)|   over all 6 dims
    speed_trans : network 1/|grad_xyz tau(wp -> goal)|  translation only

so that a collision failure can be attributed to

    FIELD   -- the network still predicts a high speed at configs that are in
               (or about to be in) collision, i.e. tau does not see the obstacle;
    PLANNER -- the network does predict a low speed there, but the MPPI update
               (stride 0.015 vs a clearance band of 0.05, 50 samples, softmax
               weighting) steps in anyway.

Optional planner variants (--step, --local-weight, --samples, --speed-gate,
--steps) let the same script measure how much each candidate fix recovers.

Run inside the pytorch docker from the nested ntrl-demo root, e.g.

    python _diag_hlB_collisions.py --dataPath testing_data/3dshape/Ashape3d_env2 \
        --checkpoint ./Experiments/3dshape/3dshape_09_04_09_09/latest.pt \
        --device cuda:2 --cases 1000 --out results/diag_hlB/Ashape3d_env2
"""
import sys
sys.path.append('.')

import os
import json
import argparse
import contextlib
import io
import time
import numpy as np
import torch
import igl

from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import load_obj, _rotvec_to_matrix_np

_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
DIM = 6
ENV_COLLISION_POINTS = 50000
MARGIN = 0.05
CTG_PAIR_CHUNK = 200000

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--modelPath', default='./Experiments/3dshape')
parser.add_argument('--device', default='cuda')
parser.add_argument('--cases', type=int, default=1000)
parser.add_argument('--batch', type=int, default=250)
parser.add_argument('--out', required=True)
parser.add_argument('--seed', type=int, default=0)
# planner variants
parser.add_argument('--step', type=float, default=0.015)
parser.add_argument('--steps', type=int, default=200)
parser.add_argument('--local-weight', dest='local_w', type=float, default=0.03)
parser.add_argument('--samples', type=int, default=50)
parser.add_argument('--speed-gate', dest='speed_gate', type=float, default=0.0,
                    help='If >0: samples whose predicted per-step speed '
                         '(||cand-cur|| / tau(cur->cand)) at ANY horizon step is '
                         'below this get a large cost penalty (hard-ish reject).')
parser.add_argument('--gate-penalty', dest='gate_penalty', type=float, default=1e3)
parser.add_argument('--interp', type=int, default=0,
                    help='If >0, also collision-check this many linear '
                         'interpolants between consecutive waypoints.')
parser.add_argument('--no-field-stats', action='store_true',
                    help='Skip the per-waypoint clearance / speed analysis '
                         '(only the SR of the variant is reported).')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = args.device
os.makedirs(args.out, exist_ok=True)


@contextlib.contextmanager
def _quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# planner (copied from evaluate_training_3d_batched.py, momentum=0 as there)
# ─────────────────────────────────────────────────────────────────────────────
def _candidate_cost(womodel, XP_tmp, cur, dim, local_step, all_horizon, local_w,
                    speed_gate=0.0, gate_penalty=1e3):
    B, S, H, _ = XP_tmp.shape
    idx = list(range(H)) if all_horizon else [0, -1]
    sel = XP_tmp[:, :, idx, :]
    K = sel.shape[2]
    f = womodel.function.TravelTimes(sel.reshape(-1, dim * 2)).reshape(B, S, K)
    if local_step or speed_gate > 0:
        cand = sel[..., 0:dim]
        src = cur[:, None, None, :].expand_as(cand)
        local = womodel.function.TravelTimes(
            torch.cat([src, cand], dim=3).reshape(-1, dim * 2)).reshape(B, S, K)
        seg = torch.norm(cand - src, dim=3).clamp(min=1e-6)
        slow = local / seg
        if local_step:
            f = f + local_w * slow
        if speed_gate > 0:
            bad = (slow > 1.0 / speed_gate).any(dim=2)          # (B, S)
            pen = gate_penalty * bad.float()
        else:
            pen = None
    else:
        pen = None
    if all_horizon:
        c = 10 * f[:, :, 0] + f[:, :, 1:].mean(dim=2)
    else:
        c = 10 * f[:, :, 0] + f[:, :, 1]
    if pen is not None:
        c = c + pen
    return c


def MPPI_alternating_batched(womodel, XP, dim, steps=200, local_step=True,
                             all_horizon=True, local_w=0.03, step=0.015,
                             sample_num=50, speed_gate=0.0, gate_penalty=1e3):
    B = XP.shape[0]
    horizon = 5
    dev = XP.device
    XP = XP.clone()
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    min_dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)
    chain_a = [XP[:, 0:dim].clone()]
    chain_b = [XP[:, dim:dim * 2].clone()]
    n_a = torch.ones(B, dtype=torch.long)
    n_b = torch.ones(B, dtype=torch.long)
    moving_a = True
    for it in range(steps):
        XP_tmp = XP.clone()
        XP_tmp = XP_tmp[:, None, None, :].repeat(1, sample_num, horizon, 1)
        radius = torch.full((B,), step, device=dev)
        r = radius[:, None, None, None]
        dP = step * torch.normal(0, 1, size=(B, sample_num, 1, dim),
                                 dtype=torch.float32, device=dev) \
            + step * torch.normal(0, 1, size=(B, sample_num, horizon, dim),
                                  dtype=torch.float32, device=dev)
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=r) / r)
        dP_cumsum = torch.cumsum(dP, dim=2)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + dP_cumsum
        cost = _candidate_cost(womodel, XP_tmp, XP[:, 0:dim], dim,
                               local_step, all_horizon, local_w,
                               speed_gate, gate_penalty)
        weight = torch.softmax(-50 * cost, dim=1)
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        live = (~done)
        XP[:, 0:dim] = XP[:, 0:dim] + step_prior * live.unsqueeze(1)
        dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)
        min_dis = torch.minimum(min_dis, dis)
        live_cpu = live.detach().cpu().long()
        if moving_a:
            chain_a.append(XP[:, 0:dim].clone())
            n_a = n_a + live_cpu
        else:
            chain_b.append(XP[:, 0:dim].clone())
            n_b = n_b + live_cpu
        newly = (dis < 0.01) & (~done)
        conv_step[newly] = it
        done = done | (dis < 0.01)
        if bool(done.all()):
            break
        XP = torch.cat([XP[:, dim:dim * 2], XP[:, 0:dim]], dim=1)
        moving_a = not moving_a
    success = done.detach().cpu().numpy()
    points_list, na_list, nb_list = [], [], []
    for b in range(B):
        a_len = int(n_a[b]); b_len = int(n_b[b])
        cfgs = [chain_a[k][b, :] for k in range(a_len)]
        cfgs += [chain_b[k][b, :] for k in reversed(range(b_len))]
        points_list.append(torch.stack(cfgs, 0))       # (L, dim)
        na_list.append(a_len); nb_list.append(b_len)
    return points_list, success.tolist(), na_list, nb_list


# ─────────────────────────────────────────────────────────────────────────────
# geometry
# ─────────────────────────────────────────────────────────────────────────────
def sample_surface_points_labeled(V, F, num_points):
    used = np.unique(F.reshape(-1))
    seed = V[used]
    # face owning each seed vertex: first face that references it
    first_face = np.full(V.shape[0], -1, dtype=np.int64)
    for fi in range(F.shape[0] - 1, -1, -1):
        first_face[F[fi]] = fi
    seed_f = first_face[used]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    probs = areas / areas.sum()
    n = max(int(num_points) - len(seed), 0)
    tri = np.random.choice(len(F), size=n, p=probs)
    r1 = np.sqrt(np.random.rand(n, 1)); r2 = np.random.rand(n, 1)
    pts = (1 - r1) * a[tri] + r1 * (1 - r2) * b[tri] + r1 * r2 * c[tri]
    return (np.concatenate([seed, pts], 0), np.concatenate([seed_f, tri], 0))


def waypoint_clearance(cfg, env_pts, env_is_wall, shape_V, shape_F, shape_radius,
                       reach=MARGIN + 0.01):
    """Signed clearance of one config (>0 free), and whether the deepest point is a wall pt.

    Only env points within shape_radius + reach are examined, so a returned
    clearance >= reach means 'at least reach' (reported clipped)."""
    t = cfg[0:3]
    d = np.linalg.norm(env_pts - t, axis=1)
    m = d <= shape_radius + reach
    if not m.any():
        return reach, False, 0
    R = _rotvec_to_matrix_np(cfg[3:6] * 2 * np.pi)
    near_local = np.ascontiguousarray((env_pts[m] - t) @ R)
    S = igl.signed_distance(near_local, shape_V, shape_F, _SDF_SIGN)[0]
    k = int(np.argmin(S))
    n_in = int((S < 0).sum())
    return float(min(S[k], reach)), bool(env_is_wall[m][k]), n_in


# ─────────────────────────────────────────────────────────────────────────────
# setup
# ─────────────────────────────────────────────────────────────────────────────
womodel = md.Model(args.modelPath, args.dataPath, DIM, [0.0] * DIM, device=DEVICE)
womodel.load(args.checkpoint)
womodel.network.eval()

with open(os.path.join(args.dataPath, 'meta.json')) as f:
    meta = json.load(f)
env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)


def _resolve(p):
    if os.path.exists(p):
        return p
    return os.path.join('./datasets/3dshape', os.path.basename(p))


with _quiet():
    V_sh, F_sh, _ = load_obj(_resolve(meta['shape_obj']))
    V_env, F_env, names_env = load_obj(_resolve(meta['env_obj']))
shape_center = 0.5 * (V_sh.min(0) + V_sh.max(0))
shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale, dtype=np.float64)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())
V_env_n = (V_env - env_center) / env_scale
env_pts, env_face = sample_surface_points_labeled(V_env_n, F_env, ENV_COLLISION_POINTS)
env_pts = np.ascontiguousarray(env_pts, dtype=np.float64)
face_is_wall = np.array(['wall' in str(n).lower() for n in names_env])
env_is_wall = face_is_wall[env_face]
print(f'shape_radius={shape_radius:.4f}  env pts={len(env_pts)}  wall frac={env_is_wall.mean():.3f}')

arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy'))
n = min(args.cases, len(arr))
XP_all = torch.tensor(arr[:n], dtype=torch.float32, device=DEVICE)

# endpoint validity filter (same as eval)
valid = []
for i in range(n):
    s = arr[i, :DIM]; g = arr[i, DIM:]
    cs = waypoint_clearance(s, env_pts, env_is_wall, shape_V, shape_F, shape_radius)[0]
    cg = waypoint_clearance(g, env_pts, env_is_wall, shape_V, shape_F, shape_radius)[0]
    valid.append(cs > 0 and cg > 0)
valid = np.array(valid)
print(f'valid cases: {valid.sum()} / {n}')
idx_valid = np.nonzero(valid)[0]

# ─────────────────────────────────────────────────────────────────────────────
# rollouts
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
paths, succ, na_all, nb_all = [], [], [], []
with torch.no_grad():
    for s0 in range(0, len(idx_valid), args.batch):
        ids = idx_valid[s0:s0 + args.batch]
        pl, sc, na, nb = MPPI_alternating_batched(
            womodel, XP_all[ids].clone(), DIM, steps=args.steps, local_w=args.local_w,
            step=args.step, sample_num=args.samples, speed_gate=args.speed_gate,
            gate_penalty=args.gate_penalty)
        paths += [p.detach().cpu().numpy() for p in pl]
        succ += sc; na_all += na; nb_all += nb
        print(f'  rollout {s0 + len(ids)}/{len(idx_valid)}  ({time.time() - t0:.0f}s)')

# ─────────────────────────────────────────────────────────────────────────────
# per-waypoint clearance + collision
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
records = []            # per case
wp_clear, wp_speed_full, wp_speed_trans, wp_case, wp_pos = [], [], [], [], []
n_coll = n_conv_fail = n_ok = 0
n_coll_interp_only = 0
for k, P in enumerate(paths):
    L = P.shape[0]
    cl = np.zeros(L); wall = np.zeros(L, bool); nin = np.zeros(L, int)
    for j in range(L):
        cl[j], wall[j], nin[j] = waypoint_clearance(
            P[j], env_pts, env_is_wall, shape_V, shape_F, shape_radius)
    collided = bool((cl < 0).any())
    interp_collided = False
    if args.interp > 0 and not collided:
        for j in range(L - 1):
            for a in np.linspace(0, 1, args.interp + 2)[1:-1]:
                c = waypoint_clearance((1 - a) * P[j] + a * P[j + 1], env_pts,
                                       env_is_wall, shape_V, shape_F, shape_radius)[0]
                if c < 0:
                    interp_collided = True; break
            if interp_collided:
                break
    converged = bool(succ[k])
    ok = converged and not collided
    if collided: n_coll += 1
    if not converged: n_conv_fail += 1
    if ok: n_ok += 1
    if interp_collided: n_coll_interp_only += 1
    coll_idx = np.nonzero(cl < 0)[0]
    rec = dict(case=int(idx_valid[k]), L=L, n_a=na_all[k], n_b=nb_all[k],
               converged=converged, collided=collided, ok=ok,
               interp_collided=interp_collided,
               n_coll_wp=int(len(coll_idx)),
               max_depth=float(-cl.min()) if collided else 0.0,
               deepest_idx=int(np.argmin(cl)) if collided else -1,
               coll_wall=bool(wall[coll_idx].all()) if collided else False,
               coll_any_wall=bool(wall[coll_idx].any()) if collided else False,
               coll_idx=coll_idx.tolist(),
               min_clear=float(cl.min()),
               n_in_max=int(nin.max()))
    records.append(rec)
    if not args.no_field_stats:
        wp_clear.append(cl); wp_case.append(np.full(L, k)); wp_pos.append(np.arange(L))
    if (k + 1) % 100 == 0:
        print(f'  clearance {k + 1}/{len(paths)}  ({time.time() - t0:.0f}s)')

n_cases = len(paths)
print('\n=== hlB variant: step={} local_w={} samples={} speed_gate={} steps={} ==='.format(
    args.step, args.local_w, args.samples, args.speed_gate, args.steps))
print(f'cases        : {n_cases}')
print(f'success      : {n_ok}  ({100.0 * n_ok / n_cases:.1f}%)')
print(f'collision    : {n_coll}')
print(f'no_converge  : {n_conv_fail}')
if args.interp > 0:
    print(f'collision only between waypoints (passed the waypoint check): {n_coll_interp_only}')

# ─────────────────────────────────────────────────────────────────────────────
# failure anatomy
# ─────────────────────────────────────────────────────────────────────────────
fails = [r for r in records if r['collided']]
if fails:
    depth = np.array([r['max_depth'] for r in fails])
    ncw = np.array([r['n_coll_wp'] for r in fails])
    nin = np.array([r['n_in_max'] for r in fails])
    print('\n--- collision failures: penetration depth (normalized units; margin=0.05, stride=0.015, test offset=0.02)')
    for q in [10, 25, 50, 75, 90]:
        print(f'  depth p{q:02d} = {np.percentile(depth, q):.4f}')
    for th in [0.002, 0.005, 0.01, 0.015, 0.02, 0.05]:
        print(f'  frac max_depth < {th:.3f} : {np.mean(depth < th):.3f}')
    print('--- number of colliding waypoints per failed path')
    for q in [10, 25, 50, 75, 90]:
        print(f'  n_coll_wp p{q:02d} = {np.percentile(ncw, q):.0f}')
    for th in [1, 2, 3, 5, 10]:
        print(f'  frac n_coll_wp <= {th:2d} : {np.mean(ncw <= th):.3f}')
    print(f'--- max env points inside shape at a waypoint: median {np.median(nin):.0f}, p90 {np.percentile(nin, 90):.0f}')
    print(f'--- wall-only collisions: {np.mean([r["coll_wall"] for r in fails]):.3f}   any-wall: {np.mean([r["coll_any_wall"] for r in fails]):.3f}')
    # where along the path
    near_meet = []
    for r in fails:
        meet = r['n_a'] - 1
        near_meet.append(min(abs(i - meet) for i in r['coll_idx']) <= 2)
    print(f'--- collision within 2 waypoints of the meeting point: {np.mean(near_meet):.3f}')
    frac_pos = np.array([r['deepest_idx'] / max(r['L'] - 1, 1) for r in fails])
    print(f'--- deepest waypoint position along path: p25 {np.percentile(frac_pos,25):.2f} p50 {np.percentile(frac_pos,50):.2f} p75 {np.percentile(frac_pos,75):.2f}')
    conv_fail = np.mean([not r['converged'] for r in fails])
    print(f'--- collided AND not converged: {conv_fail:.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# field calibration: predicted speed vs true clearance, over all waypoints
# ─────────────────────────────────────────────────────────────────────────────
if not args.no_field_stats:
    cl_all = np.concatenate(wp_clear)
    case_all = np.concatenate(wp_case)
    pos_all = np.concatenate(wp_pos)
    # build (wp | goal) pairs
    Xq = np.concatenate([np.concatenate([paths[c], np.repeat(arr[idx_valid[c]][None, DIM:], paths[c].shape[0], 0)], 1)
                         for c in range(n_cases)], 0)
    Xq = torch.tensor(Xq, dtype=torch.float32, device=DEVICE)
    sp_full, sp_trans = [], []
    for s0 in range(0, Xq.shape[0], 20000):
        xb = Xq[s0:s0 + 20000]
        tau, w, coords = womodel.network.out(xb)
        g = womodel.function.gradient(tau, coords, create_graph=False)[:, :DIM]
        sp_full.append((1.0 / torch.norm(g, dim=1)).detach().cpu().numpy())
        sp_trans.append((1.0 / torch.norm(g[:, :3], dim=1)).detach().cpu().numpy())
        del tau, w, coords, g
    sp_full = np.concatenate(sp_full); sp_trans = np.concatenate(sp_trans)
    gt = np.clip(cl_all / MARGIN, 0.02, 1.0)
    edges = [-1, -0.02, -0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 1]
    print('\n--- field calibration over ALL waypoints of ALL rollouts (clearance bins, normalized units)')
    print('  bin                 n     gt_speed   pred_full(med)  pred_trans(med)  pred_full(p10-p90)')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cl_all >= lo) & (cl_all < hi)
        if m.sum() == 0:
            continue
        print(f'  [{lo:+.3f},{hi:+.3f})  {m.sum():7d}   {gt[m].mean():.3f}      {np.median(sp_full[m]):.3f}           {np.median(sp_trans[m]):.3f}          {np.percentile(sp_full[m],10):.3f}-{np.percentile(sp_full[m],90):.3f}')
    # within failed paths: speed at colliding vs free waypoints
    fail_ids = set(k for k, r in enumerate(records) if r['collided'])
    mf = np.isin(case_all, list(fail_ids))
    coll_m = mf & (cl_all < 0)
    free_m = mf & (cl_all > 0.02)
    print(f'\n--- failed paths: median pred speed at colliding wps {np.median(sp_full[coll_m]):.3f} (n={coll_m.sum()}) vs free wps (clearance>0.02) {np.median(sp_full[free_m]):.3f} (n={free_m.sum()})')
    # discriminability: of the colliding waypoints, how many have pred speed below the
    # median speed of clearly-free waypoints?
    thr = np.median(sp_full[free_m])
    print(f'    frac colliding wps with pred speed < median-free ({thr:.3f}): {np.mean(sp_full[coll_m] < thr):.3f}')
    print(f'    frac colliding wps with pred speed < 0.5*median-free: {np.mean(sp_full[coll_m] < 0.5 * thr):.3f}')
    # last free waypoint before first collision: did the field warn?
    warn = []
    for k, r in enumerate(records):
        if not r['collided']:
            continue
        j = r['coll_idx'][0]
        if j == 0:
            continue
        m_prev = (case_all == k) & (pos_all == j - 1)
        m_j = (case_all == k) & (pos_all == j)
        warn.append((cl_all[m_prev][0], sp_full[m_prev][0], sp_full[m_j][0]))
    warn = np.array(warn)
    if len(warn):
        print(f'\n--- last free waypoint before first collision (n={len(warn)}):')
        print(f'    true clearance  p25 {np.percentile(warn[:,0],25):.4f}  p50 {np.percentile(warn[:,0],50):.4f}  p75 {np.percentile(warn[:,0],75):.4f}')
        print(f'    pred speed there p25 {np.percentile(warn[:,1],25):.3f}  p50 {np.percentile(warn[:,1],50):.3f}  p75 {np.percentile(warn[:,1],75):.3f}')
        print(f'    pred speed at first colliding wp p50 {np.percentile(warn[:,2],50):.3f}')
    np.savez(os.path.join(args.out, 'waypoints.npz'), clear=cl_all, case=case_all,
             pos=pos_all, sp_full=sp_full, sp_trans=sp_trans)

with open(os.path.join(args.out, 'records.json'), 'w') as f:
    json.dump(dict(args=vars(args), n_cases=n_cases, n_ok=n_ok, n_coll=n_coll,
                   n_conv_fail=n_conv_fail, records=records), f)
np.save(os.path.join(args.out, 'paths.npy'), np.array(paths, dtype=object), allow_pickle=True)
print('saved to', args.out)
