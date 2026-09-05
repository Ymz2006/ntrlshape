"""Is the spread of MPPI-sampled slowness correlated with clearance?

At every collision-free base config x0 we draw the *same* random cloud MPPI draws
(``MPPI_batched`` in evaluate_training_3d_batched.py: ``sample_num`` samples, each
a ``horizon``-long cumulative random walk with per-step magnitude clamped to
``--step``), query the learned field for tau(x0 -> candidate) in one batch, and
divide by the displacement length to get a *slowness* 1/s:

    slowness = tau(x0, x0+d) / ||d||

Dividing by ||d|| is what makes candidates at different radii comparable -- a
candidate two steps out has a proportionally larger tau even in free space, so
only the per-unit-distance cost carries clearance information.

Per base config we then reduce the cloud to spread statistics and correlate them
with that config's true clearance (from ``evaluate_placements``):

    1. range              max - min          vs clearance
    2. p90 - p10                             vs clearance
    3. p70 - p30                             vs mean slowness  (and vs clearance)

Usage (defaults are the rectangle_env1 row of experiments.md):

    python slowness_spread_vs_clearance.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/rectangle.obj \
        --model ./Experiments/3dshape/3dshape_08_06_17_06/latest.pt
"""

import sys
sys.path.append('.')

import os
import io
import time
import argparse
import contextlib

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import (
    load_obj,
    sample_surface_points,
    generate_radius_surface_points,
    tetrahedralize_shape,
    evaluate_placements,
    _sample_configs,
    DEFAULT_ENV_POINTS,
    DEFAULT_TET_SWITCHES,
    TWOD_SHAPE_THICKNESS,
)

DIM = 6
TWO_PI = 2.0 * np.pi
# MPPI's planar sub-space: x, y, rz (see MPPI_batched / preprocess_obj --2d).
PLANAR_FREE_DIMS = (0, 1, 5)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--env', default='datasets/3dshape/env1.obj',
                    help='Environment .obj (same one the model was trained on).')
parser.add_argument('--shape', default='datasets/3dshape/rectangle.obj',
                    help='Robot shape .obj.')
parser.add_argument('--model', default='./Experiments/3dshape/3dshape_08_06_17_06/latest.pt',
                    help='Checkpoint to query (the Model column of experiments.md).')
parser.add_argument('--out', default='./results/slowness_spread',
                    help='Directory for the scatter plots and the stats dump.')
parser.add_argument('--num_configs', type=int, default=2000,
                    help='Number of collision-free base configs to gather.')
parser.add_argument('--samples', type=int, default=50,
                    help='MPPI sample_num: candidates drawn per base config.')
parser.add_argument('--horizon', type=int, default=5,
                    help='MPPI horizon: cumulative steps per sample.')
parser.add_argument('--step', type=float, default=0.015,
                    help='MPPI per-step displacement cap (evaluate --step).')
parser.add_argument('--margin', type=float, default=0.05,
                    help='Clearance band width, only used to report the '
                         'normalized speed alongside the raw clearance.')
parser.add_argument('--offset', type=float, default=0.001,
                    help='Minimum clearance for a base config to count as free.')
parser.add_argument('--num_env_points', type=int, default=DEFAULT_ENV_POINTS)
parser.add_argument('--num_radius_points', type=int, default=1000)
parser.add_argument('--radius_bins', type=int, default=10)
parser.add_argument('--tet_switches', default=DEFAULT_TET_SWITCHES)
parser.add_argument('--shape_scale', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=2000,
                    help='Placement-evaluation batch size (geometry side).')
parser.add_argument('--tau_chunk', type=int, default=200000,
                    help='Rows of [x0|x1] per TravelTimes call.')
parser.add_argument('--2d', dest='two_d', action='store_true',
                    help='Planar mode: matches preprocess_obj --2d and the '
                         'evaluator --2d sub-space (x, y, rz only).')
parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = args.device


# ─────────────────────────── geometry setup ───────────────────────────
# Mirrors preprocess_obj.main() exactly so the configs, the clearances and the
# normalization all live in the same frame the network was trained in.
def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


print('Loading {} / {}'.format(args.env, args.shape))
with _quiet():
    V_env, F_env, names_env = load_obj(args.env)

if args.two_d:
    V_env[:, 2] = 0.0

bb_min, bb_max = V_env.min(axis=0), V_env.max(axis=0)
center_env = 0.5 * (bb_min + bb_max)
scale = float((bb_max - bb_min).max())
V_env_n = (V_env - center_env) / scale

sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
env_points = sample_surface_points(
    V_env_n, F_env[sample_mask], args.num_env_points).astype(np.float32)

ranges = (bb_max - bb_min) / scale
half_extent = ranges * 0.5 - 0.01
if args.two_d:
    half_extent[2] = 0.0

with _quiet():
    V_sh, F_sh, _ = load_obj(args.shape)
shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
V_sh_local = (V_sh - shape_center) / scale * args.shape_scale
if args.two_d:
    z = V_sh_local[:, 2]
    z_ext = float(z.max() - z.min())
    if z_ext > 1e-12:
        V_sh_local[:, 2] = (z - 0.5 * (z.max() + z.min())) * (TWOD_SHAPE_THICKNESS / z_ext)
    else:
        V_sh_local[:, 2] = 0.0

with _quiet():
    TV, TT, TF = tetrahedralize_shape(V_sh_local, F_sh, switches=args.tet_switches)
tet_verts_local = torch.tensor(TV[TT], dtype=torch.float32).to(dev)
face_verts_local = torch.tensor(TV[TF], dtype=torch.float32).to(dev)
env_t = torch.tensor(env_points, dtype=torch.float32).to(dev)
with _quiet():
    rad_points, rad_bins = generate_radius_surface_points(
        V_sh_local, F_sh, args.num_radius_points, args.radius_bins)
rad_points, rad_bins = rad_points.to(dev), rad_bins.to(dev)

hx, hy, hz = (float(half_extent[i]) for i in range(3))


# ────────────────── 1. collision-free base configs + clearance ──────────────────
print('Sampling {} collision-free base configs ...'.format(args.num_configs))
t0 = time.time()
cfg_keep, dist_keep, ang_keep = [], [], []
n_have, n_tried = 0, 0
while n_have < args.num_configs:
    cfg = _sample_configs(args.batch_size, hx, hy, hz, two_d=args.two_d)
    n_tried += cfg.shape[0]
    free, dist, min_angle, _n, _tn, _rn = evaluate_placements(
        cfg, tet_verts_local, face_verts_local, env_t, rad_points, rad_bins,
        dev, two_d=args.two_d)
    keep = free & (dist > args.offset)
    if keep.any():
        cfg_keep.append(cfg[keep])
        dist_keep.append(dist[keep])
        ang_keep.append(min_angle[keep])
        n_have += int(keep.sum())

cfg0 = torch.cat(cfg_keep)[:args.num_configs]            # raw, rotvec in radians
clearance = torch.cat(dist_keep)[:args.num_configs].numpy()
min_angle = torch.cat(ang_keep)[:args.num_configs].numpy()
N = cfg0.shape[0]
print('  kept {}/{} sampled placements in {:.1f}s   clearance [{:.4f}, {:.4f}]'
      .format(N, n_tried, time.time() - t0, clearance.min(), clearance.max()))

# rotvec -> /2pi, the scale the network is trained on.
x0 = cfg0.clone()
x0[:, 3:6] /= TWO_PI
x0 = x0.to(dev)


# ─────────────────────────── 2. load the field ───────────────────────────
print('Loading checkpoint {}'.format(args.model))
womodel = md.Model('./Experiments/3dshape', 'testing_data/3dshape/_probe',
                   DIM, [0.0] * DIM, device=dev)
womodel.load(args.model)
womodel.network.eval()


# ─────────────── 3. the MPPI sample cloud, verbatim from MPPI_batched ───────────────
# dP = step*N(0,1) shared across the horizon  +  step*N(0,1) per horizon step,
# each horizon step's magnitude clamped to `step`, then cumsum. momentum is 0 in
# MPPI_batched (it is overwritten right after the argument is read), so there is
# no prior-step bias to reproduce here.
S, H, step = args.samples, args.horizon, args.step
print('Drawing the MPPI cloud: {} samples x {} horizon = {} candidates per config'
      .format(S, H, S * H))

dP = step * torch.normal(0, 1, size=(N, S, 1, DIM), device=dev) \
    + step * torch.normal(0, 1, size=(N, S, H, DIM), device=dev)
if args.two_d:
    free_mask = torch.zeros(DIM, device=dev)
    free_mask[list(PLANAR_FREE_DIMS)] = 1.0
    dP = dP * free_mask
dP_norm = torch.norm(dP, dim=3, keepdim=True)
dP = dP / (torch.clamp(dP_norm, min=step) / step)
dP_cum = torch.cumsum(dP, dim=2)                          # (N, S, H, DIM)

cand = x0[:, None, None, :] + dP_cum                      # (N, S, H, DIM)
disp = torch.norm(dP_cum, dim=3)                          # (N, S, H)


# ────────────────── 4. batched tau -> slowness = tau / ||d|| ──────────────────
pairs = torch.cat([x0[:, None, None, :].expand(-1, S, H, -1), cand], dim=3)
pairs = pairs.reshape(-1, DIM * 2)
print('Querying TravelTimes on {} pairs ...'.format(pairs.shape[0]))
t0 = time.time()
taus = []
with torch.no_grad():
    for i in range(0, pairs.shape[0], args.tau_chunk):
        taus.append(womodel.function.TravelTimes(pairs[i:i + args.tau_chunk]).float())
tau = torch.cat(taus).reshape(N, S, H)
print('  done in {:.1f}s'.format(time.time() - t0))

slow = (tau / disp.clamp(min=1e-9)).cpu().numpy()         # (N, S, H) slowness 1/s
slow_flat = slow.reshape(N, S * H)


# ─────────────────────────── 5. spread statistics ───────────────────────────
q = np.percentile(slow_flat, [10, 30, 70, 90], axis=1)
p10, p30, p70, p90 = q[0], q[1], q[2], q[3]
stats = {
    'range':   slow_flat.max(axis=1) - slow_flat.min(axis=1),
    'p90_p10': p90 - p10,
    'p70_p30': p70 - p30,
}
mean_slow = slow_flat.mean(axis=1)
med_slow = np.median(slow_flat, axis=1)
# Clearance as the trainer sees it, for reference only.
speed_dist = np.clip(clearance / args.margin, args.offset / args.margin, 1.0)


def corr(a, b):
    """Pearson r and Spearman rho (rank-Pearson), NaN-safe."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float('nan'), float('nan')
    pear = float(np.corrcoef(a, b)[0, 1])

    def rank(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v))
        return r
    spear = float(np.corrcoef(rank(a), rank(b))[0, 1])
    return pear, spear


lines = []


def emit(s=''):
    print(s)
    lines.append(s)


emit('=' * 78)
emit('slowness spread vs clearance   env={}  shape={}'.format(
    os.path.basename(args.env), os.path.basename(args.shape)))
emit('model={}'.format(args.model))
emit('{} base configs x {} candidates   step={}  {}'.format(
    N, S * H, step, 'PLANAR (--2d)' if args.two_d else '6-D'))
emit('=' * 78)
emit()
emit('slowness = tau(x0, x0+d) / ||d||     [{:.3f}, {:.3f}]  mean {:.3f}'.format(
    slow_flat.min(), slow_flat.max(), slow_flat.mean()))
emit('clearance                            [{:.4f}, {:.4f}]  mean {:.4f}'.format(
    clearance.min(), clearance.max(), clearance.mean()))
emit()
emit('--- correlations -------------------------------------------------------')
emit('{:<34} {:>12} {:>12}'.format('', 'Pearson r', 'Spearman rho'))
requested = [
    ('1. range(slowness)   vs clearance', stats['range'], clearance),
    ('2. p90-p10           vs clearance', stats['p90_p10'], clearance),
    ('3. p70-p30           vs mean slowness', stats['p70_p30'], mean_slow),
]
for label, a, b in requested:
    p, s = corr(a, b)
    emit('{:<34} {:>12.4f} {:>12.4f}'.format(label, p, s))
emit()
emit('--- same spreads against the other axis, for completeness --------------')
extra = [
    ('3b. p70-p30          vs clearance', stats['p70_p30'], clearance),
    ('4.  mean slowness    vs clearance', mean_slow, clearance),
    ('5.  median slowness  vs clearance', med_slow, clearance),
    ('6.  range(slowness)  vs mean slowness', stats['range'], mean_slow),
    ('7.  p90-p10          vs mean slowness', stats['p90_p10'], mean_slow),
]
for label, a, b in extra:
    p, s = corr(a, b)
    emit('{:<34} {:>12.4f} {:>12.4f}'.format(label, p, s))
emit()

# Binned means make the shape of the relation legible where the scatter is dense.
emit('--- spread by clearance decile -----------------------------------------')
emit('{:>10} {:>8} {:>10} {:>10} {:>10} {:>10}'.format(
    'clearance', 'n', 'range', 'p90-p10', 'p70-p30', 'mean slow'))
edges = np.percentile(clearance, np.linspace(0, 100, 11))
edges[-1] += 1e-9
for i in range(10):
    m = (clearance >= edges[i]) & (clearance < edges[i + 1])
    if not m.any():
        continue
    emit('{:>10.4f} {:>8d} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}'.format(
        clearance[m].mean(), int(m.sum()), stats['range'][m].mean(),
        stats['p90_p10'][m].mean(), stats['p70_p30'][m].mean(), mean_slow[m].mean()))
emit()

with open(os.path.join(args.out, 'stats.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')


# ─────────────────────────── 6. plots ───────────────────────────
panels = [
    ('range(slowness) vs clearance', clearance, stats['range'], 'clearance', 'max - min'),
    ('p90-p10 vs clearance', clearance, stats['p90_p10'], 'clearance', 'p90 - p10'),
    ('p70-p30 vs mean slowness', mean_slow, stats['p70_p30'], 'mean slowness', 'p70 - p30'),
    ('p70-p30 vs clearance', clearance, stats['p70_p30'], 'clearance', 'p70 - p30'),
]
fig = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels])
for k, (title, xv, yv, xlab, ylab) in enumerate(panels):
    r, c = k // 2 + 1, k % 2 + 1
    pe, sp = corr(yv, xv)
    fig.add_trace(go.Scattergl(
        x=xv, y=yv, mode='markers',
        marker=dict(size=3, opacity=0.35, color=clearance, colorscale='Viridis'),
        name=title, showlegend=False,
        hovertemplate=xlab + '=%{x:.4f}<br>' + ylab + '=%{y:.4f}<extra></extra>'),
        row=r, col=c)
    # Binned mean trend line.
    e = np.percentile(xv, np.linspace(0, 100, 21))
    e[-1] += 1e-9
    bx, by = [], []
    for i in range(20):
        m = (xv >= e[i]) & (xv < e[i + 1])
        if m.any():
            bx.append(xv[m].mean())
            by.append(yv[m].mean())
    fig.add_trace(go.Scatter(x=bx, y=by, mode='lines+markers',
                             line=dict(color='crimson', width=2),
                             showlegend=False), row=r, col=c)
    fig.layout.annotations[k].text = '{}   r={:.3f}  rho={:.3f}'.format(title, pe, sp)
    fig.update_xaxes(title_text=xlab, row=r, col=c)
    fig.update_yaxes(title_text=ylab, row=r, col=c)

fig.update_layout(height=850, width=1250, title_text=(
    'MPPI slowness spread vs clearance -- {} / {} -- {} configs x {} candidates'
    .format(os.path.basename(args.env), os.path.basename(args.shape), N, S * H)))
html = os.path.join(args.out, 'slowness_spread_vs_clearance.html')
fig.write_html(html)
emit('wrote {}'.format(html))
emit('wrote {}'.format(os.path.join(args.out, 'stats.txt')))

np.savez(os.path.join(args.out, 'raw.npz'),
         clearance=clearance, min_angle=min_angle, speed_dist=speed_dist,
         rng=stats['range'], p90_p10=stats['p90_p10'], p70_p30=stats['p70_p30'],
         mean_slow=mean_slow, med_slow=med_slow)
