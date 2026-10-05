"""Sanity-probe the learned 3-D travel-time field.

Loads the same model + data as ``evaluate_training_3d.py`` and checks whether
``TravelTimes`` is a usable cost-to-go field, WITHOUT running MPPI.  This
isolates "is the field good?" from "is the planner good?".

For a handful of (start, goal) pairs it prints:
  * tau(goal, goal)          -- should be ~0 (distance to self)
  * tau(start, goal)         -- should be > 0, on a sane scale
  * tau(interp_t, goal) for t in [0..1]   -- should DECREASE monotonically to ~0
  * whether -d tau / d(x0) points toward the goal (cosine > 0)

Run:
    python probe_field_3d.py --dataPath testing_data/3dshape/rectangle_env1
"""

import sys
sys.path.append('.')

import os
import argparse
from glob import glob

import numpy as np
import torch

from models.metric import model_train_metric as md

DIM = 6

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1')
parser.add_argument('--modelPath', default='./Experiments/3dshape')
parser.add_argument('--n', type=int, default=5, help='number of pairs to probe')
args = parser.parse_args()

womodel = md.Model(args.modelPath, args.dataPath, DIM, [0.0] * DIM, device='cuda')
latest = os.path.join(args.modelPath, 'latest.pt')
pt = latest if os.path.exists(latest) else sorted(
    glob(os.path.join(args.modelPath, '*', 'Model_Epoch_*.pt')))[-1]
print(f'checkpoint: {pt}')
womodel.load(pt)
womodel.network.eval()

arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy'))   # (N, 12)


def tau(x0, x1):
    """x0, x1: (B, 6) numpy -> (B,) travel times."""
    xp = torch.tensor(np.concatenate([x0, x1], axis=1),
                      dtype=torch.float32, device='cuda')
    return womodel.function.TravelTimes(xp).detach().cpu().numpy().reshape(-1)


for i in range(args.n):
    x0 = arr[i, :DIM][None, :].astype(np.float64)     # start (1,6)
    x1 = arr[i, DIM:2 * DIM][None, :].astype(np.float64)  # goal  (1,6)

    t_gg = tau(x1, x1)[0]
    t_sg = tau(x0, x1)[0]

    # cost-to-go along a straight line from start to goal (linear in all 6 dims)
    ts = np.linspace(0.0, 1.0, 11)
    interp = (1 - ts)[:, None] * x0 + ts[:, None] * x1   # (11, 6)
    goal_rep = np.repeat(x1, len(ts), axis=0)            # (11, 6)
    ctg = tau(interp, goal_rep)                          # (11,)

    mono = np.all(np.diff(ctg) <= 1e-6)

    # does -d tau / d x0 point toward the goal?
    # NOTE: network.out detaches its input and returns a fresh grad-enabled
    # `coords` leaf, so differentiate w.r.t. THAT, not our input tensor.
    xp = torch.tensor(np.concatenate([x0, x1], axis=1),
                      dtype=torch.float32, device='cuda')
    t_out, _w, coords_out = womodel.network.out(xp)
    g = torch.autograd.grad(t_out.sum(), coords_out)[0].detach().cpu().numpy()[0, :DIM]
    to_goal = (x1 - x0)[0]
    cos = float(np.dot(-g, to_goal) /
                (np.linalg.norm(g) * np.linalg.norm(to_goal) + 1e-12))

    print(f'\n--- pair {i} ---')
    print(f'  tau(goal,goal) = {t_gg:.4f}   (want ~0)')
    print(f'  tau(start,goal)= {t_sg:.4f}   (want >0)')
    print(f'  cost-to-go t=0..1: ' + ' '.join(f'{v:.3f}' for v in ctg))
    print(f'  monotonically decreasing: {mono}')
    print(f'  cos(-grad, toward_goal) = {cos:+.3f}   (want > 0)')
