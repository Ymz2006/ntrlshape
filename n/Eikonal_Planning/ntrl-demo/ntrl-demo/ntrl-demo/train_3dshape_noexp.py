"""Run the 3-D shape pipeline with the exp(-0.5*T) loss weighting REMOVED.

This is an ablation entry point.  It does NOT modify any existing file: it
monkey-patches `model_function_metric.Function.Loss` in memory with a copy of
the active loss that drops the `* torch.exp(-0.5*T)` per-sample weighting (every
sample counts equally), then runs the normal training driver.

Everything else is identical to train/train_3dshape.py: same architecture, same
warps (applied in the trainer), same accept/reject loop, same per-block eikonal
residual with the speed<0.9 reweighting and the 5x angle factor.

Run (from the nested ntrl-demo root, inside the pytorch docker):
    python train_3dshape_noexp.py --dataPath datasets/3dshape/Lshape3d_env1 \
        --device cuda:0 --epochs 6000 --no-wandb

The eval script loads ./Experiments/3dshape/latest.pt, which this run overwrites
on the save cadence (and at the final epoch).
"""

import sys
sys.path.append('.')

import time
import argparse

import torch

from models.metric import model_train_metric as md
from models.metric import model_function_metric as mf
from train.wandb_utils import add_wandb_args, apply_overrides, start_run, finish_run

modelPath = './Experiments/3dshape'


def Loss_noexp(self, points, Yobs, normal, beta, gamma, epoch, speed_dist, speed_angle):
    """Active eikonal loss of Function.Loss, but WITHOUT the exp(-0.5*T) weight.

    Mirrors the live computation in model_function_metric.py (the per-block
    eikonal residual with the tight-cell clamp reweighting and the 5x angle
    factor).  The dead terms in the original (n_loss with normal_weight=0,
    tau_loss with td_weight=0, the un-added hess / cross terms) are omitted
    because they do not affect the returned loss.  The ONLY behavioural change
    vs. the original is: no per-sample exp(-0.5*T) weighting.
    """
    tau, w, Xp = self.network.out(points)
    dtau = self.gradient(tau, Xp)

    half_dim = 3
    DT0_dist = dtau[:, :half_dim]
    DT0_ang = dtau[:, half_dim:self.dim]
    DT1_dist = dtau[:, self.dim:self.dim + half_dim]
    DT1_ang = dtau[:, self.dim + half_dim:]

    DT0_dist_mag = torch.einsum('ij,ij->i', DT0_dist, DT0_dist)
    DT0_ang_mag = torch.einsum('ij,ij->i', DT0_ang, DT0_ang)
    DT1_dist_mag = torch.einsum('ij,ij->i', DT1_dist, DT1_dist)
    DT1_ang_mag = torch.einsum('ij,ij->i', DT1_ang, DT1_ang)

    LT0_dist_mag = (torch.sqrt(DT0_dist_mag + 1e-8) * speed_dist[:, 0] - 1) ** 2
    LT0_ang_mag = (torch.sqrt(DT0_ang_mag + 1e-8) * speed_angle[:, 0] - 1) ** 2
    LT1_dist_mag = (torch.sqrt(DT1_dist_mag + 1e-8) * speed_dist[:, 1] - 1) ** 2
    LT1_ang_mag = (torch.sqrt(DT1_ang_mag + 1e-8) * speed_angle[:, 1] - 1) ** 2

    mm = 10
    w0_dist = torch.clamp(1.0 / speed_dist[:, 0], max=mm)
    w0_ang = torch.clamp(1.0 / speed_angle[:, 0], max=mm)
    w1_dist = torch.clamp(1.0 / speed_dist[:, 1], max=mm)
    w1_ang = torch.clamp(1.0 / speed_angle[:, 1], max=mm)

    LT0_dist_mag = torch.where(speed_dist[:, 0] < 0.9, LT0_dist_mag * w0_dist, LT0_dist_mag)
    LT0_ang_mag = 5 * torch.where(speed_angle[:, 0] < 0.9, LT0_ang_mag * w0_ang, LT0_ang_mag)
    LT1_dist_mag = torch.where(speed_dist[:, 1] < 0.9, LT1_dist_mag * w1_dist, LT1_dist_mag)
    LT1_ang_mag = 5 * torch.where(speed_angle[:, 1] < 0.9, LT1_ang_mag * w1_ang, LT1_ang_mag)

    diff_4 = LT0_dist_mag + LT0_ang_mag + LT1_dist_mag + LT1_ang_mag

    loss_weight = 1e-2
    diff_4 = diff_4 * loss_weight
    # >>> the whole point of this variant: NO exp(-0.5*T) per-sample weighting <<<
    loss_n = torch.sum(diff_4) / Yobs.shape[0]

    loss = loss_n
    return loss, loss_n, diff_4


# Install the no-exp loss in place of the original (in memory only).
mf.Function.Loss = Loss_noexp
print('[train_3dshape_noexp] patched Function.Loss -> exp(-0.5*T) weighting REMOVED')


parser = argparse.ArgumentParser(description='Train the 3-D shape planner WITHOUT exp(-0.5*T) loss weighting.')
parser.add_argument('--dataPath', default='./datasets/3dshape/Lshape3d_env1',
                    help='Directory holding the .npy training data.')
parser.add_argument('--device', default='cuda:0',
                    help='Torch device to train on, e.g. cuda:0, cuda:2, cpu.')
add_wandb_args(parser)
args = parser.parse_args()

dataPath = args.data or args.dataPath

# source / goal configuration (x, y, z, rx, ry, rz) -- rotvec stored normalized by 2*pi
model = md.Model(modelPath, dataPath, 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=args.device)

apply_overrides(model, args)
start_run(args, model, task='3dshape')

start = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - start))

finish_run(args)
