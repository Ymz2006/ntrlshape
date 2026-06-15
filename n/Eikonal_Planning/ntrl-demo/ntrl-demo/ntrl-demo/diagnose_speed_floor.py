"""Diagnostic: is the network's predicted speed floored above the target?

Loads a trained checkpoint, runs a batch of *training* points through the field,
and compares the predicted speed ( 1/|grad tau| ) against the target speed the
loss is trained on.  If the predicted speed has a hard lower floor that the
targets dip well below, the Lipschitz / metric-output cap is the bottleneck and
the model physically cannot represent near-zero speed at obstacle boundaries.

Per point (for x0) we report two independent channels of the eikonal residual,
each as predicted-speed vs the target the loss is trained on:

    DIST channel  (translation):
        predicted = 1/sqrt(|grad_{xyz} tau|^2)            grad over dims 0:3
        target    = speed_dist**2 * (2-speed_dist)**2     (train-loop warp)

    ANGLE channel (rotation):
        predicted = 1/sqrt(|grad_{rotvec} tau|^2)         grad over dims 3:6
        target    = warp(clamp(speed_angle/angle_max, .., 1))   (train-loop warp)

For reference we also report the FULL speed  1/sqrt(|grad_{all 6} tau|^2)  --
this is what Function.Speed() / the evaluator uses.

NOTE: Loss() currently does `speed_angle = speed_dist` (line ~82, "TESTING
REMOVE"), so the rotation channel is in fact trained against the *dist* target.
A gap in the ANGLE channel may therefore be partly because the net was never
trained on speed_angles, not only the gradient cap.

Run from the nested ntrl-demo root:
    python diagnose_speed_floor.py --dataPath testing_data/3dshape/<name>
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
HALF = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', required=True,
                    help='Dataset dir containing sampled_points.npy / speed_dists.npy')
    ap.add_argument('--modelPath', default='./Experiments/3dshape')
    ap.add_argument('--ckpt', default=None,
                    help='Explicit checkpoint .pt (default: <modelPath>/latest.pt)')
    ap.add_argument('--batch', type=int, default=20000)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    # ── Load model exactly like evaluate_training_3d.py ──
    womodel = md.Model(args.modelPath, args.dataPath, DIM, [0.0] * DIM, device=args.device)
    pt = args.ckpt
    if pt is None:
        latest = os.path.join(args.modelPath, 'latest.pt')
        if os.path.exists(latest):
            pt = latest
        else:
            ckpts = sorted(glob(os.path.join(args.modelPath, '*', 'Model_Epoch_*.pt')))
            if not ckpts:
                raise FileNotFoundError('no latest.pt and no Model_Epoch_*.pt found')
            pt = ckpts[-1]
    print('Loading checkpoint:', pt)
    womodel.load(pt)
    womodel.network.eval()

    # ── Load a batch of training points + their dist / angle targets ──
    pts = np.load(os.path.join(args.dataPath, 'sampled_points.npy'))     # (N, 12)
    sd = np.load(os.path.join(args.dataPath, 'speed_dists.npy'))         # (N, 2)
    sa = np.load(os.path.join(args.dataPath, 'speed_angles.npy'))        # (N, 2) = angle/pi
    n = min(args.batch, pts.shape[0])
    sel = np.random.choice(pts.shape[0], size=n, replace=False)
    points = torch.tensor(pts[sel], dtype=torch.float32, device=args.device)

    # DIST target: warp applied in the train loop (alpha=1 => alpha-blend is a no-op).
    td_raw = torch.tensor(sd[sel, 0], dtype=torch.float32)               # x0 dist-speed
    td = td_raw**2 * (2 - td_raw)**2

    # ANGLE target: exactly the transform the train loop applies to speed_angle
    #   clamp(speed_angle/angle_max, angle_min/angle_max, 1) then the same warp.
    angle_max = np.pi / 18.0
    angle_min = 0.0001
    ta_raw = torch.tensor(sa[sel, 0], dtype=torch.float32)               # x0 angle (=angle/pi)
    ta_clamped = torch.clamp(ta_raw / angle_max, min=angle_min / angle_max, max=1.0)
    ta = ta_clamped**2 * (2 - ta_clamped)**2

    # ── Forward + gradient (same path as Function.Speed / Loss) ──
    tau, w, Xp = womodel.network.out(points)
    dtau = womodel.function.gradient(tau, Xp)

    def inv_grad_norm(g):
        s = torch.einsum('ij,ij->i', g, g)
        return (1.0 / torch.sqrt(s + 1e-12)).detach().cpu()

    pred_full = inv_grad_norm(dtau[:, :DIM])            # 1/|grad_{all 6} tau|
    pred_dist = inv_grad_norm(dtau[:, :HALF])           # 1/|grad_{xyz} tau|
    pred_ang = inv_grad_norm(dtau[:, HALF:DIM])         # 1/|grad_{rotvec} tau|

    # ── Report ──
    def pct(x):
        q = np.percentile(x.numpy(), [0, 1, 5, 50, 95, 100])
        return '  '.join('{:6.4f}'.format(v) for v in q)

    print('\n                       min     p1      p5     p50     p95     max')
    print('pred speed (full)   : ', pct(pred_full))
    print('pred speed (dist)   : ', pct(pred_dist))
    print('pred speed (angle)  : ', pct(pred_ang))
    print('target dist (warped): ', pct(td))
    print('target angle(warped): ', pct(ta))

    def floor_check(name, pred, target):
        order = torch.argsort(target)
        k = max(1, n // 100)            # lowest 1% of targets (closest to obstacles)
        lo = order[:k]
        print('\n--- {} CHANNEL FLOOR CHECK ---'.format(name))
        print('lowest predicted speed the net outputs : {:.4f}'.format(float(pred.min())))
        print('lowest target asked for                : {:.4f}'.format(float(target.min())))
        print('fraction of targets below the net floor: {:.1%}'.format(
            float((target < pred.min()).float().mean())))
        print('lowest-1% targets: mean target {:.4f}  vs  mean predicted {:.4f}  '
              '({:.2f}x overshoot)'.format(
                  float(target[lo].mean()), float(pred[lo].mean()),
                  float((pred[lo] / (target[lo] + 1e-6)).mean())))

    floor_check('DIST', pred_dist, td)
    floor_check('ANGLE', pred_ang, ta)
    print('\nIf a channel\'s predicted min sits well above its target min and the '
          'overshoot is large, the gradient cap (lip_norm + 0.2 output scale) is '
          'the bottleneck for that channel.')


if __name__ == '__main__':
    main()