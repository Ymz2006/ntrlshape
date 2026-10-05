"""For points in a clearance bin, step the QUERY translation `step` toward the
obstacle and report the network's predicted speed before vs after.

"Toward the obstacle" = -normal_xyz (the stored normal points the way that
INCREASES clearance).  Rotation and the goal endpoint are held fixed; only the
first 3 coords of x0 move, by L2 magnitude `--step` along the unit translation
normal.

speed reported two ways (both = 1/|grad tau|, as in visual_pose / Function.Speed):
    full   : 1 / |grad_{xyz,rotvec} tau|   (all 6 query dims)
    trans  : 1 / |grad_{xyz} tau|          (translation only -- what we moved)

Run:  python probe_step_toward_obstacle.py --dataPath datasets/3dshape/rectangle_env1 \
          --bin_lo 0.0 --bin_hi 0.1 --step 0.05
"""
import sys; sys.path.append('.')
import os, argparse
from glob import glob
import numpy as np
import torch
from models.metric import model_train_metric as md

DIM, HALF = 6, 3


def load_model(model_path, data_path):
    model = md.Model(model_path, data_path, DIM, [0.0] * DIM, device='cuda')
    latest = os.path.join(model_path, 'latest.pt')
    pt = latest if os.path.exists(latest) else sorted(
        glob(os.path.join(model_path, '*', 'Model_Epoch_*.pt')))[-1]
    print('checkpoint:', pt)
    model.load(pt); model.network.eval()
    return model


def speeds(model, pairs, batch=4096):
    """pairs (N,12) -> (full_speed (N,), trans_speed (N,))  using 1/|grad tau|."""
    fs = np.empty(len(pairs)); ts = np.empty(len(pairs))
    for s in range(0, len(pairs), batch):
        xp = torch.tensor(pairs[s:s+batch], dtype=torch.float32, device='cuda')
        tau, _w, coords = model.network.out(xp)
        g = torch.autograd.grad(tau.sum(), coords)[0][:, :DIM]    # d tau / d query (B,6)
        gn = g.detach().cpu().numpy()
        fs[s:s+batch] = 1.0 / (np.linalg.norm(gn, axis=1) + 1e-12)
        ts[s:s+batch] = 1.0 / (np.linalg.norm(gn[:, :HALF], axis=1) + 1e-12)
    return fs, ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--bin_lo', type=float, default=0.0)
    p.add_argument('--bin_hi', type=float, default=0.1)
    p.add_argument('--step', type=float, default=0.05)
    p.add_argument('--n', type=int, default=0, help='limit points in the bin (0=all)')
    args = p.parse_args()

    model = load_model(args.modelPath, args.dataPath)
    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    speed0 = np.load(os.path.join(args.dataPath, 'speed.npy')).astype(np.float64)[:, 0]
    normal = np.load(os.path.join(args.dataPath, 'normal.npy')).astype(np.float64)

    m = (speed0 >= args.bin_lo) & (speed0 < args.bin_hi)
    idx = np.where(m)[0]
    if args.n and args.n < len(idx):
        idx = idx[:args.n]
    print('bin [%.2f, %.2f): %d points\n' % (args.bin_lo, args.bin_hi, len(idx)))

    before = pairs[idx].copy()
    # Unit translation normal of x0; step toward obstacle = -that direction.
    n_xyz = normal[idx, :HALF]
    n_unit = n_xyz / (np.linalg.norm(n_xyz, axis=1, keepdims=True) + 1e-12)
    after = before.copy()
    after[:, :HALF] = before[:, :HALF] - args.step * n_unit       # move toward obstacle

    fb, tb = speeds(model, before)
    fa, ta = speeds(model, after)

    def line(tag, b, a):
        d = a - b
        print("  %-13s before: median=%.4g mean=%.4g   after: median=%.4g mean=%.4g"
              "   delta: median=%+.4g mean=%+.4g"
              % (tag, np.median(b), b.mean(), np.median(a), a.mean(),
                 np.median(d), d.mean()))

    print("Predicted speed (1/|grad tau|), stepping x0 translation %.3f toward obstacle:"
          % args.step)
    line('full speed', fb, fa)
    line('trans speed', tb, ta)
    print("\n  (target clipped speed in this bin: median=%.4g  -- speed.npy[:,0])"
          % np.median(speed0[idx]))


if __name__ == '__main__':
    main()
