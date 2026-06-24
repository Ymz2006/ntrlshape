"""Does the network's gradient 'collapse' (|grad tau| -> 0), and if so does that
REDUCE the loss? Computes, per training sample, the actual quantities the loss
sees and buckets them by |grad| so we can see the relationship directly.

Run: python diagnose_grad_collapse.py --dataPath datasets/3dshape/rectangle_env1
"""
import sys; sys.path.append('.')
import os, argparse
from glob import glob
import numpy as np
import torch
from models.metric import model_train_metric as md

DIM, HALF = 6, 3
LOSS_WEIGHT, MM, THRESH = 1e-2, 10.0, 0.9


def warp_dist(sd):
    sd = sd**2 * (2 - sd)**2
    return np.clip(sd, 0.001, None)

def warp_ang(sa):
    amax, amin = np.pi / 18, 0.0001
    sa = np.clip(sa / amax, amin / amax, 1.0)
    sa = sa**2 * (2 - sa)**2
    return np.clip(sa, 0.001, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--n', type=int, default=0)
    args = p.parse_args()

    model = md.Model(args.modelPath, args.dataPath, DIM, [0.0] * DIM, device='cuda')
    latest = os.path.join(args.modelPath, 'latest.pt')
    pt = latest if os.path.exists(latest) else sorted(
        glob(os.path.join(args.modelPath, '*', 'Model_Epoch_*.pt')))[-1]
    print('checkpoint:', pt)
    model.load(pt); model.network.eval()

    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    sd = warp_dist(np.load(os.path.join(args.dataPath, 'speed_dists.npy')).astype(np.float64))
    sa = warp_ang(np.load(os.path.join(args.dataPath, 'speed_angles.npy')).astype(np.float64))
    if args.n: pairs, sd, sa = pairs[:args.n], sd[:args.n], sa[:args.n]
    N = len(pairs)

    # network forward: tau and dtau
    dtau = np.empty((N, 12)); tau = np.empty(N)
    for s in range(0, N, 4096):
        xp = torch.tensor(pairs[s:s+4096], dtype=torch.float32, device='cuda')
        t, _w, coords = model.network.out(xp)
        g = torch.autograd.grad(t.sum(), coords)[0]
        dtau[s:s+4096] = g.detach().cpu().numpy()
        tau[s:s+4096] = t[:, 0].detach().cpu().numpy()

    # per-group |grad| and the speed target that multiplies it in the loss
    gmag = {
        'ep0_dist': np.linalg.norm(dtau[:, 0:HALF], axis=1),
        'ep0_ang':  np.linalg.norm(dtau[:, HALF:DIM], axis=1),
        'ep1_dist': np.linalg.norm(dtau[:, DIM:DIM+HALF], axis=1),
        'ep1_ang':  np.linalg.norm(dtau[:, DIM+HALF:], axis=1),
    }
    spd = {'ep0_dist': sd[:, 0], 'ep0_ang': sa[:, 0],
           'ep1_dist': sd[:, 1], 'ep1_ang': sa[:, 1]}

    # Eikonal residual, raw LT, and the 1/speed-weighted LT (current code)
    diff_4 = np.zeros(N)
    LT_raw = {}
    for k in gmag:
        res = np.sqrt(gmag[k] + 1e-8) * spd[k] - 1.0
        lt = res**2
        w = np.where(spd[k] < THRESH, np.clip(1.0 / spd[k], None, MM), 1.0)
        LT_raw[k] = lt
        diff_4 += lt * w
    expw = np.exp(-0.5 * tau)
    sample_loss = diff_4 * LOSS_WEIGHT * expw     # the per-sample diff_4 contribution

    full = np.linalg.norm(dtau[:, :DIM], axis=1)  # ep0 full |grad|

    print("\n--- What happens at |grad|=0 (analytic) ---")
    print("  LT = (|grad|*speed - 1)^2 ;  at |grad|=0 -> LT = (0-1)^2 = 1  (NOT 0).")
    print("  So a collapsed gradient gives the MAX low-side residual, not a free pass.\n")

    print("--- The lowest-|grad| ep0 samples (most 'collapsed') ---")
    order = np.argsort(full)[:8]
    print("  %10s %9s %10s %10s %10s %12s" %
          ("|grad|ep0", "tau", "spd_dist0", "LT_dist0", "exp(-T/2)", "loss_contrib"))
    for i in order:
        print("  %10.4g %9.3f %10.4f %10.4f %10.4f %12.4g" %
              (full[i], tau[i], spd['ep0_dist'][i], LT_raw['ep0_dist'][i], expw[i], sample_loss[i]))

    print("\n--- Buckets by ep0 full |grad| (percentile) ---")
    print("  %10s %10s %10s %9s %10s %12s" %
          ("|grad| pct", "mean|grad|", "mean tau", "mean exp", "mean LT", "mean loss"))
    pct = np.percentile(full, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    for a, b, lbl in [(pct[0], pct[1], '0-1%'), (pct[1], pct[2], '1-5%'),
                      (pct[2], pct[3], '5-25%'), (pct[3], pct[4], '25-50%'),
                      (pct[4], pct[5], '50-75%'), (pct[5], pct[6], '75-95%'),
                      (pct[6], pct[7], '95-99%'), (pct[7], pct[8], '99-100%')]:
        m = (full >= a) & (full <= b)
        print("  %10s %10.4g %10.3f %9.4f %10.4f %12.4g" %
              (lbl, full[m].mean(), tau[m].mean(), expw[m].mean(),
               LT_raw['ep0_dist'][m].mean(), sample_loss[m].mean()))


if __name__ == '__main__':
    main()
