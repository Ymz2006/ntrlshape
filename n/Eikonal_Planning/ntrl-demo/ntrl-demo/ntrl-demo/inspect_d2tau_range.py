"""Report the largest second derivative d2tau/dconfig2 the trained network predicts.

Companion to inspect_dtau_range.py.  That script differentiates the predicted
travel time tau ONCE w.r.t. the input config (the eikonal `dtau`).  This one
differentiates TWICE: it builds the Hessian  H = d2 tau / d q_i d q_j  of tau
w.r.t. the QUERY endpoint config q (the first DIM coords), evaluated at a large
batch of valid sampled config pairs, and summarizes its magnitude:

  * per-coordinate diagonal curvature   d2 tau / d q_i^2        (max |.|)
  * Hessian spectral norm  (max |eigenvalue|)  -- the cleanest single
    "magnitude of the second derivative" at a point
  * Hessian Frobenius norm
  * the same, restricted to the translation block (xyz, 3x3) and the
    rotation block (rotvec, 3x3)

The point is to read off how much curvature the field CAN represent right now,
so you can compare before/after widening the Fourier scale B.

Run (from the nested ntrl-demo root, inside the pytorch docker):
    python inspect_d2tau_range.py --dataPath datasets/3dshape/rectangle_env1 --n 100000
    python inspect_d2tau_range.py --dataPath testing_data/3dshape/rectangle_env1
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
HALF = 3   # translation = first 3 query coords (xyz), rotation = last 3 (rotvec)


def load_model(model_path, data_path):
    model = md.Model(model_path, data_path, DIM, [0.0] * DIM, device='cuda')
    latest = os.path.join(model_path, 'latest.pt')
    pt = latest if os.path.exists(latest) else sorted(
        glob(os.path.join(model_path, '*', 'Model_Epoch_*.pt')))[-1]
    print('checkpoint:', pt)
    model.load(pt)
    model.network.eval()
    return model


def compute_hessian(model, pairs, batch=2048, transform='sq'):
    """pairs: (N, 2*DIM) numpy -> H (N, DIM, DIM) numpy.

    H[n] = d2 f(tau) / d q_i d q_j  for the QUERY endpoint q = coords[:, :DIM].
    Built with double autograd: first grad with create_graph=True, then one
    grad per query output-dim to assemble the Hessian rows.

    transform: 'sq'   -> use the model output tau as-is (the trained field, which
                         already has the `x = x**2` square baked in);
               'sqrt' -> use sqrt(tau), i.e. the UN-squared metric d, to isolate
                         how much curvature the `x**2` line is injecting.
    """
    H = np.empty((len(pairs), DIM, DIM), dtype=np.float64)
    for s in range(0, len(pairs), batch):
        xp = torch.tensor(pairs[s:s + batch], dtype=torch.float32, device='cuda')
        # network.out detaches the input and returns a fresh grad-enabled leaf
        # `coords`; differentiate tau w.r.t. THAT (matches the loss / inspect_dtau).
        tau, _w, coords = model.network.out(xp)
        if transform == 'sqrt':
            tau = torch.sqrt(tau.clamp_min(0.0) + 1e-12)
        g = torch.autograd.grad(tau.sum(), coords, create_graph=True)[0]   # (B, 2*DIM)
        rows = []
        for i in range(DIM):                       # only the query block of tau's grad
            hi = torch.autograd.grad(g[:, i].sum(), coords, retain_graph=True)[0]
            rows.append(hi[:, :DIM])               # d/dq of (dtau/dq_i)  -> (B, DIM)
        Hb = torch.stack(rows, dim=1)              # (B, DIM, DIM)
        H[s:s + batch] = Hb.detach().cpu().numpy()
    return H


def _sym_eig_absmax(block):
    """block: (N, k, k) -> (N,) max |eigenvalue| of the symmetrized block."""
    b = 0.5 * (block + np.transpose(block, (0, 2, 1)))
    w = np.linalg.eigvalsh(b)                       # ascending eigenvalues
    return np.abs(w).max(axis=1)


def _report_block(name, block, labels):
    """block: (N, k, k); print spectral / Frobenius / diagonal extremes."""
    spec = _sym_eig_absmax(block)                   # (N,)
    frob = np.linalg.norm(block.reshape(block.shape[0], -1), axis=1)
    print("-" * 90)
    print("%s   (N=%d points)" % (name, block.shape[0]))
    print("  spectral norm  max|eig|   median=%.4g   max=%.4g"
          % (np.median(spec), spec.max()))
    print("  Frobenius norm |H|_F      median=%.4g   max=%.4g"
          % (np.median(frob), frob.max()))
    diag = np.diagonal(block, axis1=1, axis2=2)     # (N, k)
    print("  diagonal d2tau/dq_i^2  (signed min / max / max|.|):")
    for j, lab in enumerate(labels):
        c = diag[:, j]
        print("    %-5s  min=%12.4g  max=%12.4g  max|.|=%12.4g"
              % (lab, c.min(), c.max(), np.abs(c).max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1')
    p.add_argument('--modelPath', default='./Experiments/3dshape')
    p.add_argument('--n', type=int, default=100000,
                   help='number of input pairs to use (0 = all)')
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--transform', choices=['sq', 'sqrt'], default='sq',
                   help="'sq' = trained field tau (has x**2 baked in); "
                        "'sqrt' = sqrt(tau) to remove the square and see the metric d.")
    args = p.parse_args()

    model = load_model(args.modelPath, args.dataPath)

    pairs = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
    if args.n and args.n < len(pairs):
        pairs = pairs[:args.n]
    print('input pairs: %d  (from %s)\n' % (len(pairs), args.dataPath))

    H = compute_hessian(model, pairs, batch=args.batch, transform=args.transform)
    print('transform: %s  (%s)\n' % (args.transform,
          'tau as-is, square baked in' if args.transform == 'sq'
          else 'sqrt(tau): square removed, metric d'))

    labels = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    print("=" * 90)
    print("Second derivative d2tau/dq^2 of the QUERY endpoint  (q = [x y z rx ry rz])")
    print("=" * 90)
    _report_block('FULL config (6x6)', H, labels)
    _report_block('TRANSLATION block (xyz, 3x3)', H[:, :HALF, :HALF], labels[:HALF])
    _report_block('ROTATION block (rotvec, 3x3)', H[:, HALF:, HALF:], labels[HALF:])

    # Headline number: the single largest-magnitude second derivative anywhere.
    spec_full = _sym_eig_absmax(H)
    k = int(np.argmax(spec_full))

    # ── Curvature vs clearance ──────────────────────────────────────────────
    # x0 lives in the band offset < clearance < margin, so speed = clip(clearance
    # / margin, .., 1) is a monotone clearance proxy for the QUERY endpoint:
    # speed ~ offset/margin = tightest (touching), speed -> 1 = most open / about
    # to enter free space.  Ideal field: spectral norm ~0 as speed -> 1, big as
    # speed -> 0.  Bucket the per-point Hessian spectral norm by query clearance.
    sp_path = os.path.join(args.dataPath, 'speed.npy')
    if os.path.exists(sp_path):
        sp0 = np.load(sp_path).astype(np.float64)[:len(spec_full), 0]   # x0 clearance proxy
        edges = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001])
        print("\n" + "=" * 90)
        print("Hessian spectral norm vs QUERY clearance (speed=clip(clr/margin,..,1); "
              "1 = most open)")
        print("=" * 90)
        print("  %-16s %8s %12s %12s %12s" % ("clearance bin", "count", "median", "mean", "max"))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (sp0 >= lo) & (sp0 < hi)
            if not m.any():
                continue
            s = spec_full[m]
            print("  [%.2f, %.2f) %8d %12.4g %12.4g %12.4g"
                  % (lo, hi, m.sum(), np.median(s), s.mean(), s.max()))


    print("\n" + "=" * 90)
    print("MAX magnitude of the second derivative (full 6x6 Hessian spectral norm):")
    print("  max|eig(H)| = %.6g   at sampled pair index %d" % (spec_full[k], k))
    print("  max |single entry| of H over all points = %.6g"
          % np.abs(H).max())
    print("=" * 90)


if __name__ == '__main__':
    main()
