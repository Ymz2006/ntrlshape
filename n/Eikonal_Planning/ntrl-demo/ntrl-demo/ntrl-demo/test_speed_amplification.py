"""
Test: does the `speed < THRESH -> multiply LT by BOOST` change in
models/metric/model_function_metric.py actually fix the
"low-speed losses are less amplified than high-speed ones" problem?

Run from the nested ntrl-demo root (inside the pytorch docker):
    python test_speed_amplification.py

The threshold and boost constants are TUNABLE and are auto-detected from the
running code, so this test keeps working as you tweak them.

Background
----------
Each coordinate-group's Eikonal residual is

    r = |grad tau| * speed - 1          (LT_*_mag before squaring)
    L = boost * r**2 * loss_weight      (per-sample contribution to diff_4)

Two quantities decide whether a region actually gets learned:
  * loss magnitude L, and
  * training signal  dL/d|grad tau| = 2 * boost * speed * r * loss_weight

At a *fixed gradient error* both are proportional to `speed`, so a low-speed
sample produces a far smaller loss/gradient than a high-speed one -- the
optimizer barely corrects it. The boost is meant to compensate.

We drive the REAL `Function.Loss` with a mock network whose output gradient we
set exactly, isolate one coordinate-group (others at residual 0), and read back
the per-sample loss (`diff_4`, the 3rd return value).

  PART A  correctness   -- the boost is a clean step (only {1, boost}), has a
                           single threshold, and each group keys off its OWN
                           speed channel.
  PART B  effectiveness -- loss & training signal vs speed, with/without boost,
                           and the speed at which the boost reaches high-speed
                           parity.
"""

import sys
sys.path.append('.')

import torch

from models.metric.model_function_metric import Function

DIM = 6              # 3-D shape task: SE(3); half_dim = 3 (hardcoded in Loss)
LOSS_WEIGHT = 1e-2   # the loss_weight multiplying diff_4 in Loss()
REF_SPEED = 1.0      # high-speed reference we want low speeds to match

# column of `coeff` carrying each group's gradient magnitude, and which speed
# channel (tensor name, column) the boost for that group should key off.
GROUPS = {
    'd0': dict(col=0, chan=('dist', 0)),   # endpoint0 distance
    'a0': dict(col=3, chan=('ang', 0)),    # endpoint0 angle
    'd1': dict(col=6, chan=('dist', 1)),   # endpoint1 distance
    'a1': dict(col=9, chan=('ang', 1)),    # endpoint1 angle
}


class MockNet:
    """Linear field tau = sum(coeff * coords); grad tau == coeff exactly, so we
    set each group's gradient magnitude by choosing one column of `coeff`."""

    def __init__(self, coeff):
        self.coeff = coeff

    def out(self, points):
        Xp = points.clone().detach().requires_grad_(True)
        tau = (Xp * self.coeff).sum(dim=1, keepdim=True)
        return tau, None, Xp


def run_loss(active, speed, grad, distractor=None):
    """Run the REAL Function.Loss for a sweep of `n` samples.

    `active`  : group key in GROUPS -- the only group with non-zero residual.
    `speed`   : (n,) speed for the active group (set on its own channel).
    `grad`    : (n,) |grad tau| for the active group.
    distractor: optional (other_group_key, (n,) speed) -- lowers a DIFFERENT
                channel's speed to probe mis-wiring; that group stays at its
                optimum residual (loss 0) so diff_4 still reflects `active`.

    Returns per-sample diff_4 (1-D, length n). Other groups sit at residual 0.
    """
    n_real = speed.shape[0]
    if n_real == 1:                     # Loss squeezes -> 0-dim breaks indexing
        speed, grad = speed.repeat(2), grad.repeat(2)
        if distractor is not None:
            distractor = (distractor[0], distractor[1].repeat(2))
    n = speed.shape[0]

    # all groups optimum at speed 1.0  ->  |grad| = 1/1 = 1, residual 0
    coeff = torch.zeros(n, 2 * DIM, dtype=torch.float64)
    for g in GROUPS.values():
        coeff[:, g['col']] = 1.0
    coeff[:, GROUPS[active]['col']] = grad

    speed_dist = torch.ones(n, 2, dtype=torch.float64)
    speed_angle = torch.ones(n, 2, dtype=torch.float64)
    chan = {'dist': speed_dist, 'ang': speed_angle}

    kind, idx = GROUPS[active]['chan']
    chan[kind][:, idx] = speed
    if distractor is not None:
        dgroup, dspeed = distractor
        dkind, didx = GROUPS[dgroup]['chan']
        chan[dkind][:, didx] = dspeed
        # keep the distractor group at its OWN optimum (|grad| = 1/speed) so it
        # contributes residual 0 -- only its speed-channel changes. This way any
        # change in diff_4 must come from mis-wiring, not from a real residual.
        coeff[:, GROUPS[dgroup]['col']] = 1.0 / dspeed

    net = MockNet(coeff)
    fn = Function(path=None, device='cpu', network=net, dim=DIM)

    # coords all zero -> tau = 0 -> exp(-0.5*T) weight is exactly 1, so the
    # returned diff_4 is the bare boost * r**2 * loss_weight (no T weighting).
    points = torch.zeros(n, 2 * DIM, dtype=torch.float64)
    Yobs = torch.ones(n, 2, dtype=torch.float64)          # used only by zeroed tau_loss
    normal = torch.zeros(n, 2 * DIM, dtype=torch.float64)  # normal_weight = 0

    _, _, diff_4 = fn.Loss(points, Yobs, normal, beta=1.0, gamma=0.0, epoch=0,
                           speed_dist=speed_dist, speed_angle=speed_angle)
    return diff_4.detach()[:n_real]


def boost_factor(active, speeds, fixed_r=0.01, **kw):
    """Per-speed multiplier = actual_loss / analytic_unboosted_loss."""
    grad = (1.0 + fixed_r) / speeds                 # makes residual exactly fixed_r
    actual = run_loss(active, speeds, grad, **kw)
    unboosted = (fixed_r ** 2) * LOSS_WEIGHT
    return actual / unboosted


def detect_weighting():
    """Probe the real code to recover the (tunable) weighting: the per-sample
    multiplier as a function of speed, plus the threshold below which it applies
    and the cap. The current design is  w(s) = min(1/s, cap)  for s < threshold."""
    speeds = torch.logspace(-3, 0, 60, dtype=torch.float64)   # 0.001 .. 1.0
    factor = boost_factor('d0', speeds)                       # measured multiplier
    cap = float(factor.max().round().item())
    weighted = factor > 1.0 + 1e-3                            # multiplier clearly > 1
    s_lo_max = float(speeds[weighted].max())                  # last weighted speed
    s_hi_min = float(speeds[~weighted].min())                 # first un-weighted speed
    thresh = 0.5 * (s_lo_max + s_hi_min)                      # midpoint estimate
    return thresh, cap, speeds, factor, s_lo_max, s_hi_min


def part_a_correctness(thresh, cap, speeds, factor, s_lo_max, s_hi_min):
    print("=" * 78)
    print("PART A -- correctness  (auto-detected: threshold~%.3f, cap=%g, weight=1/speed)"
          % (thresh, cap))
    print("=" * 78)

    # (1) multiplier matches  min(1/speed, cap) below threshold, 1 at/above it.
    expected = torch.where(speeds < thresh,
                           torch.clamp(1.0 / speeds, max=cap),
                           torch.ones_like(speeds))
    shape_ok = bool(torch.allclose(factor, expected, rtol=2e-3, atol=2e-3))
    print(" (1) multiplier == min(1/speed, %g) for s<thresh else 1:   %s"
          % (cap, "OK" if shape_ok else "FAIL"))

    # (2) single threshold: all weighted speeds are below all un-weighted ones.
    single = s_lo_max < s_hi_min
    print(" (2) single threshold (weighted speeds < un-weighted):     %s"
          " [last %.4f < first %.4f]" % ("OK" if single else "FAIL", s_lo_max, s_hi_min))

    # (3) wiring: the d0 (distance) group must respond to speed_dist[:,0], NOT to
    #     speed_angle[:,0] or speed_dist[:,1]. Hold d0's own speed HIGH (no weight
    #     expected) while dropping a distractor channel far below threshold.
    lo = torch.tensor([thresh * 0.1], dtype=torch.float64)   # well below threshold
    hi = torch.tensor([REF_SPEED], dtype=torch.float64)      # d0 own speed: high
    exp_self = min(1.0 / float(lo), cap)
    f_self = float(boost_factor('d0', lo)[0])                                  # own ch low  -> weight
    f_ang  = float(boost_factor('d0', hi, distractor=('a0', lo))[0])           # angle ch low -> none
    f_d1   = float(boost_factor('d0', hi, distractor=('d1', lo))[0])           # other ep low -> none
    wired = (abs(f_self - exp_self) < 1e-2 and abs(f_ang - 1.0) < 1e-4 and abs(f_d1 - 1.0) < 1e-4)
    print(" (3) d0 keys off speed_dist[:,0] only:                     %s"
          " [own->x%.2f (exp %.2f), angle->x%.2f, other-ep->x%.2f]"
          % ("OK" if wired else "FAIL", f_self, exp_self, f_ang, f_d1))

    print()
    assert shape_ok and single and wired, "Weighting shape/wiring is NOT correct!"
    print("PART A PASS\n")


def _signal(speeds, r0):
    """Training signal dL/d|grad| at fixed residual r0, measured from the REAL
    Loss by central finite differences (so it reflects whatever boost the code
    applied at each speed)."""
    grad = (1.0 + r0) / speeds                       # |grad| s.t. residual == r0 exactly
    eps = 1e-6
    return (run_loss('d0', speeds, grad + eps) - run_loss('d0', speeds, grad - eps)) / (2 * eps)


def signal_to_ref(speeds, r0):
    """Returns (plain_ratio, boosted_ratio) vs the high-speed reference.

    boosted_ratio : measured from the real Loss (includes the code's boost).
    plain_ratio   : the exact no-boost baseline, signal ~ speed, == speeds/REF.
    Both threshold-independent, so the table never depends on estimating THRESH.
    """
    ref_speed = torch.tensor([REF_SPEED], dtype=torch.float64)
    ref_sig = float(_signal(ref_speed, r0)[0])
    return speeds / REF_SPEED, _signal(speeds, r0) / ref_sig


def part_b_effectiveness(thresh, cap):
    print("=" * 78)
    print("PART B -- effectiveness: training-signal amplification vs speed")
    print("=" * 78)
    print("At EQUAL residual the gradient update is dL/d|grad tau| ~ weight*speed.")
    print("With weight = min(1/speed, cap) the product is 1 wherever 1/speed <= cap,")
    print("so the signal should be FLAT (== high-speed reference) across the band")
    print("(1/cap, threshold), and only taper below the cap knee speed = 1/cap.")
    print("Reference = high-speed sample at speed %.2f, same residual.\n" % REF_SPEED)

    knee = 1.0 / cap
    r0 = 0.1   # fixed residual (relative Eikonal error) for every sample
    speeds = torch.tensor([0.002, 0.005, 0.010, 0.030, 0.100,
                           0.300, 0.600, 0.850, 0.950, 1.000], dtype=torch.float64)
    sig_plain, sig_weight = signal_to_ref(speeds, r0)

    print("(measured from the real Loss by finite differences; residual fixed at %.2f)" % r0)
    print("knee speed = 1/cap = %.4f, threshold = %.3f\n" % (knee, thresh))
    print("%7s %18s %20s" % ("speed", "signal/ref(plain)", "signal/ref(weighted)"))
    for i in range(len(speeds)):
        print("%7.3f %18.4f %20.4f" %
              (float(speeds[i]), float(sig_plain[i]), float(sig_weight[i])))

    # "Works" criterion: the signal is flat (~1) across the active band
    # (knee, threshold) -- that is exactly what 1/speed weighting should buy.
    band = (speeds > knee * 1.5) & (speeds < thresh * 0.95)
    flat = sig_weight[band]
    flat_ok = bool(torch.all((flat > 0.9) & (flat < 1.1)))
    print("\nFlatness across active band (%.3f, %.3f): signal/ref in [%.3f, %.3f]  %s"
          % (knee, thresh, float(flat.min()), float(flat.max()), "OK" if flat_ok else "FAIL"))

    print("\nReading the table:")
    print(" * signal/ref(plain) ~ speed -> the original collapse at low speed.")
    print(" * signal/ref(weighted) ~ 1 across (%.3f, %.3f): low-speed samples now"
          % (knee, thresh))
    print("   compete with high-speed ones at equal residual -- the intended fix.")
    print(" * below the knee %.3f the cap (%g) takes over, so signal ~ cap*speed"
          % (knee, cap))
    print("   still tapers (bounded by design); above %.2f the weight is off, so"
          % thresh)
    print("   signal ~ speed (mild taper to the reference at 1.0).")

    assert flat_ok, "1/speed weighting did NOT flatten the signal across the band!"
    print("\nVERDICT: the 1/speed (cap %g) weighting WORKS -- training signal is flat"
          % cap)
    print("across (%.3f, %.3f); only sub-%.3f speeds stay bounded-but-tapered."
          % (knee, thresh, knee))


def main():
    torch.manual_seed(0)
    thresh, cap, speeds, factor, s_lo_max, s_hi_min = detect_weighting()
    print("\nAuto-detected from model_function_metric.py: threshold~%.4f, cap=%g, weight=1/speed\n"
          % (thresh, cap))
    part_a_correctness(thresh, cap, speeds, factor, s_lo_max, s_hi_min)
    part_b_effectiveness(thresh, cap)
    print("\nDone.")


if __name__ == "__main__":
    main()
