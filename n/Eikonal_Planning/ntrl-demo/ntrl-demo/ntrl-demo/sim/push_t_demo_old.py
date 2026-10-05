"""Push the T along an Eikonal-planner trajectory with a single cylindrical pusher.

The planner (``models/metric`` + ``Experiments/3dshape/.../latest.pt``) produces an SE(2)
path for the T itself -- it says nothing about a robot.  This demo closes that gap: it
takes the planned T path as a *reference* and solves, at every frame, the inverse problem
"where must the cylinder be so that the T moves the way the plan wants it to".

Three stages:

1. **Plan** (``plan_tee_path``).  Load the checkpoint, take one start/goal pair out of the
   test set, and run the planar MPPI rollout on the travel-time field.  Output is a
   (T, 3) array of SE(2) poses in the planner's normalized frame.

2. **Lift into the sim frame** (``planner_to_world``).  The planner works in
   ``(p - env_center) / env_scale`` on the *z-up* meshes and poses the shape about its
   bounding-box centre.  ``pymunk_viser_push`` works in raw mesh units on the *y-up*
   meshes mapped through MESH_TO_WORLD, and poses the T about its footprint *centroid*
   (the pymunk body origin).  Those two frames coincide up to the centroid offset --
   MESH_TO_WORLD is exactly the y-up -> z-up rotation -- so the lift is
   ``x_world = x_norm * env_scale + env_center`` plus ``R(theta) @ (centroid - bbox_centre)``.

3. **Push IK** (``PushIK``) + a closed-loop tracker.  Quasi-static planar pushing with an
   ellipsoidal limit surface: a wrench ``(fx, fy, tau)`` applied to the object produces a
   twist proportional to ``(fx, fy, tau / c^2)``.  Inverting that for a *point* contact is
   a search, not a formula -- the contact has to lie on the boundary and the force has to
   stay inside the friction cone -- so ``PushIK.solve`` scores every (boundary point,
   in-cone direction) pair by how well the twist it induces aligns with the twist we want,
   and takes the best.  The pusher is then placed behind that contact along the push
   direction.

   ``c`` is the length scale that trades rotation against translation.  pymunk's top-down
   friction is a max-force / max-torque pair, so the sim's own anisotropy is
   ``spin_friction / friction``; that is the default, not the shape's radius of gyration.

Because a point pusher can only ever realize a 2-D slice of the 3-D wrench space, the
tracker is closed-loop: it re-reads the T's true pose from pymunk every frame and re-solves.
When the required contact moves to a face the pusher cannot reach in a straight line, the
pusher retracts to a standoff circle, arcs around the T (picking the way round that clears
the obstacles) and re-approaches.

Usage (from the ntrl-demo root):
    python sim/push_t_demo.py                        # http://localhost:8080
    python sim/push_t_demo.py --case 7 --port 8081
    python sim/push_t_demo.py --save-path plan.npy   # dump the planned T path and exit
    python sim/push_t_demo.py --traj plan.npy        # replay a saved path (no torch needed)
"""

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time

import numpy as np

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pymunk
import trimesh
import viser
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points

import pymunk_viser_push as base

COL_REF = (120, 200, 120)
COL_ACTUAL = (232, 138, 62)
COL_CONTACT = (220, 60, 60)
COL_GHOST = (150, 210, 150)
COL_STANDOFF = (150, 150, 165)      # the stand-off circle transits ride
COL_EXIT = (240, 200, 60)           # where a transit leaves the shape
COL_ENTRY = (90, 200, 230)          # where it comes back down to the contact
COL_TRANSIT = (240, 200, 60)        # the planned transit path itself


# ======================================================================================
# 1. planning
# ======================================================================================
def planar_mppi(womodel, xp, steps=200, sample_num=50, horizon=5, step=0.015):
    """Single-episode planar MPPI on the travel-time field -> (T, 6) normalized path.

    This mirrors ``evaluate_training_3d_batched.MPPI_batched`` with B = 1 and ``planar``
    always on: same 50-sample cloud, same 10*first + last cost, same softmax(-50*cost)
    weighting, same 0.01 convergence ball.  The displacement is masked onto (x, y, rz) so
    the rollout stays in the sub-space ``preprocess_obj.py --2d`` sampled.
    """
    import torch

    dim = 6
    dev = xp.device
    free = torch.zeros(dim, device=dev)
    free[[0, 1, 5]] = 1.0

    out = [xp[:, 0:dim].clone()]
    for _ in range(steps):
        tmp = xp.clone()[:, None, None, :].repeat(1, sample_num, horizon, 1)
        dP = step * torch.normal(0, 1, size=(1, sample_num, 1, dim), device=dev) \
            + step * torch.normal(0, 1, size=(1, sample_num, horizon, dim), device=dev)
        dP = dP * free
        dP = dP / (torch.clamp(torch.norm(dP, dim=3, keepdim=True), min=step) / step)
        tmp[..., 0:dim] = tmp[..., 0:dim] + torch.cumsum(dP, dim=2)

        ends = tmp[:, :, [0, -1], :]
        cost = womodel.function.TravelTimes(ends.reshape(-1, dim * 2)).reshape(1, sample_num, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        weight = torch.softmax(-50 * cost, dim=1)
        xp[:, 0:dim] = xp[:, 0:dim] + torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)

        out.append(xp[:, 0:dim].clone())
        if torch.norm(xp[:, dim:2 * dim] - xp[:, 0:dim]) < 0.01:
            break
    out.append(xp[:, dim:2 * dim].clone())        # snap the last waypoint onto the goal
    return torch.cat(out, dim=0).detach().cpu().numpy()


def load_planner(data_path, model_path, ckpt, case, device):
    """-> (womodel, start_norm(6), goal_norm(6)).  Kept alive so the demo can replan."""
    import torch
    from models.metric import model_train_metric as md

    womodel = md.Model(model_path, data_path, 6, [0.0] * 6, device=device)
    print(f'[plan] checkpoint {ckpt}')
    womodel.load(ckpt)
    womodel.network.eval()

    arr = np.load(os.path.join(data_path, 'sampled_points.npy'))
    if case >= len(arr):
        raise SystemExit(f'--case {case} out of range (test set has {len(arr)} cases)')
    return womodel, arr[case][:6].copy(), arr[case][6:].copy()


def plan_from(womodel, start_norm, goal_norm, device, steps):
    """MPPI from an arbitrary start -> (T, 3) SE(2) path in NORMALIZED coords.

    Taking the start as an argument rather than reading it out of the test set is what
    makes replanning possible: the demo feeds in wherever the T actually IS.
    """
    import torch

    xp = torch.tensor(np.concatenate([start_norm, goal_norm]), dtype=torch.float32,
                      device=device).reshape(1, 12)
    with torch.no_grad():
        path = planar_mppi(womodel, xp, steps=steps)
    return path[:, [0, 1, 5]], float(np.linalg.norm(path[-2, [0, 1, 5]]
                                                    - np.asarray(goal_norm)[[0, 1, 5]]))


# ======================================================================================
# 2. planner frame -> sim world frame
# ======================================================================================
def planner_to_world(path_norm, env_scale, env_center, bbox_to_centroid):
    """(T,3) normalized planner poses -> (T,3) world SE(2) poses of the pymunk body.

    The planner poses the shape about its bounding-box centre; the pymunk body origin is
    the footprint centroid.  ``bbox_to_centroid`` is that offset in the shape's local
    frame, so it rotates with the body.
    """
    xy = path_norm[:, :2] * env_scale + np.asarray(env_center)[:2]
    th = path_norm[:, 2] * 2.0 * math.pi
    c, s = np.cos(th), np.sin(th)
    d = np.asarray(bbox_to_centroid)
    xy = xy + np.stack([c * d[0] - s * d[1], s * d[0] + c * d[1]], axis=1)
    return np.column_stack([xy, np.unwrap(th)])


def world_to_planner(pose_world, env_scale, env_center, bbox_to_centroid):
    """Inverse of ``planner_to_world`` for one pose -> normalized (x, y, z, rx, ry, rz).

    The replanner needs this: the T's live pose comes out of pymunk in world units about
    the footprint centroid, and the network wants normalized coordinates about the
    bounding-box centre.
    """
    x, y, th = float(pose_world[0]), float(pose_world[1]), float(pose_world[2])
    c, s = math.cos(th), math.sin(th)
    d = np.asarray(bbox_to_centroid)
    xy = np.array([x, y]) - np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    xy = (xy - np.asarray(env_center)[:2]) / env_scale
    return np.array([xy[0], xy[1], 0.0, 0.0, 0.0, wrap(th) / (2.0 * math.pi)])


def resample_path(path, spacing, smooth):
    """Uniform arc-length resample + light smoothing of a (T,3) SE(2) path.

    The raw rollout is stochastic and locally jittery; a pushing controller chasing that
    jitter would thrash the contact.  Arc length is measured in position only, with the
    unwrapped heading carried along.
    """
    seg = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])
    s, path = s[keep], path[keep]
    if s[-1] < 1e-9:
        return path
    n = max(int(s[-1] / spacing) + 1, 2)
    su = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(su, s, path[:, k]) for k in range(3)], axis=1)
    # np.convolve(..., mode='same') returns max(len(signal), len(kernel)), so a kernel
    # wider than the path yields an oversized array.  Replanned paths get very short as
    # the T nears the goal, which is exactly when this bites.
    smooth = min(int(smooth), len(out))
    if smooth > 1 and len(out) > 2:
        # Normalized (edge-shrinking) boxcar: dividing by the same kernel convolved with
        # ones makes the window narrow at the ends instead of being fed invented samples.
        # Constant-padding here would bias a ramp inward -- it dragged the first waypoint
        # a full half-window along the path, and ref[0] / ref[-1] are exactly the poses
        # the planner certified collision-free, so they have to survive untouched.
        k = np.ones(smooth)
        norm = np.convolve(np.ones(len(out)), k, mode='same')
        ends = out[[0, -1]].copy()
        for c in range(3):
            out[:, c] = np.convolve(out[:, c], k, mode='same') / norm
        out[0], out[-1] = ends
    return out


# ======================================================================================
# 3. push inverse kinematics
# ======================================================================================
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PushIK:
    """Inverse quasi-static pushing: desired body twist -> (contact, push direction).

    The T's boundary is sampled once into ``P`` (points, body frame) with outward normals
    ``N``.  A pusher at boundary point ``p`` can only apply a force inside the friction
    cone about the inward normal ``-n``, so the candidate set is
    ``{(p, u) : p in boundary, angle(u, -n) <= atan(mu)}``.

    Under the ellipsoidal limit surface the unit-force candidate ``(p, u)`` induces the
    twist direction ``(u_x, u_y, cross(p, u) / c^2)``.  Scaling the angular row by ``c``
    puts all three components in length units, so alignment with the desired twist is a
    plain cosine -- and the best contact is an argmax over the (N x K) candidate grid.
    """

    def __init__(self, poly, c_length, mu=0.4, n_boundary=360, n_cone=9):
        ring = np.asarray(orient(poly, 1.0).exterior.coords)[:-1]     # CCW
        edge = np.roll(ring, -1, axis=0) - ring
        length = np.linalg.norm(edge, axis=1)
        per = np.maximum((n_boundary * length / length.sum()).astype(int), 2)

        pts, nrm = [], []
        for i in range(len(ring)):
            t = np.linspace(0.0, 1.0, per[i], endpoint=False)[:, None]
            pts.append(ring[i] + t * edge[i])
            e = edge[i] / length[i]
            nrm.append(np.tile([e[1], -e[0]], (per[i], 1)))           # CCW -> outward
        self.P = np.concatenate(pts)                                  # (N, 2)
        self.N = np.concatenate(nrm)                                  # (N, 2)

        half = math.atan(mu)
        ang = np.linspace(-half, half, n_cone)
        ca, sa = np.cos(ang), np.sin(ang)
        inward = -self.N
        # (N, K, 2): the inward normal rotated through the cone.
        self.U = np.stack([inward[:, None, 0] * ca - inward[:, None, 1] * sa,
                           inward[:, None, 0] * sa + inward[:, None, 1] * ca], axis=2)

        self.c = float(c_length)
        moment = self.P[:, None, 0] * self.U[:, :, 1] - self.P[:, None, 1] * self.U[:, :, 0]
        W = np.concatenate([self.U, (moment / self.c)[:, :, None]], axis=2)   # (N, K, 3)
        self.W = W / np.linalg.norm(W, axis=2, keepdims=True)

        self.radius = float(np.linalg.norm(self.P, axis=1).max())
        self.theta = np.arctan2(self.P[:, 1], self.P[:, 0])

    def score(self, twist, prev_idx=None, hysteresis=0.12, hys_width=0.5,
              behind=0.0):
        """twist = (vx, vy, omega) in the BODY frame -> (N, K) alignment scores.

        ``hysteresis`` adds a bonus to candidates near ``prev_idx`` (in boundary-angle
        terms, width ``hys_width`` rad) so the contact does not chatter between two
        near-equally-good faces every frame.

        ``behind`` weights a "get behind the object" prior: to push something one way you
        stand on the far side of it.  A contact at ``p`` is behind the motion when
        ``dot(p, v_hat) < 0``, so the bonus is ``-dot(p, v_hat)/radius``, which is +1 at
        the trailing extreme and -1 at the leading one.  The limit-surface term alone does
        not express this -- a leading-edge contact pulling in the right direction can score
        well on alignment while being a contact you cannot physically make with a pusher
        that can only push.  The prior is faded out by the translation fraction of the
        twist, because for a near-pure rotation "behind" is meaningless.
        """
        d = np.array([twist[0], twist[1], self.c * twist[2]], dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return None
        sc = self.W @ (d / n)                                         # (N, K)
        if behind > 0.0:
            v = d[:2]
            vn = np.linalg.norm(v)
            if vn > 1e-9:
                trans_frac = vn / n           # 1 = pure translation, 0 = pure rotation
                sc = sc + (behind * trans_frac
                           * (-(self.P @ (v / vn)) / self.radius))[:, None]
        if prev_idx is not None and hysteresis > 0.0:
            dth = np.abs(wrap(self.theta - self.theta[prev_idx]))
            sc = sc + hysteresis * np.exp(-(dth / hys_width) ** 2)[:, None]
        return sc

    def solve(self, twist, prev_idx=None, hysteresis=0.12, hys_width=0.5):
        """Best (index, contact, direction, score) ignoring reachability."""
        sc = self.score(twist, prev_idx, hysteresis, hys_width)
        if sc is None:
            return None
        i, j = np.unravel_index(np.argmax(sc), sc.shape)
        return int(i), self.P[i], self.U[i, j], float(sc[i, j])


def _exit_point(start, direction, standoff):
    """March along ``direction`` from ``start`` until outside the stand-off circle."""
    for L in np.linspace(0.0, 4.0 * standoff, 96):
        q = start + direction * L
        if np.linalg.norm(q) >= standoff:
            return q
    return None


def to_world(v, pos, ang, is_vector=False):
    c, s = math.cos(ang), math.sin(ang)
    r = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    return r if is_vector else r + np.asarray(pos)


def to_local(v, pos, ang, is_vector=False):
    r = np.asarray(v, dtype=float) - (0.0 if is_vector else np.asarray(pos))
    c, s = math.cos(-ang), math.sin(-ang)
    return np.array([c * r[0] - s * r[1], s * r[0] + c * r[1]])


def wrap_arr(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def push_length_ladder(push_len, min_len, steps):
    """Geometrically spaced push lengths from ``min_len`` up to ``push_len``.

    Geometric rather than linear because what matters is the *ratio* between consecutive
    rungs: the selector switches rungs when the demanded chunk of path halves, and a
    linear ladder would spend all its resolution at the long end where it is least needed.
    """
    steps = max(int(steps), 1)
    lo = max(min(float(min_len), float(push_len)), 0.5)
    if steps == 1 or lo >= push_len - 1e-9:
        return [float(push_len)]
    return list(np.geomspace(lo, float(push_len), steps))


class PushPrimitives:
    """A discrete library of push actions, plus a closed-form model of what each one does.

    An action is a pair (contact point on the boundary, push direction).  Contacts are
    sampled evenly along the perimeter; directions are the inward normal -- the "centre
    line" at that point -- rotated through a fan of +-``spread``.  Executing one means:
    teleport the pusher behind that contact, then drive it straight for a fixed distance.

    **Forward model.**  Under the quasi-static limit surface a unit force ``u`` at ``p``
    gives the twist ``(u_x, u_y, cross(p,u)/c^2)``.  Because the contact is carried along
    by the object, that twist is constant in the BODY frame, which makes the resulting
    motion an exact screw -- a circular arc in the world, not a straight line.  Integrating
    it properly matters here: over a 40-unit push a primitive with a big moment arm turns
    the object tens of degrees, and a first-order ``dx = s*u`` estimate would put the
    predicted pose in the wrong place entirely.

    Parameterising by translation arc length ``s``:

        kappa  = cross(p, u) / c^2          radians of rotation per unit of translation
        dtheta = s * kappa
        dxy    = SE(2) exponential of that twist, in the body frame

    The body-frame delta depends only on the primitive and ``s``, never on where the object
    currently is, so the whole table is precomputed once and selection is an argmin.
    """

    def __init__(self, ik, n_points=24, n_dirs=5, spread_deg=45.0):
        step = max(len(ik.P) // n_points, 1)
        idx = np.arange(0, len(ik.P), step)[:n_points]
        P, Nrm = ik.P[idx], ik.N[idx]

        ang = np.radians(np.linspace(-spread_deg, spread_deg, n_dirs))
        ca, sa = np.cos(ang), np.sin(ang)
        inward = -Nrm
        U = np.stack([inward[:, None, 0] * ca - inward[:, None, 1] * sa,
                      inward[:, None, 0] * sa + inward[:, None, 1] * ca], axis=2)

        self.P = np.repeat(P[:, None, :], n_dirs, axis=1).reshape(-1, 2)
        self.U = U.reshape(-1, 2)
        self.N = np.repeat(Nrm[:, None, :], n_dirs, axis=1).reshape(-1, 2)
        self.contact_idx = np.repeat(idx[:, None], n_dirs, axis=1).reshape(-1)
        self.c = float(ik.c)
        self.n_points, self.n_dirs = len(idx), n_dirs
        # radians of rotation per unit of translation
        self.kappa = (self.P[:, 0] * self.U[:, 1]
                      - self.P[:, 1] * self.U[:, 0]) / (self.c ** 2)

        # Measured forward model, filled in by calibrate().  One (M, 3) table per push
        # length: the same primitive is a different action depending on how far it drives.
        self.lengths = None          # (nL,) ascending push lengths, world units
        self.tables = None           # (nL, M, 3) body-frame pose delta of each action

    def __len__(self):
        return len(self.P)

    def predict(self, s):
        """(M, 3) body-frame pose delta after a push of translation arc length ``s``."""
        th = s * self.kappa
        tiny = np.abs(self.kappa) < 1e-12
        k = np.where(tiny, 1.0, self.kappa)
        A = np.where(tiny, s, np.sin(th) / k)              # -> s as kappa -> 0
        B = np.where(tiny, 0.0, (1.0 - np.cos(th)) / k)    # -> 0 as kappa -> 0
        return np.column_stack([A * self.U[:, 0] - B * self.U[:, 1],
                                B * self.U[:, 0] + A * self.U[:, 1],
                                th])

    def calibrate(self, args, tee_poly, push_lens, cache_dir='.', gap=2.0):
        """Measure each primitive by executing it, and use that as the forward model.

        The analytic ``predict`` is not good enough to plan with.  Measured against the
        simulator it over-predicts rotation by roughly 9x, and the measured rotation is
        near-constant in the push length (~5 deg at L = 10, 20 and 40 alike) rather than
        growing with it.  The reason is that the screw model assumes the body keeps turning
        under a constant body twist, whereas a rigid single-point push turns the object only
        until the contact normal lines up with the push direction and thereafter simply
        slides it.  Rank correlation stays high (+0.85), so the analytic model knows which
        primitives rotate more, but its magnitudes cannot place the object.

        So each primitive is rolled out once in an empty arena and the resulting body-frame
        delta is recorded.  Exact by construction, and it costs one short rollout per
        primitive, cached to disk against the parameters that affect it.

        A whole ladder of push lengths is measured, not just one.  Selection then picks the
        length as well as the contact, which is what lets the action shrink as the T closes
        on the goal: far away the demanded chunk of path is ~40 units and a long push wins;
        near the goal the demand is a couple of units and every long push overshoots it, so
        the short end of the ladder wins on cost with no distance schedule to tune.
        """
        lens = sorted({round(float(L), 3) for L in np.atleast_1d(push_lens)})
        self.lengths = np.asarray(lens, dtype=float)
        self.tables = np.stack([
            self._calibrate_one(args, tee_poly, L, cache_dir, gap) for L in lens])
        return self.tables

    def _calibrate_one(self, args, tee_poly, push_len, cache_dir, gap):
        """Measure every primitive at one push length (cached to disk)."""
        key = (os.path.basename(str(args.shape)), round(push_len, 3), self.n_points,
               self.n_dirs, round(float(self.U[0] @ self.U[0]), 6), len(self),
               round(args.pusher_radius, 3), round(args.friction, 3),
               round(args.spin_friction, 3), round(args.tee_mass, 4),
               round(args.pusher_speed, 3), int(args.fps), int(args.substeps),
               # The contact surfaces change what the rollout measures, so they belong in
               # the key -- otherwise editing SURF_PUSHER silently reuses a stale table.
               base.SURF_PUSHER, base.SURF_TEE)
        tag = hashlib.md5(repr(key).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f'.push_primitives_{tag}.npy')
        if os.path.exists(path):
            out = np.load(path)
            print(f'[primitives] loaded measured model for {len(self)} actions '
                  f'(L={push_len:.1f}) from {os.path.basename(path)}')
            return out

        print(f'[primitives] measuring {len(self)} actions at L={push_len:.1f} '
              f'(one rollout each; cached afterwards)...')
        arena = [Polygon([(-6000, -6000), (6000, -6000), (6000, 6000), (-6000, 6000)])]
        t0 = time.time()
        out = np.zeros((len(self), 3))
        for i in range(len(self)):
            sim = PushSim(args, arena, tee_poly, (0.0, 0.0), 0.0,
                          (-3000.0, -3000.0), park=(-3000.0, -3000.0))
            p_l, u_l = self.P[i], self.U[i]
            start = p_l - u_l * (args.pusher_radius + gap)
            sim.pusher.position = (float(start[0]), float(start[1]))
            sim.target = pymunk.Vec2d(*(start + u_l * (gap + push_len + args.pusher_radius)))
            sim.max_speed = args.pusher_speed
            dt = 1.0 / args.fps
            for _ in range(int(6.0 * args.fps)):
                for _ in range(args.substeps):
                    sim.step(dt / args.substeps)
                if (pymunk.Vec2d(*sim.target) - sim.pusher.position).length < 0.5:
                    break
            out[i] = (sim.tee.position.x, sim.tee.position.y, wrap(sim.tee.angle))
        np.save(path, out)
        print(f'[primitives] done in {time.time() - t0:.0f}s; '
              f'|dxy| median {np.median(np.linalg.norm(out[:, :2], axis=1)):.1f}, '
              f'|dtheta| median {math.degrees(np.median(np.abs(out[:, 2]))):.1f} deg '
              f'-> {os.path.basename(path)}')
        return out

    def select(self, delta_body, s=None, ang_weight=None, mask=None, len_bias=0.0):
        """Best (primitive, push length) for a desired body-frame pose delta.

        Returns ``(index, length, predicted delta)``.  Rotation is put in length units with
        the same ``c`` the IK uses, so position and heading error trade off consistently
        with everything else in this file.  Uses the MEASURED tables when calibrated, and
        the analytic screw model only as a fallback -- see ``calibrate`` for why the
        analytic one cannot be planned with.

        Searching over the length ladder as well as the contact is what tapers the push:
        the cost is the residual pose error the action would leave behind, so an action
        that overshoots a small demand is penalised exactly as much as one that undershoots
        a large one.  ``len_bias`` adds a small preference for the long end so that far
        from the goal the controller does not dawdle through many tiny actions when a few
        big ones would do.
        """
        if self.tables is not None:
            preds, lens = self.tables, self.lengths
        else:
            lens = np.asarray([s if s is not None else 1.0], dtype=float)
            preds = self.predict(float(lens[0]))[None]
        w = self.c if ang_weight is None else ang_weight
        d = np.asarray(delta_body, dtype=float)
        cost = (np.linalg.norm(preds[:, :, :2] - d[None, None, :2], axis=2)
                + w * np.abs(wrap_arr(preds[:, :, 2] - d[2])))
        if len_bias:
            cost = cost + len_bias * (lens.max() - lens)[:, None]
        if mask is not None:
            cost = np.where(np.asarray(mask)[None, :], cost, np.inf)
        li, i = np.unravel_index(int(np.argmin(cost)), cost.shape)
        if not np.isfinite(cost[li, i]):
            return None, None, None
        return int(i), float(lens[li]), preds[li, i]


# ======================================================================================
# 4. closed-loop tracker
# ======================================================================================
class PushController:
    """Tracks the reference T path by repositioning a single cylindrical pusher.

    Two states.  In PUSH the controller solves the push IK for the current pose error and
    drives the pusher into the chosen contact.  In REPOSITION it walks the pusher around
    the T to reach a contact it cannot get to in a straight line.

    Three things keep that from thrashing, and all three were needed to make it work:

    * **Reachable-first ranking.**  The IK's global best contact is often on the far side
      of the T.  Committing to it every frame means a walk-around every frame, so the
      controller instead takes the best contact it can reach *without* crossing the
      object, and only pays for a reposition when that costs more than ``switch_margin``
      of alignment.
    * **Retract along the push direction, not radially.**  The T is concave: a point in
      the notch between the crossbar and the stem has the object *outward* of it in the
      radial direction, so retracting radially drags the T instead of releasing it.
      Backing out along ``-push_dir`` retraces the path the pusher approached on, which
      is free by construction.
    * **A stall detector.**  A contact can be geometrically valid and still make no
      progress (the T wedged against a wall, or the push line through the centroid).  If
      the reference index has not advanced and the T has not moved for ``stall_seconds``,
      the current contact is banned for a while and the controller re-solves.

    Reposition waypoints are stored in the T's LOCAL frame, so if the T drifts while the
    pusher is walking around it the chain follows the object instead of aiming at a stale
    spot in the world.
    """

    def __init__(self, ik, ref, args, obstacles):
        self.ik = ik
        self.ref = ref                                  # (T, 3) world SE(2), live
        self.ref0 = ref                                 # the path we started from
        self.a = args
        self.obstacles = obstacles
        # Region the pusher CENTRE may not cross.  It has to be smaller than the stand-off
        # the pusher keeps while pressing (pusher_radius - press), otherwise the pusher is
        # inside its own no-go zone whenever it is in contact and every reachability test
        # trivially fails.
        self.no_go = Polygon(ik.P).buffer(max(args.pusher_radius - args.press - 0.5, 0.5))
        # A SECOND, larger keep-out used only for transit planning.  ``no_go`` above is
        # deliberately smaller than the pusher so that a pressing pusher is not inside its
        # own forbidden zone -- but that makes it far too permissive for a transit, which
        # must not touch the T at all: a straight line clearing ``no_go`` by a hair still
        # buries the disc up to (press + 0.5) deep.  A transit is checked against the real
        # swept region instead, the T grown by the whole pusher radius.
        self.transit_free = args.pusher_radius + 0.5 * args.clearance
        self.no_sweep = Polygon(ik.P).buffer(self.transit_free)
        self.standoff = ik.radius + args.pusher_radius + args.standoff
        self.tee_shape = Polygon(ik.P)                  # body frame, for clearance queries
        self.reset()

    def reset(self):
        self.ref = self.ref0                            # drop any replanned path
        self.k = 0                                      # reference index (monotonic)
        self.state = 'REPOSITION'
        self.queue = []                                 # local-frame waypoints
        self.prev_idx = None
        self.prev_dir = None                            # last push direction, local frame
        self.contact_world = None
        self.done = False
        self.banned = {}                                # contact index -> frames left
        self._ban = np.zeros(len(self.ik.P), dtype=bool)
        self.n_forced = 0                               # stall events, watched by replan
        self._blocked_frames = 0
        self._leg_frames = 0
        self.face = None            # committed contact face, as a boundary angle
        self.challenger = None      # face currently trying to take over
        self.challenge = 0          # consecutive frames it has been winning
        self.n_switch = 0           # committed face changes, the thing to keep small
        self.last_len = None        # push length of the most recent primitive
        self.entry_local = None     # --teleport-interp: body-frame entry / end of push
        self.final_local = None
        self.transit_marks = None   # (exit, entry) on the stand-off circle, for the GUI
        self.stall = 0
        self.last_k = -1
        self.last_pos = None
        self.speed_scale = 1.0          # pusher speed multiplier, set every step()

    def adopt(self, ref):
        """Replace the reference with a freshly planned one and re-anchor on it.

        The new path starts from where the T was when the replan was requested, which is
        slightly behind where it is now, so the carrot is re-seeded by *proximity* rather
        than reset to 0 -- otherwise the controller spends the first moments after every
        replan driving backwards to a waypoint it has already passed.
        """
        self.ref = ref
        self.k = int(np.argmin(np.linalg.norm(ref[:, :2] - self.last_pos, axis=1))) \
            if self.last_pos is not None else 0
        self.queue = []
        self.state = 'PUSH'
        self.stall = 0
        self.last_k = -1
        self.challenger, self.challenge = None, 0

    # -- reference tracking ----------------------------------------------------------
    def _advance(self, pos):
        """Carrot: monotonic nearest-waypoint projection -> the lookahead target index.

        This used to advance only while the T was within a fixed radius of the current
        waypoint, which silently breaks as soon as the object can move further than that
        radius in one control decision.  A discrete push action displaces the T by ~37
        units against a 9-unit threshold, so the carrot never advanced at all and the whole
        run tracked waypoint 1.  Projecting onto the nearest waypoint instead has no such
        scale assumption, and staying monotonic (searching only forward, within a window)
        keeps it from snapping backwards where the path crosses itself.
        """
        n = len(self.ref)
        hi = min(n, self.k + self.a.advance_window + 1)
        j = int(np.argmin(np.linalg.norm(self.ref[self.k:hi, :2] - pos, axis=1)))
        self.k = min(self.k + j, n - 1)
        return min(self.k + self.a.lookahead, n - 1)

    def _repel(self, pos, ang):
        """World-frame velocity that pushes the T off anything it is about to touch.

        The reference path is collision-free *for the T*, but the T does not track it
        perfectly, and nothing in the pose-error term knows that the straight line back to
        the path may run through a block.  Without this the controller cheerfully wedges
        the T into an obstacle and then keeps pressing, which is exactly how the rollout
        used to die.  Distances are taken from the T's real footprint, not its centroid --
        the T is 83 units across, so a centroid-based margin would have to be so large it
        would repel everywhere in a cluttered environment.
        """
        poly = translate(rotate(self.tee_shape, ang, origin=(0, 0), use_radians=True),
                         pos[0], pos[1])
        v = np.zeros(2)
        for o in self.obstacles:
            d = poly.distance(o)
            if d >= self.a.repel_dist:
                continue
            p_o, p_t = nearest_points(o, poly)
            away = np.array([p_t.x - p_o.x, p_t.y - p_o.y])
            n = np.linalg.norm(away)
            if n < 1e-9:                       # already overlapping -- push the centroid out
                away = pos - np.array([p_o.x, p_o.y])
                n = np.linalg.norm(away)
            if n < 1e-9:
                continue
            v += (away / n) * self.a.k_repel * (self.a.repel_dist - d) / self.a.repel_dist
        return v

    def _twist(self, pos, ang, tgt, vel, omega):
        """Body-frame twist that would drive the T from (pos, ang) toward ``tgt``.

        PD, not P.  A point push has a lot of rotational authority -- the stem tip is at
        41.7 units against c = 26.7, so a push there spins the T about as readily as it
        slides it -- and a pure proportional law overshoots the heading and then flips the
        contact to the opposite face to correct, over and over.  Damping against the T's
        measured twist settles that out.
        """
        e = (self.a.k_pos * (self.ref[tgt, :2] - pos)
             - self.a.k_damp * np.asarray(vel)
             + self._repel(pos, ang))
        eth = self.a.k_ang * wrap(self.ref[tgt, 2] - ang) - self.a.k_damp_ang * omega
        v = to_local(e, None, ang, is_vector=True)
        return np.array([v[0], v[1], eth])

    # -- contact commitment ------------------------------------------------------------
    def _face_mask(self, angle):
        """Boundary samples belonging to the face centred on ``angle``."""
        return np.abs(wrap(self.ik.theta - angle)) < self.a.face_width

    def _commit(self, sc_raw):
        """Temporal mode filter over the contact choice.  Returns the committed face.

        The per-frame argmax is noisy: a face that is briefly best flips the contact for a
        frame or two and then flips back, and every one of those flips costs a walk-around.
        The pattern looks like ``x x x x y x x x y x x`` -- the ``y`` samples are noise,
        not a decision.

        So the controller commits to a FACE (an angular neighbourhood of the boundary,
        ``--face-width``) rather than to a sample, re-solves freely inside it every frame
        -- which keeps the pusher's actual motion continuous as the T moves -- and hands
        the face over only when a single challenger has beaten the incumbent by
        ``--switch-margin`` for ``--switch-hold`` seconds without interruption.  A one-off
        good frame can never win, so isolated ``y`` samples are filtered out by
        construction.
        """
        best_i = int(np.argmax(sc_raw.max(axis=1)))
        if self.face is None:
            self.face = float(self.ik.theta[best_i])
            self.n_switch += 1
            return self.face

        held = self._face_mask(self.face)
        if not held.any():
            self.face = float(self.ik.theta[best_i])
            self.n_switch += 1
            return self.face

        row = sc_raw.max(axis=1)
        inc_score = float(row[held].max())
        outside = ~held
        if not outside.any():
            return self.face
        cand_i = int(np.flatnonzero(outside)[np.argmax(row[outside])])
        cand_a = float(self.ik.theta[cand_i])

        if row[cand_i] - inc_score <= self.a.switch_margin:
            self.challenger, self.challenge = None, 0        # incumbent still fine
            return self.face

        # A challenger only counts if it is the SAME challenger as last frame.
        if (self.challenger is not None
                and abs(wrap(cand_a - self.challenger)) < self.a.face_width):
            self.challenge += 1
        else:
            self.challenger, self.challenge = cand_a, 1

        if self.challenge >= self.a.switch_hold * self.a.fps:
            self.face = cand_a
            self.challenger, self.challenge = None, 0
            self.n_switch += 1
        return self.face

    # -- candidate selection ---------------------------------------------------------
    def _free(self, local_pt, pos, ang):
        """Is the pusher centre at this LOCAL point clear of the environment?"""
        w = Point(to_world(local_pt, pos, ang))
        # Exact point-to-polygon distance, rather than buffering the point into a 65-gon
        # and intersecting: same answer, less work, and this runs up to 60x per frame.
        return all(o.distance(w) > self.a.pusher_radius for o in self.obstacles)

    def _reachable(self, sc, pusher_local, pos, ang):
        """Best candidate the pusher can actually take -> (i, j, score, appr).

        Walks the score grid in descending order and returns the first candidate that is
        both a straight shot (the pusher's centre does not cross ``no_go``) and standing
        in free space.  The environment check is not optional: without it the controller
        happily picks a contact whose stand-off point is inside a wall, drives the pusher
        there anyway, and pinches the T between an infinite-mass kinematic body and static
        geometry -- which pymunk resolves by ejecting the T across the map.
        """
        order = np.argsort(sc, axis=None)[::-1]
        seen = set()
        for f in order:
            i, j = np.unravel_index(f, sc.shape)
            i = int(i)
            if i in seen or self._ban[i]:
                continue
            seen.add(i)
            appr = self.ik.P[i] - self.ik.U[i, j] * (self.a.pusher_radius + self.a.clearance)
            seg = LineString([pusher_local, appr])
            if ((seg.length < self.a.wp_tol or not seg.intersects(self.no_go))
                    and self._free(appr, pos, ang)):
                return i, int(j), float(sc[i, j]), appr
            if len(seen) >= self.a.probe:
                break
        return None

    def _pick_global(self, sc, pos, ang):
        """Contacts worth WALKING to, best first -> list of (i, j, approach point).

        A list rather than a single pick, because ``_plan_reposition`` can refuse a target
        whose walk-around is blocked in both directions; the caller then tries the next.

        A reposition ends with a radial move inward from the stand-off circle, and the T
        is concave: at an angle where the crossbar sits outside a notch, that radial leg
        passes straight through the object.  So a walk-around target has to be radially
        approachable as well as clear of the environment -- checking only the environment
        leaves the pusher shouldering the T aside on the way in.
        """
        seen, out = set(), []
        for f in np.argsort(sc, axis=None)[::-1]:
            i, j = np.unravel_index(f, sc.shape)
            i, j = int(i), int(j)
            if i in seen or self._ban[i] or not np.isfinite(sc[i, j]):
                continue
            seen.add(i)
            if len(seen) > self.a.probe:
                break
            appr = self.ik.P[i] - self.ik.U[i, j] * (self.a.pusher_radius + self.a.clearance)
            r = np.linalg.norm(appr)
            if r < 1e-9:
                continue
            entry = self.standoff * appr / r
            if LineString([entry, appr]).intersects(self.no_go):
                continue
            if not self._free(appr, pos, ang):
                continue
            out.append((i, j, appr))
            if len(out) >= self.a.walk_candidates:
                break
        return out

    # -- repositioning ---------------------------------------------------------------
    def _blocked(self, pts_local, pos, ang):
        """How many of these local-frame waypoints put the pusher inside an obstacle."""
        n = 0
        for p in pts_local:
            w = to_world(p, pos, ang)
            disc = Point(w).buffer(self.a.pusher_radius + self.a.clearance)
            n += any(disc.intersects(o) for o in self.obstacles)
        return n

    def _retract(self, pusher_local):
        """A point on the stand-off circle the pusher can reach WITHOUT crossing the T.

        This used to march along ``-prev_dir`` on the reasoning that it retraced the
        pusher's approach.  It does not: the pusher approaches radially (leg 3 of the
        previous reposition), while ``prev_dir`` is the push direction -- the contact's
        inward normal, tilted inside the friction cone.  On a concave T those differ, and
        from a contact in the notch the outward normal runs straight through the crossbar.
        An audit put 99.8% of all unwanted pusher/T contact on this one leg, worst-case
        clearance -5.0 with a pusher radius of 5.0, i.e. the disc fully buried.

        So: test the segment, do not assume it.  Directions are tried in order of angular
        distance from ``-prev_dir``, so when the natural back-off is clear -- the common
        case -- nothing changes, and the search only deviates when it has to.
        """
        r = np.linalg.norm(pusher_local)
        if r >= self.standoff:
            # Outside the stand-off circle already: it encloses the whole T, so radial is
            # clear by construction.
            return self.standoff * pusher_local / (r + 1e-9)

        prefer = (-self.prev_dir if self.prev_dir is not None
                  else pusher_local / (r + 1e-9))
        a_pref = math.atan2(prefer[1], prefer[0])
        offsets = np.linspace(0.0, math.pi, self.a.retract_dirs)
        order = [0.0] + [s * o for o in offsets[1:] for s in (+1.0, -1.0)]

        fallback = None
        for off in order:
            a = a_pref + off
            d = np.array([math.cos(a), math.sin(a)])
            q = _exit_point(pusher_local, d, self.standoff)
            if q is None:
                continue
            if fallback is None:
                fallback = q
            if not LineString([pusher_local, q]).intersects(self.no_go):
                return q
        # Nothing clean in any direction (the pusher is boxed in). Take the natural
        # back-off and let the leg timeout / stall path recover.
        return fallback if fallback is not None else \
            self.standoff * pusher_local / (r + 1e-9)

    def _arc(self, a0, a1, direction):
        """Waypoints along the stand-off circle from angle ``a0`` to ``a1``.

        ``direction`` forces the sense of travel (+1 CCW, -1 CW) so the caller can test
        both ways round and veto the blocked one, rather than always taking the short way
        and driving through a wall.
        """
        sweep = wrap(a1 - a0)
        if direction > 0 and sweep < 0:
            sweep += 2 * math.pi
        if direction < 0 and sweep > 0:
            sweep -= 2 * math.pi
        n = max(int(abs(sweep) / math.radians(self.a.arc_step)) + 1, 2)
        return [self.standoff * np.array([math.cos(a0 + sweep * t),
                                          math.sin(a0 + sweep * t)])
                for t in np.linspace(0.0, 1.0, n)]

    # -- transit planning (--teleport-interp) ------------------------------------------
    def _normal_exit(self, q_local):
        """March ``q_local`` straight out to the stand-off circle along the surface normal.

        "Away from the shape" is taken literally: the nearest point of the T's boundary is
        found, and the ray is the one that leaves that point perpendicular to the surface.
        On a convex patch that is the same as going radially; in one of the T's notches it
        is not, and the radial ray would cut through the crossbar -- which is exactly the
        failure ``_retract`` was written to avoid.
        """
        q = np.asarray(q_local, dtype=float)
        r = np.linalg.norm(q)
        if r >= self.standoff:
            return q                       # already outside; the circle encloses the T
        near = np.asarray(nearest_points(self.tee_shape.exterior, Point(q))[0].coords[0])
        d = q - near
        n = np.linalg.norm(d)
        if n < 1e-6:
            # Sitting exactly on the boundary: use the sampled outward normal there.
            d = self.ik.N[int(np.argmin(np.linalg.norm(self.ik.P - q, axis=1)))]
        else:
            d = d / n
        out = _exit_point(q, d, self.standoff)
        return out if out is not None else self.standoff * q / (r + 1e-9)

    def _release(self, q_local):
        """Shortest step straight off the surface until the disc is clear of the T.

        A transit starts with the pusher pressed against the object, i.e. inside the swept
        keep-out, so a straight-line test from there always fails and every transit would
        arc.  Backing off perpendicular to the surface first is both the shortest way out
        and the only direction guaranteed not to drag the T, so it costs a fraction of a
        second and makes the straight-line case available again.
        """
        q = np.asarray(q_local, dtype=float)
        near = np.asarray(nearest_points(self.tee_shape.exterior, Point(q))[0].coords[0])
        d = q - near
        n = np.linalg.norm(d)
        if n < 1e-6 or self.tee_shape.contains(Point(q)):
            d = self.ik.N[int(np.argmin(np.linalg.norm(self.ik.P - q, axis=1)))]
        else:
            if n >= self.transit_free:
                return None                      # already clear; no release leg needed
            d = d / n
        return near + d * self.transit_free

    def plan_transit(self, pusher_local, entry_local, pos, ang):
        """Body-frame waypoints from where the pusher is to a primitive's entry point.

        This is what ``--teleport-interp`` flies.  The pusher is never teleported: it
        releases straight off the surface, then either goes straight in when the straight
        line is clear, or out-around-in -- leave along the surface normal until outside the
        stand-off circle, ride the circle round, come back down the entry point's own
        normal.  Every waypoint is in the T's BODY frame, so if the T shifts mid-transit
        the path deforms with it and stays collision-free instead of going stale.

        Returns ``(waypoints, marks)`` where ``marks`` is the (exit, entry) pair on the
        stand-off circle, or None for a straight transit.  Never fails: if the environment
        blocks both ways round the circle the less-blocked one is taken anyway, because
        stopping is worse than scraping and teleporting is not an option in this mode.
        """
        entry = np.asarray(entry_local, dtype=float)
        lead = []
        rel = self._release(pusher_local)
        if rel is not None:
            lead, pusher_local = [rel], rel

        seg = LineString([pusher_local, entry])
        if seg.length < self.a.wp_tol or not seg.intersects(self.no_sweep):
            self.transit_marks = None
            return lead + [entry], None

        out0 = self._normal_exit(pusher_local)
        out1 = self._normal_exit(entry)
        a0 = math.atan2(out0[1], out0[0])
        a1 = math.atan2(out1[1], out1[0])
        ways = [self._arc(a0, a1, +1), self._arc(a0, a1, -1)]
        blocked = [self._blocked(w, pos, ang) for w in ways]
        clear = [w for w, b in zip(ways, blocked) if b == 0]
        pick = min(clear, key=len) if clear else ways[int(np.argmin(blocked))]
        self.transit_marks = (out0, out1)
        return lead + [out0] + pick + [out1, entry], (out0, out1)

    def _plan_reposition(self, approach_local, pusher_local, pos, ang):
        """Retract -> arc around the stand-off circle -> approach.  Local-frame waypoints.

        Returns False if neither way round the circle is clear of the environment, so the
        caller can fall back to a different contact.
        """
        back = self._retract(pusher_local)
        a0 = math.atan2(back[1], back[0])
        a1 = math.atan2(approach_local[1], approach_local[0])

        ccw, cw = self._arc(a0, a1, +1), self._arc(a0, a1, -1)
        # Veto, do not rank.  Picking the "less blocked" way round still drives the pusher
        # through a wall when both are blocked; refusing lets the caller try the next
        # contact instead, which is nearly always reachable some other way.
        clear = [c for c in (ccw, cw) if self._blocked(c, pos, ang) == 0]
        if not clear:
            return False
        pick = min(clear, key=len)
        self.queue = [back] + pick + [approach_local]
        self.state = 'REPOSITION'
        self._leg_frames = 0
        return True

    # -- stall detection ---------------------------------------------------------------
    def _check_stall(self, pos):
        moved = self.last_pos is None or np.linalg.norm(pos - self.last_pos) > self.a.stall_eps
        if self.k != self.last_k or moved:
            self.stall = 0
        else:
            self.stall += 1
        self.last_k, self.last_pos = self.k, pos.copy()
        for i in list(self.banned):
            self.banned[i] -= 1
            if self.banned[i] <= 0:
                del self.banned[i]
        # Banning a single sample is useless -- the next-best contact is its neighbour
        # 1/360th of the perimeter away, which pushes identically.  Ban the whole face.
        self._ban = np.zeros(len(self.ik.P), dtype=bool)
        for i in self.banned:
            self._ban |= np.abs(wrap(self.ik.theta - self.ik.theta[i])) < self.a.ban_width
        if self.stall > self.a.stall_seconds * self.a.fps and self.prev_idx is not None:
            self.banned[self.prev_idx] = int(self.a.ban_seconds * self.a.fps)
            self.prev_idx = None
            self.stall = 0
            self.n_forced += 1
            return True
        return False

    def direct_wrench(self, tee_pos, tee_ang, tee_vel=(0.0, 0.0), tee_omega=0.0):
        """Virtual-pusher tracking -> ((force, point) in BODY coords, contact_world, status).

        The push IK runs exactly as it does for the real pusher -- same twist, same
        friction cone, same "get behind it" prior -- and the chosen force is applied at the
        chosen boundary point, so the torque is generated by the real moment arm rather
        than being commanded at the centre of mass.  What is skipped is only whether a
        cylinder could physically be standing there: no reachability test, no walk-around,
        no face commitment (which exists to suppress walk-arounds and has nothing to
        suppress here).

        So this isolates the END EFFECTOR, not the contact model.
        """
        pos = np.asarray(tee_pos, dtype=float)
        tgt = self._advance(pos)
        if (self.k >= len(self.ref) - 1
                and np.linalg.norm(self.ref[-1, :2] - pos) < self.a.goal_dist
                and abs(wrap(self.ref[-1, 2] - tee_ang)) < math.radians(self.a.goal_deg)):
            self.done = True
            return None, None, 'GOAL'

        tw = self._twist(pos, tee_ang, tgt, tee_vel, tee_omega)
        sc = self.ik.score(tw, prev_idx=self.prev_idx, hysteresis=self.a.hysteresis,
                           behind=self.a.behind_weight)
        if sc is None:
            return None, None, 'HOLD'
        i, j = np.unravel_index(np.argmax(sc), sc.shape)
        i, j = int(i), int(j)
        contact_local, push_dir = self.ik.P[i], self.ik.U[i, j]

        err = (np.linalg.norm(self.ref[-1, :2] - pos)
               + self.ik.c * abs(wrap(self.ref[-1, 2] - tee_ang)))
        scale = float(np.clip(err / self.a.slow_radius, self.a.min_speed_frac, 1.0))
        force = push_dir * (self.a.direct_gain * self.a.friction * scale)

        self.prev_idx = i
        self.contact_world = to_world(contact_local, pos, tee_ang)
        return ((force, contact_local), self.contact_world,
                f'DIRECT ref {self.k}/{len(self.ref) - 1}')

    # -- discrete push primitives (teleport mode) --------------------------------------
    def begin_primitive(self, tee_pos, tee_ang, prims, push_len):
        """Choose the next push action and return where to teleport the pusher.

        Runs only when the previous action has finished, which is the whole point of the
        scheme: one inference per action instead of one per frame.  The target is the
        reference waypoint ``--action-steps`` ahead, so the action is asked to cover a
        known chunk of path rather than an instantaneous velocity.

        The push *length* is chosen here too, not fixed: ``prims.select`` ranks every
        (contact, direction, length) triple by the pose error it would leave behind.  Far
        from the goal the demand is a full ``--action-steps`` chunk and the long end of the
        ladder wins; as the T closes in, the demand shrinks below one long push and the
        short lengths win instead, which is what makes the last few actions converge rather
        than oscillate across the goal.

        Returns ``(teleport_pos, final_pos, contact_world, status)`` in world coordinates,
        or ``(None, None, None, status)`` when there is nothing sensible to do.
        """
        pos = np.asarray(tee_pos, dtype=float)
        self._advance(pos)
        tgt = min(self.k + self.a.action_steps, len(self.ref) - 1)
        if (self.k >= len(self.ref) - 1
                and np.linalg.norm(self.ref[-1, :2] - pos) < self.a.goal_dist
                and abs(wrap(self.ref[-1, 2] - tee_ang)) < math.radians(self.a.goal_deg)):
            self.done = True
            return None, None, None, 'GOAL'

        # Desired pose change over the next chunk, expressed in the body frame -- the frame
        # the primitive table lives in, so the comparison needs no per-frame geometry.
        d_world = self.ref[tgt, :2] - pos
        d_body = to_local(d_world, None, tee_ang, is_vector=True)
        d_th = wrap(self.ref[tgt, 2] - tee_ang)

        # An action is usable only if the pusher can actually stand at its start point.
        starts = prims.P - prims.U * (self.a.pusher_radius + self.a.clearance)
        ok = np.array([self._free(sp, pos, tee_ang) for sp in starts])
        if not ok.any():
            return None, None, None, 'BLOCKED'

        i, L, pred = prims.select((d_body[0], d_body[1], d_th), push_len, mask=ok,
                                  len_bias=self.a.len_bias)
        if i is None:
            return None, None, None, 'BLOCKED'

        self.prev_idx = int(prims.contact_idx[i])
        self.n_switch += 1
        contact_w = to_world(prims.P[i], pos, tee_ang)
        u_w = to_world(prims.U[i], None, tee_ang, is_vector=True)
        teleport = contact_w - u_w * (self.a.pusher_radius + self.a.clearance)
        final = teleport + u_w * (self.a.clearance + L + self.a.pusher_radius)
        self.contact_world = contact_w
        self.last_len = L
        # Body-frame twins of the two world points, for --teleport-interp: the transit is
        # planned and flown in the T's frame so it survives the T moving underneath it.
        self.entry_local = starts[i]
        self.final_local = prims.P[i] + prims.U[i] * L
        return (teleport, final, contact_w,
                f'ACTION {i} L={L:.0f} ref {self.k}->{tgt}/{len(self.ref) - 1}')

    # -- main step -------------------------------------------------------------------
    def step(self, tee_pos, tee_ang, pusher_pos, tee_vel=(0.0, 0.0), tee_omega=0.0):
        """Returns (world target for the pusher, contact point or None, status string)."""
        pos = np.asarray(tee_pos, dtype=float)
        tgt = self._advance(pos)
        if (self.k >= len(self.ref) - 1
                and np.linalg.norm(self.ref[-1, :2] - pos) < self.a.goal_dist
                and abs(wrap(self.ref[-1, 2] - tee_ang)) < math.radians(self.a.goal_deg)):
            self.done = True
            self.contact_world = None
            self.speed_scale = 0.0
            return np.asarray(pusher_pos), None, 'GOAL'

        forced = self._check_stall(pos)
        pusher_local = to_local(pusher_pos, pos, tee_ang)

        if self.state == 'REPOSITION' and self.queue and not forced:
            if np.linalg.norm(pusher_local - self.queue[0]) < self.a.wp_tol:
                self.queue.pop(0)
                self._leg_frames = 0
            else:
                self._leg_frames += 1
            if self._leg_frames > self.a.leg_timeout * self.a.fps:
                # A leg that will not finish means the pusher is stuck against something
                # the waypoint chain did not anticipate.  Abandon and re-solve rather than
                # keep shoving; count it as a stall so the replanner is asked too.
                self.queue = []
                self._leg_frames = 0
                self.n_forced += 1
            if self.queue:
                self.speed_scale = self.a.reposition_speedup
                return (to_world(self.queue[0], pos, tee_ang), None,
                        f'REPOSITION [{len(self.queue)} wp]')
        self.state = 'PUSH'
        self.queue = []

        sc = self.ik.score(self._twist(pos, tee_ang, tgt, tee_vel, tee_omega),
                           prev_idx=self.prev_idx, hysteresis=self.a.hysteresis,
                           behind=self.a.behind_weight)
        if sc is None:
            self.speed_scale = 0.0
            return np.asarray(pusher_pos), None, 'HOLD'
        if self.banned:
            sc = sc.copy()
            sc[self._ban] = -np.inf

        # Which FACE to push on is a committed decision with hysteresis in time; which
        # sample and cone direction within that face is re-solved freely every frame.  The
        # face vote uses the raw scores so the Gaussian stickiness bonus cannot inflate the
        # incumbent's side of the comparison.
        sc_raw = self.ik.score(self._twist(pos, tee_ang, tgt, tee_vel, tee_omega),
                               prev_idx=None, hysteresis=0.0,
                               behind=self.a.behind_weight)
        if sc_raw is None:
            sc_raw = sc
        if self.banned:
            sc_raw = np.where(self._ban[:, None], -np.inf, sc_raw)
        face = self._commit(sc_raw)
        sc_face = np.where(self._face_mask(face)[:, None], sc, -np.inf)

        # Straight-line push on the committed face if possible; otherwise walk around to
        # it.  Repositioning is now downstream of the commitment, so the pusher can only
        # be sent around the T when the face decision itself actually changed.
        reach = self._reachable(sc_face, pusher_local, pos, tee_ang)
        glob = self._pick_global(sc_face, pos, tee_ang)

        if reach is None:
            for gi, gj, appr in glob:
                if self._plan_reposition(appr, pusher_local, pos, tee_ang):
                    self.prev_idx = gi
                    self.speed_scale = self.a.reposition_speedup
                    return (to_world(self.queue[0], pos, tee_ang), None,
                            f'REPOSITION [{len(self.queue)} wp]')
            # No walk-around survived: nothing is radially approachable, clear of the
            # environment, AND reachable around the circle.
            # Freezing here is an absorbing state -- the pusher stops, so the T stops,
            # so the geometry never changes and it stays blocked forever.  Back the
            # pusher out to the stand-off circle (which also un-blocks contacts it was
            # itself occluding), drop the bans, and count it as a stall so the
            # replanner is asked for a path from wherever the T has ended up.
            self.banned.clear()
            self._ban[:] = False
            # Rate-limited: BLOCKED persists for many frames, and signalling a stall
            # on every one of them asks for a replan every frame.
            self._blocked_frames += 1
            if self._blocked_frames % max(int(self.a.fps), 1) == 1:
                self.n_forced += 1
            self.speed_scale = self.a.reposition_speedup
            out = self._retract(pusher_local)
            return (to_world(out, pos, tee_ang), None, 'BLOCKED')

        # Terminal taper.  Pushing at full speed right up to the goal overshoots it: the
        # rollout would come within 10 units and then be shoved 170 past.  Scale the
        # pusher's speed with the remaining error so the last approach is a nudge.
        err = (np.linalg.norm(self.ref[-1, :2] - pos)
               + self.ik.c * abs(wrap(self.ref[-1, 2] - tee_ang)))
        self.speed_scale = float(np.clip(err / self.a.slow_radius,
                                         self.a.min_speed_frac, 1.0))

        i, j, score, _ = reach
        contact_local, push_dir = self.ik.P[i], self.ik.U[i, j]
        # Aim slightly PAST the contact so the velocity controller keeps pressing instead
        # of hovering a hair off the surface.
        target_local = contact_local - push_dir * (self.a.pusher_radius - self.a.press)
        self.prev_idx, self.prev_dir = i, push_dir
        self.contact_world = to_world(contact_local, pos, tee_ang)
        return (to_world(target_local, pos, tee_ang), self.contact_world,
                f'PUSH ref {self.k}/{len(self.ref) - 1} align {score:+.2f}')


class Replanner:
    """Re-solves the T's reference path from wherever the T actually is, off-thread.

    A rollout costs 0.4-1.2 s on the GPU, which is 30-70 frames at 60 Hz -- far too long
    to block the physics loop.  So a request runs on a worker thread while the controller
    keeps tracking the path it already has, and the new path is swapped in only once it is
    ready.  That is ordinary MPC latency: the returned path starts from where the T was
    when the request went out, not where it is when the path lands.  The T is slow (and is
    usually stationary during a walk-around, which is when replans tend to fire), so the
    staleness is small, but it is the reason the controller re-anchors on adopt rather
    than trusting the first waypoint blindly.

    One request in flight at a time; ``poll`` hands back a finished path exactly once.
    """

    def __init__(self, womodel, goal_norm, device, steps, env_scale, env_center,
                 bbox_to_centroid, spacing, smooth):
        self.womodel = womodel
        self.goal_norm = np.asarray(goal_norm)
        self.device, self.steps = device, steps
        self.env_scale, self.env_center = env_scale, env_center
        self.b2c, self.spacing, self.smooth = bbox_to_centroid, spacing, smooth
        self._lock = threading.Lock()
        self._result = None
        self.busy = False
        self.n_done = 0
        self.last_ms = 0.0

    def solve_now(self, pose_world):
        """Replan synchronously.  For the headless harness only.

        Headless runs far faster than real time, so a worker thread would land its answer
        tens of SIMULATED seconds after it was asked for -- a cadence nothing like the live
        demo's.  Solving inline makes the headless run a faithful (just slower) model of
        what the viewer does, and makes its numbers reproducible.
        """
        self._run(np.asarray(pose_world, dtype=float))
        return self.poll()

    def request(self, pose_world):
        """Kick off a replan from this world pose.  Non-blocking; False if already busy."""
        if self.busy:
            return False
        self.busy = True
        threading.Thread(target=self._run, args=(np.asarray(pose_world, dtype=float),),
                         daemon=True).start()
        return True

    def _run(self, pose_world):
        t0 = time.time()
        try:
            start = world_to_planner(pose_world, self.env_scale, self.env_center, self.b2c)
            path, _ = plan_from(self.womodel, start, self.goal_norm, self.device, self.steps)
            ref = planner_to_world(path, self.env_scale, self.env_center, self.b2c)
            ref = resample_path(ref, self.spacing, self.smooth)
            if len(ref) >= 2:
                with self._lock:
                    self._result = ref
        except Exception as exc:                    # a failed replan must not kill the sim
            print(f'[replan] failed: {exc}')
        finally:
            self.last_ms = 1000.0 * (time.time() - t0)
            self.busy = False

    def poll(self):
        with self._lock:
            r, self._result = self._result, None
        if r is not None:
            self.n_done += 1
        return r


class PushSim(base.Sim):
    """base.Sim, plus the planned start POSE and an ideal-actuator experiment mode.

    ``direct`` mode replaces the cylinder with a VIRTUAL pusher: the push IK still picks a
    contact point on the boundary and a direction inside the friction cone, and that force
    is applied to the T *at that point*, so the torque comes from the real moment arm.
    What is removed is only the end effector's realization -- its body, its collisions, its
    reachability and its walk-arounds.  The contact mechanics are untouched.

    That makes it the control experiment for the pusher itself: identical reference,
    tracker, contact model and physics, differing only in whether the contact has to be
    reached by a real object.  Error that survives is the planner's or the tracker's; error
    that disappears was the cost of having to get the cylinder there.  Obstacles still
    collide with the T normally.
    """

    def __init__(self, args, env_polys, tee_poly, tee_start, tee_angle, pusher_start,
                 park=(0.0, 0.0)):
        super().__init__(args, env_polys, tee_poly, tee_start, pusher_start)
        self.tee_angle0 = float(tee_angle)
        self.tee.angle = self.tee_angle0
        self.direct = False
        # (force, point), both in BODY-local coordinates -- pymunk's
        # apply_force_at_local_point takes both in the body frame, which is exactly the
        # frame PushIK works in, so no rotation is needed anywhere.
        self.contact = None
        self.park = tuple(park)
        self._stowed = None
        self.teleport_mode = False
        self.action_final = None       # world position the pusher is driving to

    def set_direct(self, on):
        on = bool(on)
        if on == self.direct:
            return
        self.direct = on
        if on:
            # Stow the pusher outside the walls so it cannot collide with anything.
            self._stowed = tuple(self.pusher.position)
            self.pusher.position = self.park
        elif self._stowed is not None:
            self.pusher.position = self._stowed
        self.pusher.velocity = (0.0, 0.0)
        self.target = pymunk.Vec2d(*self.pusher.position)
        self.contact = None

    def teleport_pusher(self, position):
        """Place the pusher instantly.  Kinematic bodies carry no momentum, so this is
        safe -- but it is also why teleport mode cannot sweep the T on the way in, which
        is exactly the walk-around cost the mode exists to remove."""
        self.pusher.position = (float(position[0]), float(position[1]))
        self.pusher.velocity = (0.0, 0.0)
        self.target = pymunk.Vec2d(float(position[0]), float(position[1]))

    def step(self, dt):
        if not self.direct:
            super().step(dt)
            return
        # pymunk clears accumulated force/torque after every step, so re-apply each one.
        self.tee.force = (0.0, 0.0)
        self.tee.torque = 0.0
        if self.contact is not None:
            f, p = self.contact
            self.tee.apply_force_at_local_point((float(f[0]), float(f[1])),
                                                (float(p[0]), float(p[1])))
        self.pusher.velocity = (0.0, 0.0)
        self.space.step(dt)

    def reset(self):
        super().reset()
        self.tee.angle = self.tee_angle0
        self.contact = None
        if self.direct:
            self.pusher.position = self.park


# ======================================================================================
# 5. app
# ======================================================================================
def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # geometry / planner
    ap.add_argument('--env', default='datasets/3dshape/2denv4.obj')
    ap.add_argument('--shape', default='datasets/3dshape/Tshape3d.obj')
    ap.add_argument('--shape-zup', default='datasets/3dshape/Tshape3d_zup.obj',
                    help='z-up twin of --shape; only its bounding box is read, to recover '
                         'the origin the planner poses the shape about.')
    ap.add_argument('--dataPath', default='./testing_data/3dshape/Tshape3d_env4')
    ap.add_argument('--modelPath', default='./Experiments/3dshape')
    ap.add_argument('--ckpt', default='./Experiments/3dshape/3dshape_08_19_12_31/latest.pt')
    ap.add_argument('--case', type=int, default=0, help='test-set index to plan for')
    ap.add_argument('--mppi-steps', type=int, default=200)
    ap.add_argument('--plan-device', default='cuda')
    ap.add_argument('--traj', default=None,
                    help='replay a saved (T,3) SE(2) path instead of planning (world '
                         'units; skips the torch import entirely)')
    ap.add_argument('--save-path', default=None,
                    help='write the planned world-frame (T,3) path here and exit')
    ap.add_argument('--spacing', type=float, default=4.0,
                    help='arc-length resampling of the reference path, world units')
    ap.add_argument('--smooth', type=int, default=9,
                    help='boxcar width used to smooth the reference path (0 = off)')
    # physics (passed through to base.Sim)
    ap.add_argument('--pusher-radius', type=float, default=5.0)
    ap.add_argument('--pusher-height', type=float, default=60.0)
    ap.add_argument('--pusher-speed', type=float, default=20.0)
    ap.add_argument('--tee-mass', type=float, default=1.0)
    ap.add_argument('--friction', type=float, default=300.0)
    ap.add_argument('--spin-friction', type=float, default=8000.0)
    # base.Sim reads this; the push controller's clearance model treats every obstacle as
    # a fixed polygon (_repel, _free, the walk-around arc), and run_viser does not draw
    # movable blocks, so this demo is static-obstacles only.
    ap.set_defaults(dynamic_obstacles=False)
    ap.add_argument('--fps', type=float, default=60.0)
    ap.add_argument('--substeps', type=int, default=4)
    ap.add_argument('--port', type=int, default=8080)
    # push IK
    ap.add_argument('--mu', type=float, default=None,
                    help='pusher/T friction coefficient, i.e. the half-angle atan(mu) of '
                         'the cone the push direction has to stay inside. Default: read '
                         'off the simulator, SURF_PUSHER * SURF_TEE from '
                         'pymunk_viser_push.py, which is the coefficient Chipmunk actually '
                         'applies at that contact -- so the model follows the sim.')
    ap.add_argument('--c-length', type=float, default=None,
                    help='limit-surface length scale. Default spin_friction/friction, '
                         'which is the anisotropy pymunk actually simulates.')
    ap.add_argument('--n-boundary', type=int, default=360)
    ap.add_argument('--n-cone', type=int, default=9)
    ap.add_argument('--behind-weight', type=float, default=0.4,
                    help='weight on the "stand behind the object" prior: contacts on the '
                         'trailing side of the desired motion are favoured. Faded out for '
                         'near-pure rotations, where there is no trailing side. 0 = off.')
    ap.add_argument('--hysteresis', type=float, default=0.12,
                    help='bonus for keeping the previous contact (anti-chatter)')
    # controller
    ap.add_argument('--k-pos', type=float, default=1.0)
    ap.add_argument('--k-ang', type=float, default=2.0,
                    help='heading gain. Only the RATIO to --k-pos matters (the IK uses '
                         'the twist direction, not its size), and the angular row is '
                         'scaled by c inside the score -- so k_ang trades off against '
                         'k_pos * |position error| / c, not against k_pos directly.')
    ap.add_argument('--k-damp', type=float, default=0.35,
                    help='damping on the T\'s linear velocity (the D of the PD law)')
    ap.add_argument('--k-damp-ang', type=float, default=3.0,
                    help='damping on the T\'s angular velocity')
    ap.add_argument('--reposition-speedup', type=float, default=2.5,
                    help='pusher speed multiplier while walking around the T -- that leg '
                         'is free space, so there is no reason to crawl through it')
    ap.add_argument('--slow-radius', type=float, default=45.0,
                    help='pose error (units, heading folded in via c) below which the '
                         'pusher speed tapers down linearly, so the final approach '
                         'settles instead of overshooting')
    ap.add_argument('--min-speed-frac', type=float, default=0.12,
                    help='floor on that taper, so a converging push cannot stall')
    ap.add_argument('--lookahead', type=int, default=3)
    ap.add_argument('--advance-window', type=int, default=25,
                    help='how many waypoints ahead the carrot may project in one step. '
                         'Bounds how far the reference index can jump, and keeps the '
                         'projection from snapping across a self-intersection.')
    # Success threshold.  Quartered from the original 9.0 units / 12 deg.  Kept separate
    # from --reach, which used to serve as both: tightening a shared knob would have
    # slowed carrot advance along the whole path rather than only tightening success.
    ap.add_argument('--goal-dist', type=float, default=2.25,
                    help='position tolerance for declaring the goal reached, in world '
                         'units (the T is 60 units across its crossbar)')
    ap.add_argument('--goal-deg', type=float, default=3.0,
                    help='heading tolerance for declaring the goal reached, in degrees')
    ap.add_argument('--press', type=float, default=0.6,
                    help='how far past the surface the pusher aims, so it keeps pressing')
    ap.add_argument('--clearance', type=float, default=1.0)
    ap.add_argument('--standoff', type=float, default=8.0)
    ap.add_argument('--arc-step', type=float, default=12.0, help='degrees per arc waypoint')
    ap.add_argument('--retract-dirs', type=int, default=37,
                    help='candidate escape directions tried when backing the pusher out '
                         'to the stand-off circle, ordered outward from the natural '
                         'back-off direction')
    ap.add_argument('--wp-tol', type=float, default=3.0)
    ap.add_argument('--leg-timeout', type=float, default=1.5,
                    help='seconds a single reposition waypoint may take before the chain '
                         'is abandoned and re-solved')
    ap.add_argument('--repel-dist', type=float, default=8.0,
                    help='footprint clearance below which obstacles repel the T')
    ap.add_argument('--k-repel', type=float, default=8.0,
                    help='strength of that repulsion, in the same units as k_pos * error. '
                         'Swept over {0, 4, 8, 15} x repel_dist {8, 16}: 8/8 gave the '
                         'lowest cross-track error and the largest share of time actually '
                         'pushing.  The landscape is rough, so treat these as a working '
                         'setting rather than an optimum.')
    ap.add_argument('--walk-candidates', type=int, default=12,
                    help='how many ranked contacts to try planning a walk-around to '
                         'before declaring the pusher blocked')
    ap.add_argument('--probe', type=int, default=60,
                    help='how many distinct boundary points to test for straight-line '
                         'reachability before giving up and repositioning')
    ap.add_argument('--switch-margin', type=float, default=0.15,
                    help='alignment a challenger face must beat the committed face by '
                         'before it starts earning credit toward a switch')
    ap.add_argument('--face-width', type=float, default=0.6,
                    help='half-width in boundary angle (rad) of a contact "face". The '
                         'pusher re-solves freely inside the committed face, so this sets '
                         'how far the contact may slide before it counts as a switch.')
    ap.add_argument('--switch-hold', type=float, default=1.0,
                    help='seconds a single challenger must beat the committed face '
                         'WITHOUT INTERRUPTION before the face changes. This is the '
                         'de-noiser: an isolated good frame can never win.')
    ap.add_argument('--stall-seconds', type=float, default=1.5,
                    help='no reference progress and no T motion for this long -> ban the '
                         'current contact and re-solve')
    ap.add_argument('--stall-eps', type=float, default=0.5)
    ap.add_argument('--ban-seconds', type=float, default=2.5)
    ap.add_argument('--ban-width', type=float, default=0.7,
                    help='half-width (rad, in boundary angle about the centroid) of the '
                         'face banned when a contact stalls')
    # closed-loop replanning (MPC): re-solve the reference from the T's live pose
    ap.add_argument('--replan-every', type=float, default=0.5,
                    help='seconds between replans of the T reference path from the T\'s '
                         'MEASURED pose. Runs on a worker thread, so the physics never '
                         'blocks on it.')
    ap.add_argument('--replan-min-gap', type=float, default=1.0,
                    help='floor on the gap between replans triggered by drift or a stall '
                         '(the --replan-every timer is a separate, longer cadence)')
    ap.add_argument('--replan-xte', type=float, default=25.0,
                    help='also replan whenever the T drifts further than this off its '
                         'current reference -- at that point the path it is tracking no '
                         'longer describes where it is')
    ap.add_argument('--replan-steps', type=int, default=200,
                    help='MPPI steps per replan (the rollout exits early on convergence, '
                         'so replans get cheaper as the T nears the goal)')
    ap.add_argument('--no-replan', action='store_true',
                    help='track the initial path open-loop at the PLANNING level (the '
                         'pusher control loop stays closed either way)')
    # experiment: skip the end effector and drive the T with a wrench directly
    ap.add_argument('--direct', action='store_true',
                    help='virtual-pusher experiment: the IK still chooses a contact point '
                         'and an in-cone direction, and the force is applied AT that point '
                         'so torque comes from the real moment arm -- but no cylinder has '
                         'to reach it. Isolates end-effector cost from contact mechanics.')
    # experiment: discrete push primitives executed by teleporting the pusher
    ap.add_argument('--teleport', action='store_true',
                    help='primitive experiment: sample (surface point, push direction) '
                         'actions, pick the one whose MEASURED effect best matches the '
                         'next chunk of path, teleport the pusher behind it and push. One '
                         'inference per action instead of per frame; no walk-arounds.')
    ap.add_argument('--action-steps', type=int, default=10,
                    help='how many reference waypoints ahead one action aims to cover. '
                         'With --spacing this also sets the push length.')
    ap.add_argument('--push-len', type=float, default=None,
                    help='pusher travel per action, world units. Default '
                         'spacing*action_steps, i.e. exactly the chunk of path the action '
                         'is aiming at -- measured T travel is ~0.93x pusher travel, so '
                         'the two are close to 1:1.')
    ap.add_argument('--teleport-interp', action='store_true',
                    help='same as --teleport, except the pusher FLIES to each entry point '
                         'instead of being teleported there: straight in when the straight '
                         'line is clear of the T, otherwise out along the surface normal '
                         'to the stand-off circle, round the circle, and back in along the '
                         'entry point\'s own normal. Implies --teleport.')
    ap.add_argument('--transit-speed', type=float, default=0.0,
                    help='pusher speed during a --teleport-interp transit, units/s '
                         '(0 = same as --pusher-speed)')
    ap.add_argument('--transit-timeout', type=float, default=4.0,
                    help='seconds before a transit is abandoned and the push starts from '
                         'wherever the pusher got to')
    ap.add_argument('--push-len-min', type=float, default=5.0,
                    help='shortest push in the length ladder, world units. The action '
                         'length is selected per action, so pushes shrink towards this as '
                         'the T closes on the goal.')
    ap.add_argument('--push-len-steps', type=int, default=4,
                    help='how many push lengths to measure, geometrically spaced between '
                         '--push-len-min and --push-len. 1 disables the taper.')
    ap.add_argument('--len-bias', type=float, default=0.02,
                    help='preference for long pushes, in world units of cost per unit of '
                         'unused length. Keeps the controller from creeping in tiny '
                         'actions far from the goal; too large and it will not taper.')
    ap.add_argument('--n-contacts', type=int, default=24,
                    help='surface points sampled evenly around the perimeter')
    ap.add_argument('--n-dirs', type=int, default=5,
                    help='push directions per contact, fanned about the inward normal')
    ap.add_argument('--dir-spread', type=float, default=45.0,
                    help='half-angle of that fan, degrees')
    ap.add_argument('--action-timeout', type=float, default=4.0,
                    help='seconds before an action is abandoned and re-planned')
    ap.add_argument('--direct-gain', type=float, default=3.0,
                    help='direct-wrench magnitude, in multiples of the ground friction '
                         'limits (--friction / --spin-friction), so 1.0 is exactly on the '
                         'edge of moving the T at all')
    ap.add_argument('--headless', action='store_true',
                    help='run the physics without viser and print a tracking report')
    ap.add_argument('--max-seconds', type=float, default=0.0,
                    help='stop after this much simulated time (0 = run forever)')
    return ap.parse_args()


def main():
    args = build_args()
    if args.teleport_interp:
        args.teleport = True

    # ---- geometry ------------------------------------------------------------------
    env_mesh = base.load_mesh(args.env)
    tee_mesh = base.load_mesh(args.shape)
    env_polys = sorted(base.footprint(env_mesh), key=lambda p: -p.area)
    tee_polys = base.footprint(tee_mesh)
    assert len(tee_polys) == 1, f'expected one footprint for the shape, got {len(tee_polys)}'

    c = tee_polys[0].centroid
    z0 = tee_mesh.bounds[0][2]
    tee_mesh.apply_translation([-c.x, -c.y, -z0])
    tee_poly = Polygon([(x - c.x, y - c.y) for x, y in tee_polys[0].exterior.coords])
    tee_height = float(tee_mesh.extents[2])

    # The planner poses the shape about the z-up mesh's bounding-box centre.
    Vz = np.array([[float(t) for t in ln.split()[1:4]]
                   for ln in open(args.shape_zup) if ln.startswith('v ')])
    bbox_c = 0.5 * (Vz.min(0) + Vz.max(0))
    bbox_to_centroid = np.array([c.x - bbox_c[0], c.y - bbox_c[1]])

    # ---- reference path --------------------------------------------------------------
    womodel = goal_norm = meta = None
    if args.traj:
        ref = np.load(args.traj)
        assert ref.ndim == 2 and ref.shape[1] == 3, f'--traj must be (T,3), got {ref.shape}'
        print(f'[plan] replaying {args.traj}: {len(ref)} waypoints')
        if not args.no_replan:
            print('[plan] --traj has no network behind it, so the reference cannot be '
                  'replanned; the pusher control loop is still closed.')
    else:
        with open(os.path.join(args.dataPath, 'meta.json')) as f:
            meta = json.load(f)
        womodel, start_norm, goal_norm = load_planner(
            args.dataPath, args.modelPath, args.ckpt, args.case, args.plan_device)
        path_norm, dist = plan_from(womodel, start_norm, goal_norm,
                                    args.plan_device, args.mppi_steps)
        print(f'[plan] case {args.case}: {len(path_norm)} waypoints, '
              f'final |goal - x| = {dist:.4f} '
              f'({"converged" if dist < 0.01 else "DID NOT converge"})')
        ref = planner_to_world(path_norm, meta['env_scale'], meta['env_center'],
                               bbox_to_centroid)

    ref = resample_path(ref, args.spacing, args.smooth)
    print(f'[plan] reference: {len(ref)} waypoints, '
          f'{np.linalg.norm(np.diff(ref[:, :2], axis=0), axis=1).sum():.0f} units of travel, '
          f'net rotation {math.degrees(ref[-1, 2] - ref[0, 2]):+.0f} deg')
    if args.save_path:
        np.save(args.save_path, ref)
        print(f'[plan] wrote {args.save_path}')
        return

    # ---- push IK -------------------------------------------------------------------
    c_len = args.c_length if args.c_length else args.spin_friction / args.friction
    # Both of these are read off the simulator rather than tuned separately: c is the
    # anisotropy of the ground-friction joint pair, mu the product Chipmunk forms from the
    # two shape coefficients.  Change the physics and the contact model follows.
    mu = args.mu if args.mu else base.pair_friction(base.SURF_PUSHER, base.SURF_TEE)
    args.mu = mu
    ik = PushIK(tee_poly, c_len, mu=mu, n_boundary=args.n_boundary, n_cone=args.n_cone)
    gyration = math.sqrt(sum(
        pymunk.moment_for_poly(Polygon(p).area / tee_poly.area, p, (0, 0))
        for p in base.convex_parts(tee_poly)))
    print(f'[ik] c = {c_len:.1f} (spin/linear friction ratio); '
          f'radius of gyration = {gyration:.1f}; T bounding radius = {ik.radius:.1f}; '
          f'{len(ik.P)} boundary samples x {args.n_cone} cone directions')
    print(f'[ik] mu = {mu:.3f} = {base.SURF_PUSHER[0]:.2f} (pusher) x '
          f'{base.SURF_TEE[0]:.2f} (T), Chipmunk\'s multiplicative pair rule '
          f'-> friction cone half-angle {math.degrees(math.atan(mu)):.1f} deg')

    # ---- sim ---------------------------------------------------------------------
    tee_start = (float(ref[0, 0]), float(ref[0, 1]))
    # Park the pusher on the standoff circle BEHIND the initial motion, so the very first
    # thing it does is press rather than walk around the T.
    heading = ref[min(args.lookahead, len(ref) - 1), :2] - ref[0, :2]
    if np.linalg.norm(heading) < 1e-9:
        heading = np.array([1.0, 0.0])
    heading = heading / np.linalg.norm(heading)
    pusher_start = np.asarray(tee_start) - heading * (
        ik.radius + args.pusher_radius + args.standoff)
    lo, hi = env_polys[0].bounds[:2], env_polys[0].bounds[2:]
    park = (lo[0] - 200.0, lo[1] - 200.0)      # well outside the walls, for --direct
    sim = PushSim(args, env_polys, tee_poly, tee_start, float(ref[0, 2]),
                  tuple(pusher_start), park=park)
    if args.direct:
        sim.set_direct(True)
        print('[experiment] virtual pusher: contact chosen by the same IK, force applied '
              f'at the contact point at {args.direct_gain:.1f}x the linear friction limit; '
              'no cylinder, so no reachability or walk-arounds')

    obstacles = list(env_polys)          # wall ring + interior blocks, all solid
    ctrl = PushController(ik, ref, args, obstacles)

    push_len = args.push_len if args.push_len else args.spacing * args.action_steps
    push_lens = push_length_ladder(push_len, args.push_len_min, args.push_len_steps)
    prims = PushPrimitives(ik, n_points=args.n_contacts, n_dirs=args.n_dirs,
                           spread_deg=args.dir_spread)
    print(f'[primitives] {len(prims)} actions = {prims.n_points} contacts x '
          f'{prims.n_dirs} directions (+-{args.dir_spread:.0f} deg), push length '
          f'{push_len:.1f} units = {args.action_steps} x {args.spacing:.1f}')
    print('[primitives] length ladder: '
          + ', '.join(f'{L:.1f}' for L in push_lens)
          + ' units (chosen per action; short pushes win near the goal)')
    prims.calibrate(args, tee_poly, push_lens,
                    cache_dir=os.path.dirname(args.dataPath) or '.')
    runner = PrimitiveRunner(ctrl, prims, push_len, args)
    if args.teleport:
        sim.teleport_mode = True

    rep = None
    if womodel is not None and not args.no_replan:
        rep = Replanner(womodel, goal_norm, args.plan_device, args.replan_steps,
                        meta['env_scale'], meta['env_center'], bbox_to_centroid,
                        args.spacing, args.smooth)
        print(f'[replan] closed-loop planning ON: every {args.replan_every:.1f}s, '
              f'on drift > {args.replan_xte:.0f} units, and on every stall')

    if args.headless:
        run_headless(args, sim, ctrl, ref, rep, env_polys[0].bounds, runner)
        return
    run_viser(args, sim, ctrl, ref, env_mesh, env_polys, tee_mesh, tee_poly, tee_height,
              rep, runner)


def replan_tick(rep, ctrl, sim, args, state, enabled=True, interval=None, force=False,
                sync=False):
    """Adopt any finished replan, then decide whether to request another.

    Fires on three conditions, not just a timer: the timer, the T drifting further than
    ``--replan-xte`` off the reference (the path it is tracking no longer describes where
    it is), and a stall (the controller has already given up on a contact, so the geometry
    it planned for is probably wrong).  Returns a short status string for display.
    """
    fresh = rep.poll()
    if fresh is not None:
        ctrl.adopt(fresh)
        state['last'] = state['t']
        state['xte'] = 0.0
    if not enabled or ctrl.done:
        return f"replans {rep.n_done}"

    pos = np.array([sim.tee.position.x, sim.tee.position.y])
    try:
        state['xte'] = LineString(ctrl.ref[:, :2]).distance(Point(pos))
    except Exception:
        state['xte'] = 0.0
    every = args.replan_every if interval is None else interval
    since = state['t'] - state['last']
    due = since >= every
    drifted = state['xte'] > args.replan_xte
    stalled = ctrl.n_forced > state.get('nf', 0)
    state['nf'] = ctrl.n_forced
    # The drift and stall triggers bypass the interval, so they need their own floor or a
    # persistent problem requests a replan on every single frame.
    reactive = (drifted or stalled) and since >= args.replan_min_gap
    if (due or reactive or force) and not rep.busy:
        if sync:
            state['last'] = state['t']
            fresh = rep.solve_now((pos[0], pos[1], sim.tee.angle))
            if fresh is not None:
                ctrl.adopt(fresh)
                state['xte'] = 0.0
            return f"replans {rep.n_done}"
        if rep.request((pos[0], pos[1], sim.tee.angle)):
            state['last'] = state['t']
            return f"replans {rep.n_done} (solving)"
    return (f"replans {rep.n_done}  xte {state['xte']:.0f}"
            + ("  [solving]" if rep.busy else ""))


def escaped(sim, bounds, margin=100.0):
    """Has the T left the world?  pymunk can eject a pinched body through a thin wall.

    Segment walls plus an infinite-mass kinematic pusher can produce an unresolvable
    overlap, and the solver answers by flinging the T thousands of units away.  Detecting
    it keeps a blow-up from being reported as a merely-bad tracking number.
    """
    x, y = sim.tee.position.x, sim.tee.position.y
    return not (bounds[0] - margin <= x <= bounds[2] + margin
                and bounds[1] - margin <= y <= bounds[3] + margin)


class PrimitiveRunner:
    """Drives the teleport mode: pick an action, teleport, push, repeat.

    One inference per ACTION, not per frame.  While an action is executing nothing is
    re-decided -- that is what makes the switching count small and the motion legible.
    Closed-loop-ness comes from the fact that the next action is chosen from the T's
    measured pose, so per-action modelling error is corrected rather than accumulated.
    """

    def __init__(self, ctrl, prims, push_len, args):
        self.ctrl, self.prims, self.L, self.a = ctrl, prims, push_len, args
        self.reset()

    def reset(self):
        self.final = None
        self.frames = 0
        self.n_actions = 0
        self.cur_len = self.L
        self.queue = []             # --teleport-interp: body-frame transit waypoints
        self.final_local = None
        self.t_frames = 0
        self.n_transit = 0
        self.n_arc = 0
        self.status = 'idle'

    def _tee(self, sim):
        return np.array([sim.tee.position.x, sim.tee.position.y]), sim.tee.angle

    def _fly(self, sim):
        """Advance along the transit queue.  True while still flying, False when arrived.

        Waypoints are stored in the T's body frame and mapped to world every frame, so a
        T that gets nudged during the transit carries the remaining path with it.
        """
        pos, ang = self._tee(sim)
        self.t_frames += 1
        while self.queue:
            w = to_world(self.queue[0], pos, ang)
            if (pymunk.Vec2d(*w) - sim.pusher.position).length < self.a.wp_tol:
                self.queue.pop(0)
                continue
            if self.t_frames > self.a.transit_timeout * self.a.fps:
                break                       # give up and let the push start from here
            sim.target = pymunk.Vec2d(float(w[0]), float(w[1]))
            sim.max_speed = self.a.transit_speed or self.a.pusher_speed
            return True
        self.queue = []
        return False

    def _start_push(self, sim):
        """Begin the push leg, re-deriving the end point from the T's CURRENT pose."""
        pos, ang = self._tee(sim)
        self.final = to_world(self.final_local, pos, ang)
        self.frames = 0
        sim.target = pymunk.Vec2d(float(self.final[0]), float(self.final[1]))
        sim.max_speed = self.a.pusher_speed

    def update(self, sim):
        """Call once per frame.  Returns a status string."""
        if self.ctrl.done:
            self.final = None
            self.queue = []
            return 'GOAL'

        if self.queue:
            if self._fly(sim):
                return self.status
            self._start_push(sim)
            return self.status

        if self.final is not None:
            self.frames += 1
            reached = (pymunk.Vec2d(*self.final) - sim.pusher.position).length < self.a.wp_tol
            # Timeout scaled by the push length: a short action that is going nowhere
            # should not hold the controller for as long as a full-length one.
            budget = self.a.action_timeout * self.a.fps * max(0.25, self.cur_len / self.L)
            if not reached and self.frames < budget:
                sim.target = pymunk.Vec2d(float(self.final[0]), float(self.final[1]))
                sim.max_speed = self.a.pusher_speed
                return self.status
            self.final = None                      # action finished (or timed out)

        tele, final, _c, st = self.ctrl.begin_primitive(
            (sim.tee.position.x, sim.tee.position.y), sim.tee.angle, self.prims, self.L)
        self.status = st
        if tele is None:
            sim.max_speed = 0.0
            return st
        self.cur_len = self.ctrl.last_len or self.L
        self.n_actions += 1
        self.final_local = self.ctrl.final_local

        if self.a.teleport_interp:
            # Fly there rather than teleport.  The push itself is unchanged; only how the
            # pusher reaches the entry point differs, and plan_transit always returns one.
            pos, ang = self._tee(sim)
            q, marks = self.ctrl.plan_transit(
                to_local((sim.pusher.position.x, sim.pusher.position.y), pos, ang),
                self.ctrl.entry_local, pos, ang)
            self.queue, self.t_frames = q, 0
            self.n_transit += 1
            self.n_arc += marks is not None
            self.status = st + (' +arc' if marks else ' +straight') + f'({len(q)})'
            self._fly(sim)
            return self.status

        sim.teleport_pusher(tele)
        self._start_push(sim)
        return st


def pose_error(sim, ref):
    p = np.array([sim.tee.position.x, sim.tee.position.y])
    return np.linalg.norm(ref[-1, :2] - p), abs(wrap(ref[-1, 2] - sim.tee.angle))


def run_headless(args, sim, ctrl, ref, rep=None, bounds=None, runner=None):
    """Physics-only run; prints how well the T tracked the plan.  Used to sanity-check."""
    dt = 1.0 / args.fps
    limit = args.max_seconds or 120.0
    trace, t, states = [], 0.0, {}
    rstate = {'t': 0.0, 'last': -1e9, 'xte': 0.0}
    # Distance to the GOAL is the only progress measure that stays comparable when the
    # reference is being replanned -- reference index and cross-track error are both
    # relative to a path that changes underneath them.
    goal_xy = ref[-1, :2].copy()
    goal_hist = []
    blew_up = False
    while t < limit and not ctrl.done:
        if bounds is not None and escaped(sim, bounds):
            blew_up = True
            break
        if rep is not None:
            rstate['t'] = t
            replan_tick(rep, ctrl, sim, args, rstate, enabled=not args.no_replan,
                        sync=True)
        if sim.teleport_mode:
            status = runner.update(sim)
        elif sim.direct:
            sim.contact, _, status = ctrl.direct_wrench(
                (sim.tee.position.x, sim.tee.position.y), sim.tee.angle,
                (sim.tee.velocity.x, sim.tee.velocity.y), sim.tee.angular_velocity)
        else:
            target, _, status = ctrl.step(
                (sim.tee.position.x, sim.tee.position.y), sim.tee.angle,
                (sim.pusher.position.x, sim.pusher.position.y),
                (sim.tee.velocity.x, sim.tee.velocity.y), sim.tee.angular_velocity)
            sim.max_speed = args.pusher_speed * ctrl.speed_scale
            sim.target = pymunk.Vec2d(float(target[0]), float(target[1]))
        key = status.split()[0]
        states[key] = states.get(key, 0) + 1
        for _ in range(args.substeps):
            sim.step(dt / args.substeps)
        trace.append([sim.tee.position.x, sim.tee.position.y, sim.tee.angle])
        goal_hist.append(np.linalg.norm(goal_xy - np.array([sim.tee.position.x,
                                                            sim.tee.position.y])))
        t += dt
    trace = np.array(trace)
    # Cross-track error: distance from each realized T position to the reference polyline.
    # NOTE: with replanning on, the reference changes during the run, so a single
    # cross-track number against any one path is not meaningful; xte is sampled live in
    # rstate instead and reported as the mean of what the controller actually saw.
    gh = np.array(goal_hist)
    dp, dth = pose_error(sim, ref)
    verdict = ('GOAL REACHED' if ctrl.done else
               'BLEW UP -- the T was ejected from the world by the physics solver'
               if blew_up else 'not reached')
    print(f'\n[headless] {t:.1f}s simulated, {verdict}')
    print(f'[headless] distance to goal: start {gh[0]:.1f} -> final {gh[-1]:.1f}  '
          f'(closest approach {gh.min():.1f} at t={dt * int(np.argmin(gh)):.1f}s)')
    print(f'[headless] final pose error: {dp:.1f} units, {math.degrees(dth):.1f} deg')
    print(f'[headless] reference index reached: {ctrl.k}/{len(ctrl.ref) - 1}'
          + ('  [reference was replanned, so this is progress along the LATEST path]'
             if rep is not None and rep.n_done else ''))
    if runner is not None and sim.teleport_mode:
        print(f'[headless] primitive actions executed: {runner.n_actions}'
              f'  ({t / max(runner.n_actions, 1):.1f}s per action)')
        if args.teleport_interp:
            print(f'[headless] transits flown: {runner.n_transit} '
                  f'({runner.n_arc} around the stand-off circle, '
                  f'{runner.n_transit - runner.n_arc} straight in); 0 teleported')
    print(f'[headless] committed contact-face switches: {ctrl.n_switch}'
          f'  ({t / max(ctrl.n_switch, 1):.1f}s per face)')
    if rep is not None:
        print(f'[headless] replans completed: {rep.n_done}'
              + (f'  (last took {rep.last_ms:.0f} ms)' if rep.n_done else ''))
    if rep is not None:
        for _ in range(60):                 # a daemon thread mid-CUDA-call at interpreter
            if not rep.busy:                # shutdown can abort the process; let it land
                break
            time.sleep(0.1)
    tot = sum(states.values()) or 1
    print('[headless] time in state: '
          + '  '.join(f'{k}={100.0 * v / tot:.0f}%' for k, v in sorted(states.items())))


def _ref_segments(ref, z):
    """(N,2,3) line segments for a reference path drawn at height ``z``."""
    a = np.column_stack([ref[:-1, 0], ref[:-1, 1], np.full(len(ref) - 1, z)])
    b = np.column_stack([ref[1:, 0], ref[1:, 1], np.full(len(ref) - 1, z)])
    return np.stack([a, b], axis=1).astype(np.float32)


def run_viser(args, sim, ctrl, ref, env_mesh, env_polys, tee_mesh, tee_poly, tee_height,
              rep=None, runner=None):
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction('+z')
    server.scene.world_axes.visible = False

    lo, hi = env_polys[0].bounds[:2], env_polys[0].bounds[2:]
    fw = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.15
    fc = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
    floor = server.scene.add_box('/floor', color=base.COL_FLOOR, dimensions=(fw, fw, 2.0),
                                 position=(fc[0], fc[1], -1.0))
    grid = server.scene.add_grid('/grid', width=fw, height=fw, plane='xy', cell_size=10.0,
                                 section_size=50.0, position=(fc[0], fc[1], 0.02))
    env_node = server.scene.add_mesh_simple('/env', env_mesh.vertices, env_mesh.faces,
                                            color=base.COL_ENV, flat_shading=True)

    # Reference path: the planned trace of the T's origin, plus a ghost T at the goal.
    ref_z = tee_height + 1.0
    ref_node = {'n': server.scene.add_line_segments(
        '/ref', _ref_segments(ref, ref_z), colors=COL_REF, thickness=3.0)}
    goal_frame = server.scene.add_frame('/goal', show_axes=False,
                                        position=(ref[-1, 0], ref[-1, 1], 0.0),
                                        wxyz=base.quat_z(ref[-1, 2]))
    server.scene.add_mesh_simple('/goal/mesh', tee_mesh.vertices, tee_mesh.faces,
                                 color=COL_GHOST, flat_shading=True, opacity=0.35,
                                 material='standard')

    tee_frame = server.scene.add_frame('/tee', show_axes=False, position=(*ref[0, :2], 0.0),
                                       wxyz=base.quat_z(ref[0, 2]))
    server.scene.add_mesh_simple('/tee/mesh', tee_mesh.vertices, tee_mesh.faces,
                                 color=base.COL_TEE, flat_shading=True)

    pusher_mesh = trimesh.creation.cylinder(radius=args.pusher_radius,
                                            height=args.pusher_height, sections=32)
    pusher_mesh.apply_translation([0.0, 0.0, args.pusher_height / 2.0])
    pusher_frame = server.scene.add_frame('/pusher', show_axes=False,
                                          position=(sim.pusher.position.x,
                                                    sim.pusher.position.y, 0.0))
    server.scene.add_mesh_simple('/pusher/mesh', pusher_mesh.vertices, pusher_mesh.faces,
                                 color=base.COL_PUSHER)

    contact_frame = server.scene.add_frame('/contact', show_axes=False, visible=False)
    server.scene.add_icosphere('/contact/dot', radius=2.5, color=COL_CONTACT)

    # The stand-off circle a blocked transit rides, drawn as a child of the T so it
    # follows the object, plus the exit / entry points the current transit uses.
    circ_z = tee_height + 0.5
    ring = np.array([[ctrl.standoff * math.cos(a), ctrl.standoff * math.sin(a), circ_z]
                     for a in np.linspace(0.0, 2 * math.pi, 121)], dtype=np.float32)
    standoff_node = server.scene.add_line_segments(
        '/tee/standoff', np.stack([ring[:-1], ring[1:]], axis=1),
        colors=COL_STANDOFF, thickness=2.0, visible=args.teleport_interp)
    exit_frame = server.scene.add_frame('/tee/exit', show_axes=False, visible=False)
    server.scene.add_icosphere('/tee/exit/dot', radius=3.0, color=COL_EXIT)
    entry_frame = server.scene.add_frame('/tee/entry', show_axes=False, visible=False)
    server.scene.add_icosphere('/tee/entry/dot', radius=3.0, color=COL_ENTRY)
    transit_node = {'n': None}

    server.scene.add_light_directional('/sun', color=(255, 255, 255), intensity=2.0,
                                       position=(fc[0] + 100, fc[1] - 150, 400))

    # ---- GUI -----------------------------------------------------------------------
    with server.gui.add_folder('Experiment'):
        actuation = server.gui.add_dropdown(
            'actuation',
            options=('pusher (contact)', 'direct wrench', 'teleport primitives'),
            initial_value=('teleport primitives' if args.teleport else
                           'direct wrench' if args.direct else 'pusher (contact)'))
        server.gui.add_markdown(
            'Direct wrench = a **virtual pusher**: the same IK picks a contact point and '
            'an in-cone direction, and the force is applied at that point (torque comes '
            'from the moment arm). Only the cylinder itself is removed -- no reachability, '
            'no walk-arounds. The red dot still shows the contact.\n\n'
            '**Teleport primitives** picks one of a discrete set of (surface point, push '
            'direction) actions using a MEASURED model of what each one does, teleports '
            'the cylinder behind it and pushes for a fixed distance. One decision per '
            'action, not per frame.')
        interp = server.gui.add_checkbox('interpolate transits', args.teleport_interp)
        server.gui.add_markdown(
            'With **interpolate transits** on, the cylinder is not teleported: it flies '
            'to each entry point, straight in when the line is clear and otherwise out '
            'along the surface normal, round the stand-off circle, and back in along the '
            'entry normal. Same pushes, visible travel between them.')
    with server.gui.add_folder('Run'):
        running = server.gui.add_checkbox('run', True)
        time_scale = server.gui.add_slider('time scale', 0.1, 3.0, 0.05, 1.0)
        reset_btn = server.gui.add_button('reset')
    with server.gui.add_folder('Controller'):
        g_pos = server.gui.add_slider('k_pos', 0.0, 5.0, 0.05, args.k_pos)
        g_ang = server.gui.add_slider('k_ang', 0.0, 20.0, 0.1, args.k_ang)
        g_look = server.gui.add_slider('lookahead', 0, 20, 1, args.lookahead)
        g_speed = server.gui.add_slider('pusher speed', 5.0, 300.0, 5.0, args.pusher_speed)
    with server.gui.add_folder('Replan (closed-loop planning)'):
        if rep is None:
            server.gui.add_markdown(
                'Replanning needs the network; this run has none (`--traj` / '
                '`--no-replan`). The pusher control loop is closed regardless.')
            do_replan = None
            replan_every = None
            replan_btn = None
        else:
            do_replan = server.gui.add_checkbox('replan from live pose', True)
            replan_every = server.gui.add_slider('interval (s)', 0.5, 15.0, 0.5,
                                                 args.replan_every)
            replan_btn = server.gui.add_button('replan now')
    with server.gui.add_folder('Physics'):
        fric = server.gui.add_slider('ground friction', 0.0, 2000.0, 10.0, args.friction)
        spin = server.gui.add_slider('spin friction', 0.0, 60000.0, 250.0, args.spin_friction)
    with server.gui.add_folder('View'):
        show_env = server.gui.add_checkbox('environment', True)
        show_ref = server.gui.add_checkbox('reference path', True)
        show_goal = server.gui.add_checkbox('goal ghost', True)
        show_trace = server.gui.add_checkbox('actual trace', True)
        show_floor = server.gui.add_checkbox('floor', True)
        show_grid = server.gui.add_checkbox('grid', True)
        show_circle = server.gui.add_checkbox('stand-off circle', args.teleport_interp)
        server.gui.add_markdown(
            'Grey ring = the stand-off circle (smallest circle enclosing the shape, grown '
            'by the pusher radius and `--standoff`). Yellow dot = the transit\'s exit '
            'point, blue dot = its entry point, yellow line = the path being flown.')
    status = server.gui.add_markdown('')

    trace_pts = [np.array([*ref[0, :2], ref_z])]
    trace_node = {'n': None}

    @actuation.on_update
    def _(_):
        mode = actuation.value
        sim.set_direct(mode.startswith('direct'))
        sim.teleport_mode = mode.startswith('teleport')
        if runner is not None:
            runner.reset()
        pusher_frame.visible = not mode.startswith('direct')
        contact_frame.visible = False

    @interp.on_update
    def _(_):
        args.teleport_interp = interp.value
        if runner is not None:
            runner.queue = []

    @reset_btn.on_click
    def _(_):
        sim.reset()
        ctrl.reset()
        trace_pts.clear()
        trace_pts.append(np.array([*ref[0, :2], ref_z]))
        if trace_node['n'] is not None:
            trace_node['n'].remove()
            trace_node['n'] = None

    @fric.on_update
    def _(_):
        sim.set_friction(fric.value, spin.value)

    @spin.on_update
    def _(_):
        sim.set_friction(fric.value, spin.value)

    @show_env.on_update
    def _(_):
        env_node.visible = show_env.value

    @show_ref.on_update
    def _(_):
        ref_node['n'].visible = show_ref.value

    @show_goal.on_update
    def _(_):
        goal_frame.visible = show_goal.value

    @show_floor.on_update
    def _(_):
        floor.visible = show_floor.value

    @show_grid.on_update
    def _(_):
        grid.visible = show_grid.value

    @show_trace.on_update
    def _(_):
        if trace_node['n'] is not None:
            trace_node['n'].visible = show_trace.value

    @show_circle.on_update
    def _(_):
        standoff_node.visible = show_circle.value
        if not show_circle.value:
            exit_frame.visible = entry_frame.visible = False
            if transit_node['n'] is not None:
                transit_node['n'].remove()
                transit_node['n'] = None

    force = {'v': False}
    if replan_btn is not None:
        @replan_btn.on_click
        def _(_):
            force['v'] = True

    print(f'[push-demo] serving on http://localhost:{args.port}')
    print(f'[push-demo] reference {len(ref)} waypoints; green line = planned T path, '
          f'green ghost = goal pose, red dot = current contact point')

    frame_dt = 1.0 / args.fps
    next_frame = time.time()
    last_status = 0.0
    rstate = {'t': 0.0, 'last': -1e9, 'xte': 0.0}
    sim_t = 0.0
    rstatus = 'replanning off'
    world = env_polys[0].bounds
    while True:
        if running.value and escaped(sim, world):
            running.value = False
            st = 'BLEW UP -- T ejected from the world; press reset'
            print('[push-demo] ' + st)
        if running.value:
            args.k_pos, args.k_ang, args.lookahead = g_pos.value, g_ang.value, int(g_look.value)
            if rep is not None:
                rstate['t'] = sim_t
                before = ctrl.ref
                rstatus = replan_tick(rep, ctrl, sim, args, rstate,
                                      enabled=do_replan.value,
                                      interval=replan_every.value, force=force['v'])
                force['v'] = False
                if ctrl.ref is not before:          # a replan landed -- redraw the path
                    ref_node['n'].remove()
                    ref_node['n'] = server.scene.add_line_segments(
                        '/ref', _ref_segments(ctrl.ref, ref_z), colors=COL_REF,
                        thickness=3.0, visible=show_ref.value)
            if sim.teleport_mode:
                st = runner.update(sim)
                contact = ctrl.contact_world
            elif sim.direct:
                sim.contact, contact, st = ctrl.direct_wrench(
                    (sim.tee.position.x, sim.tee.position.y), sim.tee.angle,
                    (sim.tee.velocity.x, sim.tee.velocity.y), sim.tee.angular_velocity)
            else:
                target, contact, st = ctrl.step(
                    (sim.tee.position.x, sim.tee.position.y), sim.tee.angle,
                    (sim.pusher.position.x, sim.pusher.position.y),
                    (sim.tee.velocity.x, sim.tee.velocity.y), sim.tee.angular_velocity)
                sim.max_speed = g_speed.value * ctrl.speed_scale
                sim.target = pymunk.Vec2d(float(target[0]), float(target[1]))
            dt = frame_dt * time_scale.value / args.substeps
            for _ in range(args.substeps):
                sim.step(dt)
            sim_t += frame_dt * time_scale.value

            if contact is None:
                contact_frame.visible = False
            else:
                contact_frame.position = (float(contact[0]), float(contact[1]), tee_height + 1.0)
                contact_frame.visible = True

            p = np.array([sim.tee.position.x, sim.tee.position.y, ref_z])
            if np.linalg.norm(p - trace_pts[-1]) > 2.0:
                trace_pts.append(p)
                if len(trace_pts) > 1:
                    if trace_node['n'] is not None:
                        trace_node['n'].remove()
                    segs = np.stack([np.array(trace_pts[:-1]), np.array(trace_pts[1:])],
                                    axis=1).astype(np.float32)
                    trace_node['n'] = server.scene.add_line_segments(
                        '/trace', segs, colors=COL_ACTUAL, thickness=3.0,
                        visible=show_trace.value)
        else:
            st = 'PAUSED'

        tee_frame.position = (sim.tee.position.x, sim.tee.position.y, 0.0)
        tee_frame.wxyz = base.quat_z(sim.tee.angle)
        pusher_frame.position = (sim.pusher.position.x, sim.pusher.position.y, 0.0)

        # Transit overlay.  The circle is a child of /tee so it tracks the object for
        # free; the marks and the flown path are body-frame points lifted to world.
        if show_circle.value and runner is not None and sim.teleport_mode:
            marks = ctrl.transit_marks if runner.queue else None
            for frame, m in ((exit_frame, 0), (entry_frame, 1)):
                if marks is None:
                    frame.visible = False
                else:
                    frame.position = (float(marks[m][0]), float(marks[m][1]), circ_z)
                    frame.visible = True
            if transit_node['n'] is not None:
                transit_node['n'].remove()
                transit_node['n'] = None
            if runner.queue:
                tp, ta = np.array([sim.tee.position.x, sim.tee.position.y]), sim.tee.angle
                pts = [np.array([sim.pusher.position.x, sim.pusher.position.y])] + \
                      [to_world(w, tp, ta) for w in runner.queue]
                pts = np.array([[q[0], q[1], circ_z] for q in pts], dtype=np.float32)
                transit_node['n'] = server.scene.add_line_segments(
                    '/transit', np.stack([pts[:-1], pts[1:]], axis=1),
                    colors=COL_TRANSIT, thickness=2.5)

        now = time.time()
        if now - last_status > 0.1:
            last_status = now
            dp, dth = pose_error(sim, ctrl.ref)
            status.content = (
                f'**{st}**\n\n'
                f'**T**  x={sim.tee.position.x:7.1f}  y={sim.tee.position.y:7.1f}  '
                f'θ={math.degrees(sim.tee.angle) % 360:6.1f}°\n\n'
                f'**to goal**  {dp:6.1f} units, {math.degrees(dth):5.1f}°\n\n'
                f'**pusher**  |v|={sim.pusher.velocity.length:6.1f}\n\n'
                f'**plan**  {rstatus}\n\n'
                f'**face switches**  {ctrl.n_switch}'
                + (f'\n\n**actions**  {runner.n_actions}'
                   + (f'   **transits**  {runner.n_transit}'
                      f' ({runner.n_arc} arc)' if args.teleport_interp else '')
                   if runner is not None and sim.teleport_mode else ''))

        next_frame += frame_dt
        time.sleep(max(0.0, next_frame - time.time()))
        if next_frame < time.time() - 0.5:
            next_frame = time.time()


if __name__ == '__main__':
    main()
