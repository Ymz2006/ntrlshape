"""Real-world push-T on the UR5 -- stage 1: interactive workspace calibration.

Only the calibration stage is implemented.  It is the piece that has to exist before any
planned push can be replayed on hardware: the planner works in its own normalized SE(2)
frame, and mapping that frame onto the physical table needs a handful of TCP poses that a
human has eyeballed into place.  This script is how you collect them -- you jog the pusher
around with the arrow keys and press SPACE at each point you care about.

Two invariants hold for the whole session:

* **The tool always points straight down.**  Every target is rebuilt from scratch as
  ``down_pose(position, yaw)`` rather than by accumulating deltas onto the measured pose,
  so orientation drift cannot creep in over a long jogging session.  On startup the
  current orientation is *snapped* to exactly down while keeping its existing yaw (the arm
  is typically already within a few degrees of down, so this is a small wrist correction).
* **Motion is relative.**  Each keypress offsets the current commanded position; there are
  no absolute waypoints to get wrong.

Everything is bounded before it is sent: the target is clamped to a workspace box, then
checked against the controller's own ``isPoseWithinSafetyLimits`` (reachability + safety
planes), and only then does it become a short ``moveL``.  Keys pressed while a move is in
flight are coalesced into one motion instead of queueing up a backlog of stale jogs.

Frames
------
Poses are UR ``(x, y, z, rx, ry, rz)`` in the robot BASE frame, metres and rotation-vector
radians.  With ``--view-yaw 0`` the right arrow is base +X and the up arrow is base +Y;
set ``--view-yaw`` to the angle you are standing at so the arrows match what you see.

Usage (from the ntrl-demo root, inside the pytorchserver container -- needs a TTY):
    docker exec -it ntrl_ur python push_t_realworld.py --dry-run   # no motion, try the keys
    docker exec -it ntrl_ur python push_t_realworld.py             # live
    docker exec -it ntrl_ur python push_t_realworld.py --view-yaw 90 --step 0.01
"""

import argparse
import json
import math
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime

import numpy as np

ROBOT_IP = "10.168.4.249"

# Robot modes reported by RTDEReceiveInterface.getRobotMode().
ROBOT_MODE_RUNNING = 7
ROBOT_MODE_NAMES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING",
    3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}
SAFETY_MODE_NAMES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
}

STEP_MIN, STEP_MAX = 0.0005, 0.05          # jog step bounds [m]
YAW_STEP = math.radians(5.0)               # per keypress rotation about the down axis


# ======================================================================================
# orientation: the "pointing straight down" family
# ======================================================================================
def rotvec_to_matrix(rv):
    """Rodrigues: rotation vector -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def down_matrix(yaw):
    """Rotation whose tool Z is base -Z (straight down) and whose tool X is at `yaw`.

        R = [[ cos y,  sin y,  0],
             [ sin y, -cos y,  0],
             [     0,      0, -1]]

    Columns are the tool axes in base coordinates; X x Y = Z = (0,0,-1), so this is a
    proper right-handed rotation, not a reflection.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def down_rotvec(yaw):
    """Rotation vector for `down_matrix(yaw)`.

    Every such matrix is a half-turn (its trace is -1), so the general matrix logarithm
    degenerates.  Using R = 2*k*k^T - I for a pi rotation gives (R + I)/2 = k*k^T, whose
    diagonal reads off k = (cos(yaw/2), sin(yaw/2), 0) -- exact, and with no branch.
    """
    return np.array([math.pi * math.cos(yaw / 2.0), math.pi * math.sin(yaw / 2.0), 0.0])


def yaw_of(pose):
    """Yaw of a TCP pose about the vertical, i.e. the heading of the tool X axis."""
    R = rotvec_to_matrix(pose[3:6])
    return math.atan2(R[1, 0], R[0, 0])


def tilt_from_down_deg(pose):
    """Angle between the tool Z axis and straight down, in degrees."""
    R = rotvec_to_matrix(pose[3:6])
    return math.degrees(math.acos(float(np.clip(-R[2, 2], -1.0, 1.0))))


def down_pose(position, yaw):
    """Assemble a UR pose (x,y,z,rx,ry,rz) that points straight down at `yaw`."""
    return np.concatenate([np.asarray(position, dtype=float), down_rotvec(yaw)])


# ======================================================================================
# keyboard
# ======================================================================================
KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT = "UP", "DOWN", "RIGHT", "LEFT"
KEY_PGUP, KEY_PGDN, KEY_ESC = "PGUP", "PGDN", "ESC"

_ESCAPES = {
    "[A": KEY_UP, "[B": KEY_DOWN, "[C": KEY_RIGHT, "[D": KEY_LEFT,
    "OA": KEY_UP, "OB": KEY_DOWN, "OC": KEY_RIGHT, "OD": KEY_LEFT,   # application mode
    "[5~": KEY_PGUP, "[6~": KEY_PGDN,
}


class RawKeyboard:
    """Put stdin in cbreak mode and hand back decoded keys without blocking."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._saved = None

    def __enter__(self):
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def poll(self, timeout):
        """Return every key pressed within `timeout` seconds, in order.

        Reads the whole pending buffer at once so that held-down arrow keys arrive as a
        batch the caller can coalesce, rather than one stale jog per iteration.
        """
        if not select.select([self.fd], [], [], timeout)[0]:
            return []
        data = os.read(self.fd, 1024).decode("utf-8", errors="ignore")
        return self._tokenize(data)

    @staticmethod
    def _tokenize(data):
        keys, i = [], 0
        while i < len(data):
            ch = data[i]
            if ch != "\x1b":
                keys.append(ch)
                i += 1
                continue
            # An escape sequence, or a bare ESC if nothing recognisable follows.
            matched = None
            for length in (3, 2):
                seq = data[i + 1:i + 1 + length]
                if seq in _ESCAPES:
                    matched = (_ESCAPES[seq], 1 + length)
                    break
            if matched is None:
                keys.append(KEY_ESC)
                i += 1
            else:
                keys.append(matched[0])
                i += matched[1]
        return keys


# ======================================================================================
# calibration session
# ======================================================================================
class Calibration:
    """Jog the tool around, record poses, write them out as JSON."""

    def __init__(self, args):
        self.args = args
        self.step = float(np.clip(args.step, STEP_MIN, STEP_MAX))
        self.points = []
        self.message = ""      # feedback from a discrete action, cleared each batch
        self.warn = ""         # clamp / refusal from the last move
        self.rtde_c = None
        self.rtde_r = None
        self.live = False      # True once a control interface exists and may move the arm

        # Arrow-key basis in the base XY plane, rotated to the operator's viewpoint.
        v = math.radians(args.view_yaw)
        self.right = np.array([math.cos(v), math.sin(v), 0.0])
        self.forward = np.array([-math.sin(v), math.cos(v), 0.0])

    # -- connection ---------------------------------------------------------------
    def connect(self):
        from rtde_receive import RTDEReceiveInterface

        print(f"connecting to {self.args.ip} ...")
        self.rtde_r = RTDEReceiveInterface(self.args.ip)
        mode, safety = self.rtde_r.getRobotMode(), self.rtde_r.getSafetyMode()
        print(f"  robot mode  : {mode} ({ROBOT_MODE_NAMES.get(mode, '?')})")
        print(f"  safety mode : {safety} ({SAFETY_MODE_NAMES.get(safety, '?')})")

        if self.args.dry_run:
            print("  --dry-run: no control interface, nothing will move.")
            return
        if mode != ROBOT_MODE_RUNNING:
            raise SystemExit(
                f"\nrobot is in mode {mode} ({ROBOT_MODE_NAMES.get(mode, '?')}), not RUNNING.\n"
                "Power on and release the brakes on the teach pendant (and put it in Remote\n"
                "Control if the control interface refuses to connect), or re-run with --dry-run."
            )

        from rtde_control import RTDEControlInterface

        self.rtde_c = RTDEControlInterface(self.args.ip)
        self.live = True
        print(f"  TCP offset  : {np.round(self.rtde_c.getTCPOffset(), 5).tolist()}")

    def close(self):
        if self.rtde_c is not None:
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
        if self.rtde_r is not None:
            self.rtde_r.disconnect()

    # -- motion -------------------------------------------------------------------
    def clamp(self, position):
        """Clamp a target into the workspace box; returns (position, was_clamped)."""
        lo = np.array([self.args.xlim[0], self.args.ylim[0], self.args.zlim[0]])
        hi = np.array([self.args.xlim[1], self.args.ylim[1], self.args.zlim[1]])
        clamped = np.clip(position, lo, hi)
        return clamped, bool(np.any(np.abs(clamped - position) > 1e-9))

    def move_to(self, position, yaw):
        """Send one bounded moveL to a straight-down pose. Returns True if accepted."""
        position, was_clamped = self.clamp(position)
        target = down_pose(position, yaw)

        self.warn = "at workspace limit" if was_clamped else ""

        if not self.live:                            # dry run: accept, move nothing
            self.position, self.yaw = position, yaw
            return True

        if not self.rtde_c.isPoseWithinSafetyLimits(target.tolist()):
            self.warn = "REFUSED: outside the robot's safety limits"
            return False

        if not self.rtde_c.moveL(target.tolist(), self.args.speed, self.args.accel):
            self.warn = "REFUSED: moveL failed"
            return False
        self.position, self.yaw = position, yaw
        return True

    # -- points -------------------------------------------------------------------
    def record(self):
        pose = (list(self.rtde_r.getActualTCPPose()) if self.live
                else down_pose(self.position, self.yaw).tolist())
        self.points.append({
            "index": len(self.points),
            "name": f"p{len(self.points)}",
            "tcp_pose": pose,
            "joints": list(self.rtde_r.getActualQ()) if self.live else None,
            "commanded": down_pose(self.position, self.yaw).tolist(),
            "yaw_rad": self.yaw,
            "time": datetime.now().isoformat(timespec="seconds"),
        })
        self.message = f"recorded p{len(self.points) - 1}"

    def save(self):
        if not self.points:
            self.message = "nothing to save"
            return
        out = os.path.abspath(self.args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        payload = {
            "robot_ip": self.args.ip,
            "created": datetime.now().isoformat(timespec="seconds"),
            "dry_run": bool(self.args.dry_run),
            "view_yaw_deg": self.args.view_yaw,
            "start_pose": list(self.start_pose),
            "pose_convention": "UR base frame (x,y,z,rx,ry,rz); metres, rotation vector",
            "points": self.points,
        }
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        self.message = f"saved {len(self.points)} point(s) -> {out}"

    # -- display ------------------------------------------------------------------
    def banner(self):
        print(f"""
{'=' * 78}
  push-T real-world calibration -- tool locked pointing straight down
{'=' * 78}
  arrows  jog in the table plane   (up/down = forward/back, left/right)
  w / s   jog +Z / -Z  (up / down in height)     [PgUp / PgDn also work]
  , / .   yaw -5 / +5 deg about the vertical
  [ / ]   step size  x0.5 / x2      (currently {self.step * 1000:.1f} mm)
  SPACE   record this pose      u  undo last      l  list points
  o       re-level (snap exactly down, keep yaw)  h  go back to start pose
  ENTER   save to {self.args.out}
  q/ESC   quit
{'-' * 78}
  arrow frame: right = base {np.round(self.right, 3).tolist()},
               up    = base {np.round(self.forward, 3).tolist()}
  workspace  : x {self.args.xlim}  y {self.args.ylim}  z {self.args.zlim}  [m]
{'=' * 78}
""")

    def status(self):
        actual = (np.array(self.rtde_r.getActualTCPPose()[:3]) if self.live
                  else self.position)
        line = (f"\r x={actual[0]:+.4f} y={actual[1]:+.4f} z={actual[2]:+.4f} m | "
                f"yaw={math.degrees(self.yaw):+7.2f} deg | step={self.step * 1000:5.1f} mm | "
                f"pts={len(self.points)}")
        if self.args.dry_run:
            line += " | DRY-RUN"
        for extra in (self.message, self.warn):
            if extra:
                line += f" | {extra}"
        sys.stdout.write(line.ljust(140))
        sys.stdout.flush()

    # -- main loop ----------------------------------------------------------------
    def run(self):
        self.connect()

        self.start_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
        self.position = self.start_pose[:3].copy()
        self.yaw = yaw_of(self.start_pose)
        tilt = tilt_from_down_deg(self.start_pose)
        print(f"  start TCP   : {np.round(self.start_pose, 5).tolist()}")
        print(f"  yaw         : {math.degrees(self.yaw):.2f} deg")
        print(f"  tool tilt   : {tilt:.2f} deg off straight down")

        if tilt > self.args.max_initial_tilt:
            raise SystemExit(
                f"\nthe tool is {tilt:.1f} deg off vertical, more than --max-initial-tilt "
                f"({self.args.max_initial_tilt} deg).\nJog it roughly down on the pendant "
                "first -- levelling it from here would be a large uncommanded wrist motion."
            )

        self.banner()
        if not self.args.dry_run:
            input("press ENTER to level the tool and begin (Ctrl-C to abort) ...")
            print("levelling ...")
            self.move_to(self.position, self.yaw)

        with RawKeyboard() as kb:
            self.status()
            while True:
                keys = kb.poll(0.05)
                if not keys:
                    continue

                self.message = ""         # stale feedback should not outlive its keypress
                delta = np.zeros(3)       # coalesced translation for this batch
                dyaw = 0.0
                relevel = False

                for key in keys:
                    if key in ("q", KEY_ESC):
                        print()
                        return
                    elif key == KEY_UP:
                        delta += self.step * self.forward
                    elif key == KEY_DOWN:
                        delta -= self.step * self.forward
                    elif key == KEY_RIGHT:
                        delta += self.step * self.right
                    elif key == KEY_LEFT:
                        delta -= self.step * self.right
                    elif key in ("w", KEY_PGUP):
                        delta += self.step * np.array([0.0, 0.0, 1.0])
                    elif key in ("s", KEY_PGDN):
                        delta -= self.step * np.array([0.0, 0.0, 1.0])
                    elif key == ".":
                        dyaw += YAW_STEP
                    elif key == ",":
                        dyaw -= YAW_STEP
                    elif key == "]":
                        self.step = float(np.clip(self.step * 2.0, STEP_MIN, STEP_MAX))
                        self.message = f"step {self.step * 1000:.1f} mm"
                    elif key == "[":
                        self.step = float(np.clip(self.step / 2.0, STEP_MIN, STEP_MAX))
                        self.message = f"step {self.step * 1000:.1f} mm"
                    elif key == " ":
                        self.record()
                    elif key == "u":
                        if self.points:
                            self.message = f"dropped {self.points.pop()['name']}"
                        else:
                            self.message = "no points to drop"
                    elif key == "l":
                        print()
                        for p in self.points:
                            print(f"  {p['name']}: {np.round(p['tcp_pose'], 5).tolist()}")
                        self.message = ""
                    elif key == "o":
                        if self.live:
                            actual = np.array(self.rtde_r.getActualTCPPose(), dtype=float)
                            self.position, self.yaw = actual[:3].copy(), yaw_of(actual)
                            self.message = f"re-levelled from {tilt_from_down_deg(actual):.2f} deg"
                        relevel = True
                    elif key == "h":
                        self.message = "returning to start pose"
                        self.status()
                        self.move_to(self.start_pose[:3], yaw_of(self.start_pose))
                    elif key in ("\r", "\n"):
                        self.save()

                if np.any(delta) or dyaw or relevel:
                    self.move_to(self.position + delta, self.yaw + dyaw)

                self.status()


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=ROBOT_IP, help="robot IP (default %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="read the robot but never command motion; the keys still work")
    ap.add_argument("--step", type=float, default=0.005, help="initial jog step [m]")
    ap.add_argument("--speed", type=float, default=0.05, help="moveL tool speed [m/s]")
    ap.add_argument("--accel", type=float, default=0.3, help="moveL tool accel [m/s^2]")
    ap.add_argument("--view-yaw", type=float, default=0.0,
                    help="rotate the arrow-key directions in the base XY plane [deg]; "
                         "0 means right arrow = base +X, up arrow = base +Y")
    ap.add_argument("--xlim", type=float, nargs=2, default=[-0.90, 0.90], metavar=("LO", "HI"))
    ap.add_argument("--ylim", type=float, nargs=2, default=[-0.90, 0.90], metavar=("LO", "HI"))
    ap.add_argument("--zlim", type=float, nargs=2, default=[0.02, 0.70], metavar=("LO", "HI"),
                    help="height bounds [m]; the lower one is what keeps the pusher off the table")
    ap.add_argument("--max-initial-tilt", type=float, default=25.0,
                    help="refuse to start if the tool is further than this off vertical [deg]")
    ap.add_argument("--out", default="calibration/push_t_calib.json",
                    help="where ENTER writes the recorded points")
    args = ap.parse_args()

    cal = Calibration(args)
    try:
        cal.run()
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        cal.close()
    if cal.points:
        print(f"{len(cal.points)} point(s) recorded; press ENTER inside the tool to save, "
              f"or re-run. Last save target: {args.out}")


if __name__ == "__main__":
    main()
