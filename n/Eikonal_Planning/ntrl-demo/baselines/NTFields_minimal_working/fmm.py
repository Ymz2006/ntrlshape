#!/usr/bin/env python3
"""Run a compact CPU FMM baseline for one or more bundled Aloha queries."""

import argparse
from pathlib import Path
from timeit import default_timer as timer

import igl
import numpy as np
import pykonal


ROOT = Path(__file__).resolve().parent


def build_speed(mesh_path: Path, resolution: int):
    vertices, faces = igl.read_triangle_mesh(str(mesh_path))
    lower = vertices.min(axis=0) - 0.1
    upper = vertices.max(axis=0) + 0.1
    spacing = (upper - lower) / (resolution - 1)
    axes = [lower[i] + spacing[i] * np.arange(resolution) for i in range(3)]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    distance = np.abs(igl.signed_distance(xyz.reshape(-1, 3), vertices, faces)[0])
    margin = 0.5
    speed = np.clip(distance, 0.01, margin) / margin
    return lower, spacing, speed.reshape((resolution,) * 3)


def solve_one(lower, spacing, speed, start, goal):
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = tuple(lower)
    solver.velocity.node_intervals = tuple(spacing)
    solver.velocity.npts = speed.shape
    solver.velocity.values = speed
    source_index = np.rint((start - lower) / spacing).astype(int)
    source_index = np.clip(source_index, 0, np.asarray(speed.shape) - 1)
    solver.traveltime.values[tuple(source_index)] = 0.0
    solver.unknown[tuple(source_index)] = False
    solver.trial.push(*source_index)
    begin = timer()
    solver.solve()
    path = solver.trace_ray(np.asarray(goal, dtype=np.float64))
    return path, timer() - begin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=ROOT / "data" / "Aloha" / "Aloha.obj")
    parser.add_argument("--start-goals", type=Path,
                        default=ROOT / "data" / "Aloha" / "valid_start_goal_large.npy")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "fmm_paths.npy")
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=41,
                        help="Grid cells per axis; increase for accuracy at higher cost")
    args = parser.parse_args()
    if args.resolution < 11:
        raise ValueError("--resolution must be at least 11")

    print(f"Building {args.resolution}^3 speed grid from {args.mesh} ...")
    lower, spacing, speed = build_speed(args.mesh, args.resolution)
    pairs = np.load(args.start_goals)
    paths, durations = [], []
    for index, pair in enumerate(pairs[:args.num_trajectories]):
        path, duration = solve_one(lower, spacing, speed, pair[:3], pair[3:])
        paths.append(path)
        durations.append(duration)
        print(f"trajectory {index}: {len(path)} points, {duration:.4f} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.array(paths, dtype=object), allow_pickle=True)
    np.save(args.output.with_name(args.output.stem + "_times.npy"), np.asarray(durations))
    print(f"Saved {len(paths)} trajectory/trajectories to {args.output}")


if __name__ == "__main__":
    main()
