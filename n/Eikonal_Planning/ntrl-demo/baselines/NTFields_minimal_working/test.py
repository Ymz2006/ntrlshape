#!/usr/bin/env python3
"""Generate NTFields trajectories with the bundled Aloha checkpoint."""

import argparse
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch

from models.model_3d import Model


ROOT = Path(__file__).resolve().parent


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "ntfields_aloha_model_size_2.pt")
    parser.add_argument("--start-goals", type=Path,
                        default=ROOT / "data" / "Aloha" / "valid_start_goal_large.npy")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ntfields_paths.npy")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--goal-tolerance", type=float, default=0.06)
    args = parser.parse_args()

    device = resolve_device(args.device)
    start_goals = np.load(args.start_goals).astype(np.float32)
    count = min(args.num_trajectories, len(start_goals))
    model = Model(str(args.checkpoint.parent), str(args.start_goals.parent), 3, [0, 0],
                  device=device, model_size=2)
    model.load(str(args.checkpoint))

    paths = []
    durations = []
    for index in range(count):
        state = torch.from_numpy(start_goals[index:index + 1] / 10.0).to(device)
        left = [state[:, :3].detach().clone()]
        right = [state[:, 3:].detach().clone()]
        start_time = timer()
        for _ in range(args.max_steps):
            if torch.linalg.vector_norm(state[:, 3:] - state[:, :3]).item() <= args.goal_tolerance:
                break
            state = (state + args.step_size * model.Gradient(state.clone())).detach()
            left.append(state[:, :3].clone())
            right.append(state[:, 3:].clone())
        durations.append(timer() - start_time)
        right.reverse()
        path = 10.0 * torch.cat(left + right).cpu().numpy()
        paths.append(path)
        print(f"trajectory {index}: {len(path)} points, {durations[-1]:.4f} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.array(paths, dtype=object), allow_pickle=True)
    np.save(args.output.with_name(args.output.stem + "_times.npy"), np.asarray(durations))
    print(f"Saved {len(paths)} trajectory/trajectories to {args.output}")


if __name__ == "__main__":
    main()
