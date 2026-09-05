#!/usr/bin/env python3
"""Minimal NTFields training entry point for the bundled Aloha example."""

import argparse
import random
from pathlib import Path

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
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Aloha")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "training")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--model-size", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--max-batches", type=int, default=6,
        help="batches per epoch; 0 consumes the complete dataloader (reproduction mode)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative")

    if not (args.data / "sampled_points.npy").is_file() or not (args.data / "speed.npy").is_file():
        raise FileNotFoundError(f"Training arrays are missing under {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Training on {device}; output: {args.output}")
    model = Model(str(args.output), str(args.data), 3, [0, 0.35], device=device,
                  model_size=args.model_size)
    model.Params["Training"]["Number of Epochs"] = args.epochs
    model.Params["Training"]["Batch Size"] = args.batch_size
    model.Params["Training"]["Save Every * Epoch"] = args.save_every
    model.Params["Training"]["Print Every * Epoch"] = args.print_every
    model.Params["Training"]["Max Batches Per Epoch"] = args.max_batches
    model.train()


if __name__ == "__main__":
    main()
