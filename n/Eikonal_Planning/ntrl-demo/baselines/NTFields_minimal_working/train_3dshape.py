#!/usr/bin/env python3
"""NTFields training entry point for the 3-D shape (SE(3)) datasets.

The bundled Aloha example plans for a point in a 3-D workspace (dim = 3); the
3-D shape task plans a rigid body over SE(3) (dim = 6, configurations stored as
(x, y, z, rx, ry, rz) with the rotation vector normalized by 2*pi).  NTFields'
loss is isotropic and dimension-agnostic and consumes only
``sampled_points.npy`` (N, 2*dim) and ``speed.npy`` (N, 2) -- the extra
``normal.npy`` written by the ntrl-demo preprocessor is simply unused -- so the
same network trains on those datasets directly.

Two dataset-specific details differ from the Aloha defaults:

* ``--data-scale 1.0``: the shape datasets are already stored in [-0.5, 0.5],
  while the Aloha arrays are raw metres normalized by 10.
* ``--max-batches 5`` with ``--batch-size 2000``: matches the 5 x 2000 samples
  per epoch that models/metric and models/metric_arm consume, so a 4000-epoch
  run here sees the same number of samples as the ntrl-demo baselines.
* ``--repeat-ratio`` / ``--max-repeats``: NTFields replays an epoch whose mean
  loss exceeds 1.2x the previous epoch's, starting from a hardcoded
  ``prev_diff = 1.0``.  The 6-D shape data starts near 2.0, so epoch 1 is
  rejected forever; the relaxed ratio lets the first epochs through and the
  replay cap guarantees progress if a later epoch stalls.

    python train_3dshape.py --env rectangle_env1 --device cuda:0
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from models.model_3d import Model


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / ".." / ".." / "ntrl-demo" / "ntrl-demo" / "datasets" / "3dshape"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="rectangle_env1",
                        help="Dataset name under --data-root, e.g. Lshape3d_env2.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--data", type=Path, default=None,
                        help="Full dataset path; overrides --data-root/--env.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to outputs/3dshape/<env>.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--model-size", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-batches", type=int, default=5,
                        help="Batches per epoch; 0 consumes the complete dataloader.")
    parser.add_argument("--data-scale", type=float, default=1.0,
                        help="Divisor applied to the raw configurations.")
    parser.add_argument("--lr", type=float, default=None, help="Override the learning rate.")
    parser.add_argument("--repeat-ratio", type=float, default=2.5,
                        help="Reject an epoch whose mean loss exceeds this multiple of the "
                             "previous epoch's. NTFields ships 1.2 against a prev_diff that "
                             "starts at 1.0, which the 6-D shape data never clears at epoch 1.")
    parser.add_argument("--max-repeats", type=int, default=20,
                        help="Accept an epoch after this many replays regardless of the ratio; "
                             "0 replays without limit (the upstream behaviour).")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4,
                        help="Torch intra-op threads. An epoch here is 5 small batches, so the "
                             "run is launch-bound, not compute-bound; leaving torch's default "
                             "(one thread per core) makes concurrent runs thrash the CPU.")
    args = parser.parse_args()

    if args.max_batches < 0:
        parser.error("--max-batches must be non-negative")

    data = args.data if args.data is not None else args.data_root / args.env
    output = args.output if args.output is not None else ROOT / "outputs" / "3dshape" / args.env
    for name in ("sampled_points.npy", "speed.npy"):
        if not (data / name).is_file():
            raise FileNotFoundError("{} is missing under {}".format(name, data))
    output.mkdir(parents=True, exist_ok=True)

    points = np.load(str(data / "sampled_points.npy"), mmap_mode="r")
    dim = points.shape[1] // 2
    if points.shape[1] % 2:
        raise ValueError("sampled_points.npy has an odd width {}".format(points.shape[1]))

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("data   : {} (dim {})".format(data, dim))
    print("output : {}".format(output))
    print("device : {}".format(device))

    # ``pos`` is the source configuration used by the periodic travel-time plot.
    model = Model(str(output), str(data), dim, [0.0] * dim, device=device,
                  model_size=args.model_size)
    model.Params["Training"]["Number of Epochs"] = args.epochs
    model.Params["Training"]["Batch Size"] = args.batch_size
    model.Params["Training"]["Max Batches Per Epoch"] = args.max_batches
    model.Params["Training"]["Save Every * Epoch"] = args.save_every
    model.Params["Training"]["Print Every * Epoch"] = args.print_every
    model.Params["Training"]["Data Scale"] = args.data_scale
    model.Params["Training"]["Repeat Ratio"] = args.repeat_ratio
    model.Params["Training"]["Max Repeats Per Epoch"] = args.max_repeats
    if args.lr is not None:
        model.Params["Training"]["Learning Rate"] = args.lr
    start = time.time()
    model.train()
    print("Training time: {:.1f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
