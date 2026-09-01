"""Train vanilla NTFields on the ntrl-demo 3-D shape (SE(3)) datasets.

The datasets under ntrl-demo/datasets/3dshape already match the NTFields
loader contract -- sampled_points.npy is (N, 12) = [x0(6), x1(6)] and
speed.npy is (N, 2) = [speed0, speed1] -- so no preprocessing is needed.
Do NOT run dataprocessing/preprocess.py on them; it would overwrite both
files with a 3-D point-robot speed field.

The configuration space is SE(3): (x, y, z, rx, ry, rz), so dim = 6, with
the rotation vector stored normalized by 2*pi.  NTFields treats all six
coordinates as Euclidean (no wrap-around, fixed 1:1 trans/rot weighting),
which is the baseline approximation this script is here to measure.

    python train/train_3dshape.py --dataPath <dataset dir> --device cuda:0
"""

import sys
sys.path.append('.')

import argparse
import os

from models import model_3d as md

DEFAULT_DATA = ('/workspace/ntrl-demo/ntrl-demo/datasets/3dshape/Lshape3d_env1')

parser = argparse.ArgumentParser(description='Train NTFields on a 3dshape SE(3) dataset.')
parser.add_argument('--dataPath', default=DEFAULT_DATA,
                    help='Directory holding sampled_points.npy and speed.npy.')
parser.add_argument('--modelPath', default='./Experiments/3dshape',
                    help='Directory to write checkpoints into.')
parser.add_argument('--device', default='cuda:0',
                    help='Torch device to train on, e.g. cuda:0, cpu.')
args = parser.parse_args()

# Model.save() writes straight into modelPath and is called at epoch 1.
os.makedirs(args.modelPath, exist_ok=True)

# pos is only consumed by Model.plot(), which model_3d never calls; any
# two-element list is fine.
model = md.Model(args.modelPath, args.dataPath, 6, [0.0, 0.0], device=args.device)

model.train()
