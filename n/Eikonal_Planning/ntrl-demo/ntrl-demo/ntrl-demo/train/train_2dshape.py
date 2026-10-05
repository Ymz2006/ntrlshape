"""Training entry point for the 2-D shape path-planning task.

Run ``dataprocessing/preprocess_dxf.py`` first to produce the .npy training
data, then start training with:

    python train/train_2dshape.py

The configuration space is SE(2): (x, y, theta), so dim = 3.  ``source`` is the
goal configuration used for the periodic travel-time plots written into the
experiment folder.

Experiment tracking (Weights & Biases) is on by default; configure it with the
flags from ``train/wandb_utils.py``, e.g.::

    python train/train_2dshape.py --project ntrl-demo --tags Fmaze2 baseline \
        --notes "lr sweep" --epochs 5000
    python train/train_2dshape.py --no-wandb            # disable tracking
"""

import sys
sys.path.append('.')

import time
import argparse

from models.metric_2dshape import model_train_metric as md
from train.wandb_utils import add_wandb_args, apply_overrides, start_run, finish_run

modelPath = './Experiments/2dshape'
dataPath = './datasets/2dshape/Fmaze2_norm'

parser = argparse.ArgumentParser(description='Train the 2-D shape planner.')
parser.add_argument('--dataPath', default=None,
                    help='Directory holding the .npy training data. Overrides the '
                         'default dataset path (alias of --data).')
add_wandb_args(parser)
args = parser.parse_args()

# --dataPath / --data override the dataset path; resolve BEFORE building the Model
# since the experiment folder name is derived from it.
dataPath = args.dataPath
# source / goal configuration (x, y, theta) -- theta is stored normalized by 2*pi
model = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0], device='cuda:0')

apply_overrides(model, args)
start_run(args, model, task='2dshape')

start = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - start))

finish_run(args)
