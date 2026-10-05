"""Train models/metric_bdiff (or the stock models/metric) on a 3-D shape dataset.

``models/metric_bdiff`` is a copy of ``models/metric`` whose ONLY difference is
how the Fourier matrix B is drawn:

    metric        B = trunc_normal_(mean=0, std=2, a=-2, b=2)   realized sigma ~1.08
    metric_bdiff  B = 0.2 * normal(0, 1)                        sigma exactly 0.20

(the ``metric_arm`` scheme).  B is a fixed, non-trainable tensor consumed as
``w = 2*pi*B`` in ``input_mapping``, so its scale is the Fourier bandwidth of
both routes; everything else -- network, loss, optimizer, schedule -- is
byte-identical between the two packages.

``--models`` selects the package so a run and its control go through exactly the
same code path.  Pair the two with the same ``--seed`` for a controlled
comparison::

    python train/train_3dshape_bdiff.py --dataPath ./datasets/3dshape/Ashape3d_env2 \
        --models metric_bdiff --seed 1 --device cuda:0

Unlike ``train/train_3dshape.py`` this names the experiment folder after the
dataset (the Model class names it after the data path's *parent*, i.e. always
"3dshape"), copies the final checkpoint to ``latest.pt`` so evaluation code has a
stable target, and writes ``run_config.json`` recording what was run.
"""

import sys
sys.path.append('.')

import argparse
import glob
import json
import os
import random
import shutil
import time
from datetime import datetime, timedelta

import numpy as np
import torch

parser = argparse.ArgumentParser(description='Train metric_bdiff / metric on a 3-D shape dataset.')
parser.add_argument('--dataPath', default='./datasets/3dshape/Ashape3d_env2',
                    help='Directory holding sampled_points/speed/normal/speed_angles/speed_dists .npy.')
parser.add_argument('--modelPath', default='./Experiments/3dshape_bdiff',
                    help='Root directory for the experiment folder.')
parser.add_argument('--models', default='metric_bdiff', choices=('metric_bdiff', 'metric'),
                    help='Model package to train with.')
parser.add_argument('--tag', default='', help='Suffix for the experiment folder name.')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs.')
parser.add_argument('--batch-size', type=int, default=None, help='Override batch size.')
parser.add_argument('--lr', type=float, default=None, help='Override learning rate.')
parser.add_argument('--seed', type=int, default=1,
                    help='Seed for B and the weight init; pair runs by seed to compare packages.')
args = parser.parse_args()

if args.models == 'metric_bdiff':
    from models.metric_bdiff import model_train_metric as md
else:
    from models.metric import model_train_metric as md

dataPath = args.dataPath.rstrip('/')

# source / goal configuration (x, y, z, rx, ry, rz); the rotation part is a
# rotation vector normalized by 2*pi, as written by preprocess_obj.py.
model = md.Model(args.modelPath, dataPath, 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=args.device)

# md.Model names the folder after the data path's *parent* ("3dshape"); name it
# after the dataset and package instead so runs stay distinct.
stamp = (datetime.utcnow() - timedelta(hours=5)).strftime("%m_%d_%H_%M")
name = '{}_{}_{}'.format(os.path.basename(dataPath), args.models, stamp)
if args.tag:
    name += '_' + args.tag
model.folder = os.path.join(args.modelPath, name)

if args.epochs is not None:
    model.Params['Training']['Number of Epochs'] = args.epochs
if args.batch_size is not None:
    model.Params['Training']['Batch Size'] = args.batch_size
if args.lr is not None:
    model.Params['Training']['Learning Rate'] = args.lr

# B is drawn inside Model.train(), so seed here, immediately before it.
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

print('data   : {}'.format(dataPath))
print('models : {}'.format(args.models))
print('seed   : {}'.format(args.seed))
print('output : {}'.format(model.folder))

start = time.time()
model.train()
elapsed = time.time() - start
print('Training time: {:.1f}s'.format(elapsed))

# Report the realized spread of the B actually used, so the log records the one
# thing that differs between the two packages.
B = model.B.detach().float()
b_std = float(B.std())

# Stable target for evaluation code, matching train_3dshape_arm_all.sh.
ckpts = sorted(glob.glob(os.path.join(model.folder, 'Model_Epoch_*.pt')))
latest = None
if ckpts:
    latest = os.path.join(model.folder, 'latest.pt')
    shutil.copy2(ckpts[-1], latest)

with open(os.path.join(model.folder, 'run_config.json'), 'w') as f:
    json.dump({
        'dataPath': dataPath,
        'models': args.models,
        'seed': args.seed,
        'device': args.device,
        'epochs': model.Params['Training']['Number of Epochs'],
        'batch_size': model.Params['Training']['Batch Size'],
        'learning_rate': model.Params['Training']['Learning Rate'],
        'B_std': b_std,
        'B_shape': list(B.shape),
        'train_time_s': elapsed,
        'final_checkpoint': os.path.basename(ckpts[-1]) if ckpts else None,
        'train_loss': model.total_train_loss,
    }, f, indent=2)

print('B std  : {:.4f}'.format(b_std))
print('latest : {}'.format(latest))
