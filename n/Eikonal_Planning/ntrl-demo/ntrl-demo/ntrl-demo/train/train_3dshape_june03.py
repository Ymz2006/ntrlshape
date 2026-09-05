"""Train the JUNE-3 pipeline: frozen models/metric_june03 on frozen-generator data.

Reproduces the run behind pretrained/baseline_rectangle_env1.pt so the old
pipeline can be measured against the current one on equal footing.  Pair it with
data from dataprocessing/preprocess_obj_june03.py; the checkpoint's own metadata
says 5000 epochs (24,940 optimizer steps at 5 batches/epoch) and its Fourier B
matrix has std 0.86, matching this trainer's truncated-normal init.

    python train/train_3dshape_june03.py \
        --dataPath ./datasets/3dshape/rectangle_env1_june03 --epochs 5000
"""

import sys
sys.path.append('.')

import time
import argparse

from models.metric_june03 import model_train_metric as md

parser = argparse.ArgumentParser(description='Train the June-3 3-D shape pipeline.')
parser.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1_june03')
parser.add_argument('--modelPath', default='./Experiments/3dshape_june03')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--epochs', type=int, default=5000)
args = parser.parse_args()

model = md.Model(args.modelPath, args.dataPath, 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 device=args.device)
model.Params['Training']['Number of Epochs'] = args.epochs

print('data   : {}'.format(args.dataPath))
print('lr     : {}'.format(model.Params['Training']['Learning Rate']))
start = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - start))
