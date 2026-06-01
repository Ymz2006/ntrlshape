"""Training entry point for the 2-D shape path-planning task.

Run ``dataprocessing/preprocess_dxf.py`` first to produce the .npy training
data, then start training with:

    python train/train_2dshape.py

The configuration space is SE(2): (x, y, theta), so dim = 3.  ``source`` is the
goal configuration used for the periodic travel-time plots written into the
experiment folder.
"""

import sys
sys.path.append('.')

import time
from models.metric import model_train_metric as md

modelPath = './Experiments/2dshape'
dataPath = './datasets/2dshape/Fmaze2_norm'

# source / goal configuration (x, y, theta) -- theta is stored normalized by 2*pi
model = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0], device='cuda:0')

start = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - start))
