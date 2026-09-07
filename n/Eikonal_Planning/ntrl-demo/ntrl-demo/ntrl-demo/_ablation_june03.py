"""Ablate the June-3 trainer against the current one on the SAME dataset.

Isolates the two candidate causes of the pretrained checkpoint's low planning
success: the Fourier-feature scale B and the learning rate.  Nothing in
models/metric_june03 is edited -- B and the LR are overridden on the constructed
Model before train() is called.
"""
import sys
sys.path.append('.')

import argparse
import time
import torch

p = argparse.ArgumentParser()
p.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1_june03')
p.add_argument('--modelPath', default='./Experiments/june03_ablation')
p.add_argument('--name', required=True, help='experiment folder name')
p.add_argument('--trainer', choices=('june03', 'june03_nonormal', 'arm'), default='june03')
p.add_argument('--b', choices=('june', 'arm'), default='june',
               help="june = trunc_normal(std=1) (~0.86); arm = 0.2*normal(0,1)")
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--epochs', type=int, default=5000)
p.add_argument('--device', default='cuda:0')
args = p.parse_args()

if args.trainer == 'june03':
    from models.metric_june03 import model_train_metric as md
    from models.metric_june03 import model_network_metric as mn
elif args.trainer == 'june03_nonormal':
    from models.metric_june03_nonormal import model_train_metric as md
    from models.metric_june03_nonormal import model_network_metric as mn
else:
    sys.path.insert(0, '../../baselines/ntrl-demo')
    from models.metric_arm import model_train_metric as md
    from models.metric_arm import model_network_metric as mn

model = md.Model(args.modelPath, args.dataPath, 6, [0.0] * 6, device=args.device)
model.folder = args.modelPath + '/' + args.name

# ── Override the Fourier feature matrix and rebuild the network ──
B = torch.normal(0, 1, size=(128, 6))
if args.b == 'june':
    torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
else:
    B = 0.2 * B
model.B = B
model.network = mn.NN(model.Params['Device'], 6, model.B)
model.network.apply(model.network.init_weights)
model.network.to(model.Params['Device'])

model.Params['Training']['Learning Rate'] = args.lr
model.Params['Training']['Number of Epochs'] = args.epochs

import os
os.makedirs(model.folder, exist_ok=True)
print('name    : {}'.format(args.name))
print('trainer : {}'.format(args.trainer))
print('data    : {}'.format(args.dataPath))
print('B std   : {:.4f}  ({})'.format(float(model.B.std()), args.b))
print('lr      : {}'.format(model.Params['Training']['Learning Rate']))
print('epochs  : {}'.format(args.epochs))
print('folder  : {}'.format(model.folder))

t0 = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - t0))
