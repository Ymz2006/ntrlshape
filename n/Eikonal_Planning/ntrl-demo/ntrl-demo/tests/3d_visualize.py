import sys
sys.path.append('.')
from models.metric_arm import model_test_metric as md
import torch
import os 
import numpy as np
import matplotlib.pylab as plt
import torch
from torch import Tensor
from torch.autograd import Variable, grad


from timeit import default_timer as timer
import math
import igl
from glob import glob
import random

from PIL import Image
import pytorch_kinematics as pk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import torch_kdtree #import build_kd_tree

def MPPI(womodel, XP):
    steps = 200
    sample_num = 50
    horizon = 5

    dP_prior = torch.zeros((1,3)).cuda()

    point0=[]
    point0.append(XP[:,0:3].clone())

    for iter in range(steps):
        #print(iter)
        XP_tmp = XP.clone()#[:,0:3]
        #print(XP_tmp)
        XP_tmp = XP_tmp.unsqueeze(0).repeat(sample_num,horizon,1)
        dP_list = []
        cost_path = 0
        XP_list = []
        #dP_list = []
        dP = 0.015 * torch.normal(0,1,size=(sample_num, 1, 3),dtype=torch.float32, device='cuda') \
            +0.015 * torch.normal(0,1,size=(sample_num, horizon, 3),dtype=torch.float32, device='cuda')
        #dP = 0.02 * torch.nn.functional.normalize(dP + dP_prior,dim=2)
        dP = dP + 2*dP_prior
        dP_norm = torch.norm(dP,dim=2,keepdim=True)
        dP = dP/(torch.clamp(dP_norm,min=0.015)/0.015)
        #print(dP)
        dP_cumsum = torch.cumsum(dP, dim=1)
        #print(XP_tmp.shape)
        XP_tmp[...,0:3] = XP_tmp[...,0:3]+dP_cumsum
        
        indices = [0, -1]

        cost = womodel.function.TravelTimes(XP_tmp[:,indices,:].reshape(-1,6))
        
        cost = cost.reshape(-1,2)
        cost = 10*cost[:,0] + cost[:,1]#torch.sum(cost.reshape(-1,2),dim=1)#
        
        weight = torch.softmax(-50*cost, dim=0)
        
        dP_prior = (weight@dP[:,0,:]) 

        XP[:,0:3] = dP_prior + XP[:,0:3]

        #print(XP.shape)
        dis=torch.norm(XP[:,3:6]-XP[:,0:3])
        #print(XP)
        point0.append(XP[:,0:3].clone())
        
        if(dis<0.01):
            print("path found")
            break

    point0.append(XP[:,3:6].clone())
    return point0, iter

modelPath = './Experiments/UR5'
meshname = 'Auburn'#
#meshname = 'Spotswood'
dataPath = './datasets/arm/'+ meshname
#dataPath = './datasets/new/'

womodel    = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0], device='cuda')
pt='./Experiments/Gib/gibson_12_30_22_05/Model_Epoch_00200_ValLoss_3.309497e-03.pt'
print(pt)
womodel.load(pt)#
womodel.network.eval()

#dataPath = './datasets/Gib'
paths = dataPath

XP=torch.tensor([[0,-0.4,-0.05, 0,0.35,-0.05]]).cuda()

BASE=torch.tensor([[0, 0, 0.0, 0.0,0.0, 0]]).cuda()
#XP = start_goal
XP = XP+BASE #Variable(Tensor(XP)).to('cuda').unsqueeze(0)

for ii in range(5):
    
    start = timer()
    with torch.no_grad():
        point, iter = MPPI(womodel, XP.clone())

    end = timer()

    print('Time:', end-start)


if iter == 200:
    print('Failed')
    
    #continue

# query_points = torch.cat(point).to('cpu').data.numpy()#np.asarray(point)

cnt = 0
for p in point:
    #cnt+=1
    print(f"[{p[0][0].item()}, {p[0][1].item()}, {p[0][2].item()}],")


points_cpu = torch.cat([p.cpu() for p in point], dim=0)  # Extract x, y, z

# Extract x, y, z coordinates
x = points_cpu[:, 0].numpy()
y = points_cpu[:, 1].numpy()
z = points_cpu[:, 2].numpy()

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points Plot')

# Save the figure
plt.savefig('3d_points_plot.png', dpi=300)
plt.show()
