import sys
sys.path.append('.')
import model2d as md
import torch
import os 
import numpy as np
import matplotlib.pylab as plt
import torch
from torch import Tensor
from torch.autograd import Variable, grad

import ezdxf
from visualfunctions import rotate_points, visual_training
from timeit import default_timer as timer
import math
import igl
from glob import glob

from parse_shape import dxf_to_shape, shape_to_points
from preprocess2d import sample_points


import random

from PIL import Image
import pytorch_kinematics as pk

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import torch_kdtree #import build_kd_tree
def rotate_points(points, x, y, theta):


    rot = torch.tensor([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    shape_points = torch.tensor(points, dtype=torch.float32)  # n x 2
    new_col = torch.ones((shape_points.shape[0], 1))
    shape_points = torch.cat([shape_points, new_col], dim=1)  # n x 3

    transformed = (rot @ shape_points.T).T  # n x 3

    # Return only x, y as list of tuples for Shapely
    return [tuple(p[:2].numpy()) for p in transformed]


def MPPI(womodel, XP):

    found_path = False
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
            found_path = True
            break

    point0.append(XP[:,3:6].clone())
    return point0, iter, found_path

modelPath = './Experiments/UR5'

meshname = 'Auburn'#
#meshname = 'Spotswood'
dataPath = './datasets/arm/'+ meshname
#dataPath = './datasets/new/'

womodel    = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0], device='cuda')
pt='./Experiments/Fshape_FmazeEasy/training_data2d_03_01_14_59/Model_Epoch_05000_ValLoss_2.126409e+00.pt'
print(pt)
womodel.load(pt)#
womodel.network.eval()

#dataPath = './datasets/Gib'
paths = dataPath




arr = np.load("testing_data2d/Fshape_FmazeEasy/sampled_points.npy")


test_list = []
for i in range (200):
    curr=torch.tensor(arr[i]).cuda()
    test_list.append(curr)
    

print (len(test_list))


BASE=torch.tensor([[0, 0, 0.0, 0.0,0.0, 0]]).cuda()


total = 0
for XP in test_list:

    #XP = start_goal
    XP = XP+BASE #Variable(Tensor(XP)).to('cuda').unsqueeze(0)


        
    start = timer()
    with torch.no_grad():
        point, iter , success= MPPI(womodel, XP.clone())

    end = timer()

    print('Time:', end-start)
    print (success)
    if (success):
        total += 1

print ("total:" + str(total))
