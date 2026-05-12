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
pt='./Experiments/Fshape_FmazeEasy/training_data2d_02_23_10_41/Model_Epoch_05000_ValLoss_1.852163e+00.pt'
print(pt)
womodel.load(pt)#
womodel.network.eval()

#dataPath = './datasets/Gib'
paths = dataPath






XP1=torch.tensor([[-0.4,-0.3,0, 0.2,0.2,np.pi/2]]).cuda()

XP2=torch.tensor([[-0.4,-0.3,0, -0.4,0.1,0]]).cuda()

XP3=torch.tensor([[-0.4,-0.3,0, -0.25,0.1,np.pi/4]]).cuda()

XP4=torch.tensor([[-0.4,-0.3,0, -0.2,0,0]]).cuda()

XP5=torch.tensor([[-0.4,-0.3,0, 0.3,0.07,1]]).cuda()

XP6=torch.tensor([[0,-0.1,0, 0,0.1,1]]).cuda()


test_list = [XP6]

BASE=torch.tensor([[0, 0, 0.0, 0.0,0.0, 0]]).cuda()



for XP in test_list:

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



    Fshape_norm = dxf_to_shape("./datasets/Fshape_norm.dxf")
    Fshape_points = shape_to_points(Fshape_norm)

    environment_boundary_points = np.load('./training_data2d/Fshape_FmazeEasy/Easyenv.npy')

    speed_temp = np.full((len(point), 1), 0.5)



    cnt =0 
    start_list = np.zeros((len(point), 3))
    for p in point:

        print(f"[{p[0][0].item()}, {p[0][1].item()}, {p[0][2].item()}],")
        start_list[cnt, 0] = p[0][0].item()
        start_list[cnt, 1] = p[0][1].item()
        start_list[cnt, 2] = p[0][2].item()
        cnt+=1

    print(type(environment_boundary_points))
    print(environment_boundary_points.shape)
    visual_training(start=start_list,shape_points=Fshape_points,env_points=environment_boundary_points, 
                    cnt=len(point),speed=speed_temp, vmin = 0.2)


    points_cpu = torch.cat([p.cpu() for p in point], dim=0)  # Extract x, y, z

