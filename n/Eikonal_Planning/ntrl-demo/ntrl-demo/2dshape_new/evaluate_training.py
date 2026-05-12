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


def MPPI(womodel, XP, dim):

    found_path = False
    steps = 200
    sample_num = 50
    horizon = 5

    dP_prior = torch.zeros((1,dim)).cuda()

    point0=[]
    point0.append(XP[:,0:dim].clone())

    for iter in range(steps):
        #print(iter)
        XP_tmp = XP.clone()#[:,0:3]
        #print(XP_tmp)
        XP_tmp = XP_tmp.unsqueeze(0).repeat(sample_num,horizon,1)

        dP = 0.015 * torch.normal(0,1,size=(sample_num, 1, dim),dtype=torch.float32, device='cuda') \
            +0.015 * torch.normal(0,1,size=(sample_num, horizon, dim),dtype=torch.float32, device='cuda')
        #dP = 0.02 * torch.nn.functional.normalize(dP + dP_prior,dim=2)
        dP = dP + 2*dP_prior
        dP_norm = torch.norm(dP,dim=2,keepdim=True)
        dP = dP/(torch.clamp(dP_norm,min=0.015)/0.015)
        #print(dP)
        dP_cumsum = torch.cumsum(dP, dim=1)
        #print(XP_tmp.shape)
        XP_tmp[...,0:dim] = XP_tmp[...,0:dim]+dP_cumsum
        
        indices = [0, -1]

        cost = womodel.function.TravelTimes(XP_tmp[:,indices,:].reshape(-1,dim*2))
        
        cost = cost.reshape(-1,2)
        cost = 10*cost[:,0] + cost[:,1]#torch.sum(cost.reshape(-1,2),dim=1)#
        
        weight = torch.softmax(-50*cost, dim=0)
        
        dP_prior = (weight@dP[:,0,:]) 

        XP[:,0:dim] = dP_prior + XP[:,0:dim]

        #print(XP.shape)
        dis=torch.norm(XP[:,dim:dim*2]-XP[:,0:dim])
        #print(XP)
        point0.append(XP[:,0:dim].clone())
        
        if(dis<0.01):
            found_path = True
            break

    point0.append(XP[:,dim:dim*2].clone())
    return point0, iter, found_path

modelPath = './Experiments/UR5'

meshname = 'Auburn'#
#meshname = 'Spotswood'
dataPath = './datasets/arm/'+ meshname
#dataPath = './datasets/new/'

womodel    = md.Model(modelPath, dataPath, 3, [0.0, 0.0, 0.0,0.0], device='cuda')
pt = './Experiments/latest/Model_latest.pt'
print(pt)
womodel.load(pt)
womodel.network.eval()

#dataPath = './datasets/Gib'
paths = dataPath



arr = np.load("./testing_data2d/Fshape_FmazeEasy/sampled_points.npy")
arr_speeds = np.load("./testing_data2d/Fshape_FmazeEasy/speed.npy")
environment_boundary_points = np.load('./testing_data2d/Fshape_FmazeEasy/Fmaze3env.npy')
Fshape_norm = dxf_to_shape("./datasets/Fshape_norm.dxf")
Fshape_points = shape_to_points(Fshape_norm)

test_list = []
test_list_speed = []

#test_list.append(torch.tensor((-0.3,-0.03,0.25,   0.4,0.23,0.25)).cuda())
#test_list.append(torch.tensor((-0.3,-0.03,0.25,   0,-0.03,0.25, )).cuda())
for i in range (100):
    curr=torch.tensor(arr[i]).cuda()
    test_list.append(curr)
    test_list_speed.append(min(arr_speeds[i]))

# print (len(test_list))


BASE=torch.tensor([[0, 0, 0,0,0, 0]]).cuda()


total = 0

success_list = []
fail_list = []
test_list_speed_failed = []

cnt2=  0
for XP in test_list:

    print(XP)
    # diff = [
    #     (XP[4]-XP[0]).item(),
    #     (XP[5]-XP[1]).item(),
    #     (math.atan2(XP[6],XP[7]) - math.atan2(XP[2],XP[3]))
    # ]
    diff = [
        (XP[3]-XP[0]).item(),
        (XP[4]-XP[1]).item(),
        (XP[5]-XP[3]).item()
    ]
    #XP = start_goal
    XP = XP+BASE #Variable(Tensor(XP)).to('cuda').unsqueeze(0)
    

        
    start = timer()
    with torch.no_grad():
        point, iter , success= MPPI(womodel, XP.clone(),dim=3)

    end = timer()


    cnt =0 
    start_list = np.zeros((len(point), 3))

    # point = [[XP[0:3], XP[3:6]]]

    # start_list[0,0] = XP[0]
    # start_list[0,1] = XP[1]
    # start_list[0,2] = XP[2]

    # start_list[1,0] = XP[3]
    # start_list[1,1] = XP[4]
    # start_list[1,2] = XP[5]

    for p in point:
        start_list[cnt, 0] = p[0][0].item()
        start_list[cnt, 1] = p[0][1].item()
        start_list[cnt, 2] = p[0][2].item()*2*np.pi
        cnt+=1
    
    print(XP.shape)

    begin_point = [XP[0][0].item(), XP[0][1].item(), XP[0][2].item()*2*np.pi]
    end_point = [XP[0][3].item(), XP[0][4].item(), XP[0][5].item()*2*np.pi]
    start_list[0,0] = XP[0][0].item()
    start_list[0,1] = XP[0][1].item()
    start_list[0,2] = XP[0][2].item()*2*np.pi

    # start_list[1,0] = XP[0][3].item()
    # start_list[1,1] = XP[0][4].item()
    # start_list[1,2] = XP[0][5].item()*2*np.pi

    speed_temp = np.full((len(point), 1), 0.5)
    visual_training(start=start_list,shape_points=Fshape_points,env_points=environment_boundary_points, 
                    cnt=len(point),speed=speed_temp, vmin = 0.2, begin_point=begin_point, end_point=end_point)
    # print('Time:', end-start)

    
    if (success):
        success_list.append(diff)
        print("success")
        total += 1
    else:
        fail_list.append(diff)
        print("fail")
        test_list_speed_failed.append(test_list_speed[cnt2])

    cnt2 +=1



print ("total:" + str(total))

success_list = np.array(success_list)
fail_list = np.array(fail_list)

x = success_list[:,0]
y = success_list[:,1]
z = success_list[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)

plt.show()




x = fail_list[:,0]
y = fail_list[:,1]
z = fail_list[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)

plt.show()



plt.hist(test_list_speed_failed, bins=10, color='skyblue', edgecolor='black', alpha=0.7)

# Labeling the plot
plt.title('Distribution of Speed Failed')
plt.xlabel('Speed Value')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save or display
plt.show()