import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad
#import open3d as o3d

class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, points, speed):
        # Creating identical pairs
        points    = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        print(points.shape)
        print(speed.shape)

        self.data=torch.cat((points,speed),dim=1)
        #self.grid  = Variable(Tensor(grid))

    def send_device(self,device):
        self.data    = self.data.to(device)

    def __getitem__(self, index):
        data = self.data[index]
        #print(index)
        return data, index
    def __len__(self):
        return self.data.shape[0]

def Database(PATH):
    
    points = np.load('{}/sampled_points.npy'.format(PATH))#[:100000,:]
    speed = np.load('{}/speed.npy'.format(PATH))#[:100000,:]

    print("DATABASE INIT")
    print(points.shape)

    print(points[0:100,:])



    print(speed.shape)

    print(speed[0:100,:])


    print("min speed: " + str(speed.min()))
    print("max speed: " + str(speed.max()))
    
    count = np.isnan(points).sum()
    print(count)
    count = np.isnan(speed).sum()
    print(count)
    print("DATABASE DONE INIT")


    #print(np.shape(grid))
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed)
    #database = _numpy2dataset(XP,YP)
    return database





