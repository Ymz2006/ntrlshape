import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad



class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, points, speed):
        # Creating identical pairs
        points    = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        self.data=torch.cat((points,speed),dim=1)
        # self.grid  = Variable(Tensor(grid))

    def send_device(self,device):
        self.data    = self.data.to(device)
        # self.grid    = self.grid.to(device)


    def __getitem__(self, index):
        data = self.data[index]
        return data, index
    def __len__(self):
        return self.data.shape[0]

def Database(PATH, scale=10.0):
    """Load NTFields training arrays.

    ``scale`` divides the raw configurations; the bundled Aloha arrays are in
    metres over roughly +/-5 and the default 10.0 maps them into [-0.5, 0.5].
    Datasets already stored in that range (e.g. the 3-D shape SE(3) sets) must
    pass ``scale=1.0``.
    """
    
    try:
        points = np.load('{}/sampled_points.npy'.format(PATH))
        speed = np.load('{}/speed.npy'.format(PATH))
        # occupancies = np.unpackbits(np.load('{}/voxelized_point_cloud_128res_20000points.npz'.format(PATH))['compressed_occupancies'])
        # input = np.reshape(occupancies, (128,)*3)
        # grid = np.array(input, dtype=np.float32)
        #print(tau.min())
    except ValueError:
        print('Please specify a correct source path, or create a dataset')
    rows=points.shape[0]
    
    print(points.shape,speed.shape)
    points = points / scale
    # print(np.shape(grid))
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed)
    #database = _numpy2dataset(XP,YP)
    return database





