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

    def send_device(self,device):
        self.data    = self.data.to(device)


    def __getitem__(self, index):
        data = self.data[index]
        return data, index
    def __len__(self):
        return self.data.shape[0]

def Database(PATH):
    
    try:
        points = np.load('{}/sampled_points.npy'.format(PATH))
        speed = np.load('{}/speed.npy'.format(PATH))
        #print(tau.min())
    except ValueError:
        print('Please specify a correct source path, or create a dataset')
    rows=points.shape[0]
    
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed)
    #database = _numpy2dataset(XP,YP)
    return database





