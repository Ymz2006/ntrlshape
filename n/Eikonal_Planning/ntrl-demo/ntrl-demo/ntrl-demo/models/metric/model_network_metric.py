import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import LayerNorm, InstanceNorm1d
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_mlp as db
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

torch.backends.cudnn.benchmark = True

#3dnewserver
def sigmoid_out(input):
 
    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid_out(input) 

class NN(torch.nn.Module):
    
    def __init__(self, device, dim ,B):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 256

        # Split the configuration into a translational route (first half of the
        # dims: x,y,z) and a rotational route (last half: rx,ry,rz).  Each route
        # owns its own Fourier frequencies (a column-slice of B) and a completely
        # independent set of weights -- pe_gate, encoder, gate, encoder_norm.
        half = self.dim // 2
        self.B_trans = B[:, :half].T.to(device)   # (half, input_size)
        self.B_rot   = B[:, half:].T.to(device)   # (half, input_size)

        input_size = B.shape[0]

        self.scale = 10



        self.act = torch.nn.Softplus(beta=self.scale)
        self.actout = Sigmoid_out()
        self.nl1=2

        # Translational route (operates on the first `half` coordinates).
        self.encoder_t, self.gate_t, self.pe_gate_t, self.encoder_norm_t = self._build_route(h_size)
        # Rotational route (operates on the last `half` coordinates).
        self.encoder_r, self.gate_r, self.pe_gate_r, self.encoder_norm_r = self._build_route(h_size)



        self.fuse_len = 3
        self.fuse = torch.nn.ModuleList()
        self.fuse.append(Linear(2*h_size, h_size))   # 512 -> 256
        for i in range (self.fuse_len):
            self.fuse.append(Linear(h_size, h_size))     
            self.fuse.append(Linear(h_size, h_size))     

    def _build_route(self, h_size):
        """Build one independent embedding route (same architecture as the
        original single route).  Returns (encoder, gate, pe_gate, encoder_norm)."""
        encoder = torch.nn.ModuleList()
        encoder.append(Linear(252, h_size))
        for i in range(0, 3*self.nl1):
            encoder.append(Linear(h_size, h_size))
        encoder.append(Linear(h_size, h_size))

        gate = torch.nn.ModuleList()
        for i in range(self.nl1):
            gate.append(Linear(1, 1))

        pe_gate = torch.nn.ModuleList()
        pe_gate.append(Linear(h_size, h_size))
        pe_gate.append(Linear(h_size, h_size))

        encoder_norm = InstanceNorm1d(h_size)
        return encoder, gate, pe_gate, encoder_norm


    #'''
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            stdv = np.sqrt(2.0 / (m.weight.size(0)+m.weight.size(1)))
            torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stdv, a=-2.0*stdv, b=2.0*stdv)
            m.bias.data.fill_(0)
        
        for i in range(self.nl1):
            self.gate_t[i].weight.data.fill_(0)
            self.gate_t[i].bias.data.fill_(0)
            self.gate_r[i].weight.data.fill_(0)
            self.gate_r[i].bias.data.fill_(0)

        
   
    #'''
    def input_mapping(self, x, B):
        w = 2.*np.pi*B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)
    
    def lip_norm(self, w):
        absrowsum = torch.sqrt(torch.sum ( w**2 , dim =1)).detach()
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        return w * scale.unsqueeze(1) #[: , None ]
    

    def to_translational_embedding(self, x, B, pe_gate, gate, encoder, encoder_norm):
        x = self.input_mapping(x, B)

        w = self.lip_norm(pe_gate[0].weight)
        b = pe_gate[0].bias
        u = torch.sin(x@w.T+b)

        w = self.lip_norm(pe_gate[1].weight)
        b = pe_gate[1].bias
        v = torch.sin(x@w.T+b)

        for ii in range(0,self.nl1):
            x_tmp = x

            w = self.lip_norm(encoder[3*ii+1].weight)
            b = encoder[3*ii+1].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+2].weight)
            b = encoder[3*ii+2].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+3].weight)
            b = encoder[3*ii+3].bias
            y = x@w.T+b

            weight = torch.sigmoid(0.1*gate[ii].weight)
            x  = (1-weight)*x_tmp+(weight)*torch.sin(y)

        w = self.lip_norm(encoder[-1].weight)
        b = encoder[-1].bias

        y = x@w.T+b
        y = encoder_norm(y)

        return y


    def to_rotational_embedding(self, x, B, pe_gate, gate, encoder, encoder_norm):
        x = self.input_mapping(x, B)

        w = self.lip_norm(pe_gate[0].weight)
        b = pe_gate[0].bias
        u = torch.sin(x@w.T+b)

        w = self.lip_norm(pe_gate[1].weight)
        b = pe_gate[1].bias
        v = torch.sin(x@w.T+b)

        for ii in range(0,self.nl1):
            x_tmp = x

            w = self.lip_norm(encoder[3*ii+1].weight)
            b = encoder[3*ii+1].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+2].weight)
            b = encoder[3*ii+2].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+3].weight)
            b = encoder[3*ii+3].bias
            y = x@w.T+b

            weight = torch.sigmoid(0.1*gate[ii].weight)
            x  = (1-weight)*x_tmp+(weight)*torch.sin(y)

        w = self.lip_norm(encoder[-1].weight)
        b = encoder[-1].bias

        y = x@w.T+b
        y = encoder_norm(y)

        return y


    def out(self, coords):
        
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))


        half = 3
        x_trans = x[:, :half]
        x_rot   = x[:, half:]

        translational_embedding = self.to_translational_embedding(
            x_trans, self.B_trans, self.pe_gate_t, self.gate_t, self.encoder_t, self.encoder_norm_t)
        rotational_embedding = self.to_rotational_embedding(
            x_rot, self.B_rot, self.pe_gate_r, self.gate_r, self.encoder_r, self.encoder_norm_r)

        y = torch.cat([translational_embedding, rotational_embedding], dim=-1)


        residual_connection = y@self.fuse[0].weight.T + self.fuse[0].bias
        for i in range (self.fuse_len):
            y1 = residual_connection@self.fuse[2*i+1].weight.T + self.fuse[2*i+1].bias
            y1 = self.act(y1)
            y2 = y1@self.fuse[2*i+2].weight.T + self.fuse[2*i+2].bias
            
            residual_connection = residual_connection + y2
            residual_connection = self.act(residual_connection)


        # w = self.lip_norm(self.fuse[0].weight)
        # y = y@w.T + self.fuse[0].bias
        # w = self.lip_norm(self.fuse[1].weight)
        # y = y@w.T + self.fuse[1].bias



        x0 = residual_connection[:size,...]
        x1 = residual_connection[size:,...]

        #OURS
        x = torch.sqrt((x0-x1)**2+1e-6)
        x = x.view(x.shape[0],-1,16)
        x = (torch.logsumexp(10*x, 2)-np.log(16))/10
        x = 0.2*(torch.sum(x,dim=1,keepdim=True))
        
        #test
        #x = torch.exp(x)
        #L1
        # x = 0.01*torch.norm(x0-x1,p=1,dim=1).unsqueeze(1)

        
        
        return x, None, coords
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

        output, coords = self.out(coords)
        return output, coords
