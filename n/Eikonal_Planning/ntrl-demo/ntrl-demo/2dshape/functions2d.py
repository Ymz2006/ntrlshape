import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_mlp as db
#from models import model_network_one as model_network
import igl 
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

# Prior versions relied on ``torch_kdtree`` for nearest neighbour queries.
# The implementation no longer depends on that package.

torch.backends.cudnn.benchmark = True


class Function():
    def __init__(self, path, device, network, dim):

        # ======================= JSON Template =======================
        self.path = path
        self.device = device
        self.dim = dim

        self.network = network

        # Pass the JSON information
        #self.Params['Device'] = device

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        #input_file = "datasets/gibson/Cabin/mesh_z_up_scaled.off"
        #self.kdtree, self.v_obs, self.n_obs = self.pc_kdtree(input_file)

        self.alpha = 1.025
        limit = 0.5
        self.margin = limit/15.0
        self.offset = self.margin/10.0 
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def Loss(self, points, Yobs, beta):

        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]
        
        
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)

        Ypred0 = torch.sqrt(S0+1e-8)#torch.sqrt
        Ypred1 = torch.sqrt(S1+1e-8)#torch.sqrt


        Ypred0_visco = Ypred0
        Ypred1_visco = Ypred1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = (Yobs[:,0])#**2
        sq_Yobs1 = (Yobs[:,1])#**2

        #loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        #loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1
        l0 = ((sq_Yobs0*(sq_Ypred0)))
        l1 = ((sq_Yobs1*(sq_Ypred1)))
        

        count = torch.isnan(l1).sum().item()
        # print("l1 nan count:", count)


        l0_2 = (torch.sqrt(l0))#**(1/4)
        l1_2 = (torch.sqrt(l1))#**(1/4)    


        count = torch.isnan(l1_2).sum().item()
        # print("l1_2 nan count:", count)

        #w_num = w.clone().detach()
        loss_weight = 1e-2
        loss0 = loss_weight*(l0_2-1)**2  #/scale#+relu_loss0#**2#+gamma*lap0#**2
        loss1 = loss_weight*(l1_2-1)**2  #/scale#+relu_loss1#**2#+gamma*lap1#**2
        # print("=============losses======")
        # print (loss0)
        # print(loss1)
        diff = loss0 + loss1 
        # print("=============diff======")
        # print(diff)



        td_weight = 5e-3
        with torch.no_grad():

            length0 = (0.02)/(Yobs[:,0]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir0 = length0*(DT0*Yobs[:,0].unsqueeze(1)**2).clone().detach()  
            #Dir1 = 0.03*(DT1/S1.unsqueeze(1)).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length0#*1/Yobs[:,0].unsqueeze(1)
            del Xp_new0, Dir0#Xp_new1

        tau_loss0 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        #(1.01-Yobs[:,0])*td_weight*
        #(1.4-Yobs[:,0])*

        with torch.no_grad():

            length1 = (0.02)/(Yobs[:,1]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir1 = length1*(DT1*Yobs[:,1].unsqueeze(1)**2).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            #Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0
            Xp_new0[:,self.dim:] = Xp_new0[:,self.dim:] - Dir1
            #Xp_new[:,self.dim:]+=0.04*DT1/S1

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length1#*1/Yobs[:,1].unsqueeze(1)
            del Xp_new0, Dir1#Xp_new1

        tau_loss1 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        
        where_d0 = (tau[:,0] < length0.squeeze())
        where_d1 = (tau[:,0] < length1.squeeze())
        tau_loss0[where_d0] = 0 
        tau_loss1[where_d1] = 0 

        tau_loss = tau_loss0+tau_loss1

        T = tau[:,0] #* torch.sqrt(T0)
        #loss_n =(torch.sum((diff +tau_loss)*torch.exp(-0.5*T)))/Yobs.shape[0]#*torch.exp(-para*T)

        loss_n = torch.sum(diff) + torch.sum(tau_loss)
        loss = loss_n * beta
        return  loss, loss_n

    def TravelTimes(self, Xp):
     
        tau, w, coords = self.network.out(Xp)        

        TT = tau[:,0] #* torch.sqrt(T0)
            
        return TT

    def Speed(self, Xp):

   

        Xp = Xp.to(torch.device(self.device))

        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)
        
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.einsum('ij,ij->i', D, D)

        #DT0 = dtau[:,self.dim:]
        DT0 = dtau[:,:self.dim]

        #T3    = tau[:,0]**2



        #TT = LogTau * torch.sqrt(T0)

        #print(tau.shape)

        #print(T02.shape)
        #T1    = T0*torch.einsum('ij,ij->i', DT0, DT0)
        #T2    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
        
        
        S = torch.einsum('ij,ij->i', DT0, DT0)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau#, T0#, T1, T2, T3
        return Ypred
    

