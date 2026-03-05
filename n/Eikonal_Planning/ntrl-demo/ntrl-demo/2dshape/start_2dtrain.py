import sys
from os import path
from glob import glob
from model2d import Model


modelPath = './Experiments/Fshape_FmazeEasy'
dataPath = './training_data2d/Fshape_FmazeEasy_02'
#dataPath = './datasets/gibson/Spotswood'



#model    = md.Model(modelPath, dataPath, 3, [0, 0.3,-0.03],device='cuda:0')
model    = Model(modelPath, dataPath, 3, [-0.15, 0.1,0.1],device='cuda:0')

model.train()


