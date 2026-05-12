import sys
from os import path
from glob import glob
from model2d import Model
import time

modelPath = './Experiments/Fshape_FmazeEasy'
dataPath = './training_data2d/Fshape_FmazeEasy'
#dataPath = './datasets/gibson/Spotswood'



#model    = md.Model(modelPath, dataPath, 3, [0, 0.3,-0.03],device='cuda:0')
model    = Model(modelPath, dataPath, 3, [0,0,0],device='cuda:0')

start = time.time()
model.train()


end = time.time()
print("Training Time", end - start, "seconds")