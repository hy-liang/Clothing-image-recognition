__author__ = 'Administrator'
import os,sys
sys.path.append('../')
from load_data import Small_X_train
import numpy as np

var = np.var(Small_X_train)
print (var.shape)