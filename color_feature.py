from __future__ import division
__author__ = 'devilc'

from skimage import exposure
import skimage.color as color
import numpy as np

def cal_hist_feature(X_train, X_val):
    print('calculating hist_feature.py')
    hist_train = []
    hist_val = []
    for img in X_train:
        img =  color.rgb2gray(img)
        hist = exposure.histogram(img)
        hist_train.append(np.array(hist[0]))
    hist_train = np.array(hist_train)
    hist_train = (hist_train.astype('float')- 1250)/1250

    for img in X_val:
        img =  color.rgb2gray(img)
        hist = exposure.histogram(img)
        hist_val.append(np.array(hist[0]))
    hist_val = np.array(hist_val)
    hist_val = (hist_val.astype('float') - 1250)/1250

    np.save('./data/cad_hist_feature_train.npy', hist_train)
    np.save('./data/cad_hist_feature_val.npy', hist_val)
    return hist_train, hist_val



