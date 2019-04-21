import sys
import skimage.feature as ft
from skimage import io
import skimage.color as color
import numpy as np
import os
from sklearn import preprocessing

'''''
for c in range(8):
    path  = './data/small_category_val/'+category_name[c]
    for dirpath,dirnames,filenames in os.walk(path):
        for file in filenames:
            fullpath=os.path.join(dirpath, file)
            img = io.imread(fullpath)
            GrayImage = color.rgb2gray(img)
            feature = ft.hog(GrayImage, orientations=8, pixels_per_cell=(10, 10), transform_sqrt=True, feature_vector=True)
            hog_feature.append(feature)
            print (fullpath)

np.save('./data/Small_X_val_hog_feature.npy', np.array(hog_feature))
'''''

def cal_hog_feature(X_train, X_val):
    hog_feature_train = []
    hog_feature_val = []
    print('extracting train set hog feature!')
    for img in X_train:
        print img
        GrayImage = color.rgb2gray(img)
        feature = ft.hog(GrayImage, orientations=8, pixels_per_cell=(10, 10), transform_sqrt=True, feature_vector=True)
        hog_feature_train.append(feature)

    print('extracting val set hog feature!')
    for img in X_val:
        GrayImage = color.rgb2gray(img)
        feature = ft.hog(GrayImage, orientations=8, pixels_per_cell=(10, 10), transform_sqrt=True, feature_vector=True)
        hog_feature_val.append(feature)

    hog_feature_train = np.array(hog_feature_train)
    hog_feature_val = np.array(hog_feature_val)
    hog_feature_train = preprocessing.scale(hog_feature_train)
    hog_feature_val = preprocessing.scale(hog_feature_val)
    np.save('./data/cad_hog_feature_train.npy', hog_feature_train)
    np.save('./data/cad_hog_feature_val.npy', hog_feature_val)
    return hog_feature_train, hog_feature_val


