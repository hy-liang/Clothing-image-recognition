__author__ = 'Administrator'

from load_data import hog_X_train,hog_X_val,cnn_X_train,cnn_X_val,hist_X_train,hist_X_val,Small_Y_train,Small_Y_val
import hpelm
import numpy as np
import os
from sklearn import preprocessing
import datetime

def combine_feature(feature1, feature2):
    combine = []
    feature1 = preprocessing.scale(feature1)
    feature2 = preprocessing.scale(feature2)
    for f1, f2 in zip(feature1, feature2):
        f=[]
        f.extend(f1)
        f.extend(f2)
        combine.append(f)
    return np.array(combine)

def combine_3_feature(feature1, feature2, feature3):
    combine = []
    feature1 = preprocessing.scale(feature1)
    feature2 = preprocessing.scale(feature2)
    feature3 = preprocessing.scale(feature3)
    for f1, f2, f3 in zip(feature1, feature2,feature3):
        f=[]
        f.extend(f1)
        f.extend(f2)
        f.extend(f3)
        combine.append(f)
    return np.array(combine)

def cal_error(predict, target):
    difference = (predict-target)*(predict-target)
    error = np.sum(difference)/(predict.shape[0]*predict.shape[1])
    print('mean ',np.mean(np.abs(target)))
    return error

hidden_num = 1024


def late_fusion_cnn_hog_hist():
    print ('1')
    cnn_hog_hist_train = combine_3_feature(cnn_X_train, hog_X_train, hist_X_train)
    cnn_hog_hist_val = combine_3_feature(cnn_X_val, hog_X_val, hist_X_val)
    cnn_hog_hist_size = cnn_hog_hist_train.shape[1]
    autoencoder = hpelm.ELM(inputs = cnn_hog_hist_size, outputs = cnn_hog_hist_size)
    autoencoder.add_neurons(hidden_num, 'sigm')
    print ('start')
    starttime = datetime.datetime.now()
    autoencoder.train(cnn_hog_hist_train, cnn_hog_hist_train)
    endtime = datetime.datetime.now()
    print('time:' + str((endtime - starttime).seconds))


    late_fusion_cnn_hog_hist_train = autoencoder.project(cnn_hog_hist_train)
    late_fusion_cnn_hog_hist_val = autoencoder.project(cnn_hog_hist_val)

    predict = autoencoder.predict(cnn_hog_hist_train)
    print('cnn_hog error '+str(hidden_num)+':'+str(cal_error(predict, cnn_hog_hist_train)))
    np.save('./data/late_fusion_cnn_hog_hist_train.npy', late_fusion_cnn_hog_hist_train)
    np.save('./data/late_fusion_cnn_hog_hist_val.npy', late_fusion_cnn_hog_hist_val)

def late_fusion_cnn_hog():
    cnn_hog_train = combine_feature(cnn_X_train, hog_X_train)
    cnn_hog_val = combine_feature(cnn_X_val, hog_X_val)
    cnn_hog_size = cnn_hog_train.shape[1]
    autoencoder = hpelm.ELM(inputs = cnn_hog_size, outputs=cnn_hog_size)
    autoencoder.add_neurons(hidden_num, 'sigm')
    autoencoder.train(cnn_hog_train, cnn_hog_train)

    late_fusion_cnn_hog_train = autoencoder.project(cnn_hog_train)
    late_fusion_cnn_hog_val = autoencoder.project(cnn_hog_val)

    predict = autoencoder.predict(cnn_hog_train)
    print('cnn_hog error '+str(hidden_num)+':'+str(cal_error(predict, cnn_hog_train)))
    #np.save('./data/late_fusion_cnn_hog_train.npy', late_fusion_cnn_hog_train)
    #np.save('./data/late_fusion_cnn_hog_val.npy', late_fusion_cnn_hog_val)

def late_fusion_hog_hist():
    hog_hist_train = combine_feature(hog_X_train, hist_X_train)
    hog_hist_val = combine_feature(hog_X_val, hist_X_val)
    hog_hist_size = hog_hist_train.shape[1]
    autoencoder = hpelm.ELM(inputs = hog_hist_size, outputs=hog_hist_size)
    autoencoder.add_neurons(hidden_num, 'sigm')
    autoencoder.train(hog_hist_train, hog_hist_train)

    late_fusion_hog_hist_train = autoencoder.project(hog_hist_train)
    late_fusion_hog_hist_val = autoencoder.project(hog_hist_val)

    predict = autoencoder.predict(hog_hist_train)
    print('hog_hist error '+str(hidden_num)+':'+str(cal_error(predict, hog_hist_train)))

    #np.save('./data/late_fusion_hog_hist_train.npy', late_fusion_hog_hist_train)
    #np.save('./data/late_fusion_hog_hist_val.npy', late_fusion_hog_hist_val)

def late_fusion_cnn_hist():
    cnn_hist_train = combine_feature(cnn_X_train, hist_X_train)
    cnn_hist_val = combine_feature(cnn_X_val, hist_X_val)
    cnn_hist_size = cnn_hist_train.shape[1]
    autoencoder = hpelm.ELM(inputs = cnn_hist_size, outputs=cnn_hist_size)
    autoencoder.add_neurons(hidden_num, 'sigm')
    autoencoder.train(cnn_hist_train, cnn_hist_train)

    late_fusion_cnn_hist_train = autoencoder.project(cnn_hist_train)
    late_fusion_cnn_hist_val = autoencoder.project(cnn_hist_val)

    predict = autoencoder.predict(cnn_hist_train)
    print('cnn_hist error '+str(hidden_num)+':'+str(cal_error(predict, cnn_hist_train)))

    #np.save('./data/late_fusion_cnn_hist_train.npy', late_fusion_cnn_hist_train)
    #np.save('./data/late_fusion_cnn_hist_val.npy', late_fusion_cnn_hist_val)

#late_fusion_cnn_hog()
#late_fusion_hog_hist()
#late_fusion_cnn_hist()
late_fusion_cnn_hog_hist()

