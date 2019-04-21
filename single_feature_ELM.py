__author__ = 'Administrator'

__author__ = 'Administrator'
import hpelm
import numpy as np
from confusion_matrix import get_sampleNumOfEachClass, confusion_matrix
from load_data import hog_X_train, hog_X_val, Y_train, Y_val, hist_X_train, hist_X_val, cnn_X_train, cnn_X_val
import os,sys

hog_feature_size=hog_X_train.shape[1]
hist_size = hist_X_val.shape[1]
cnn_size = cnn_X_train.shape[1]
category_num=8
hidden_num = 1024

'''''
--------------test function-------------
elm----the elm for test
x----input
y----target output
type----name of test:'hog_train', 'hog_val', 'cnn_train', 'cnn_val', 'hist_train', 'hist_val'
'''''
def test(elm, x, y, type=''):
    sample_num = x.shape[0]
    print( 'predicting!')
    predict = elm.predict(x)
    counter = .0
    print('-------------------------------------------\n testing')
    for r,t in zip(predict, y):
        if np.argmax(r)==np.argmax(t):
            counter+=1

    print(type+' right rate:', counter/sample_num)
    con_mat = confusion_matrix(predict, y)
    print (con_mat)
    np.save('./con_mat3.npy', con_mat)

print  Y_train.shape
print  Y_val.shape
for i in range(1):
    '''''
    ------------------hog elm------------------
    

    hog_elm = hpelm.HPELM(inputs=hog_feature_size, outputs=category_num)
    hog_elm.add_neurons(hidden_num, 'sigm')
    #hog_elm.add_neurons(512, 'rbf2')
    hog_elm.train(hog_X_train, Y_train)

    test(hog_elm, hog_X_train, Y_train, type='hog_train')
    test(hog_elm, cad_hog_X_val, cad_y_val, type='hog_val')
    '''''

    '''''
    -------------------hist elm-------------------
    
    print ('hist_X_train.shape:',cad_hist_X_train.shape)
    hist_elm = hpelm.HPELM(inputs=hist_size, outputs=category_num)
    hist_elm.add_neurons(hidden_num, 'sigm')
    #hog_elm.add_neurons(512, 'rbf2')
    hist_elm.train(cad_hist_X_train, cad_y_train)

    test(hist_elm, cad_hist_X_train, cad_y_train, type='hist_train')
    test(hist_elm, cad_hist_X_val, cad_y_val, type='hist_val')
    '''''
    '''''
    -------------------cnn elm-------------------
    '''''
    cnn_elm = hpelm.HPELM(inputs=cnn_size, outputs=category_num)
    cnn_elm.add_neurons(hidden_num, 'sigm')
    #hog_elm.add_neurons(512, 'rbf2')
    cnn_elm.train(cnn_X_train, Y_train)
    test(cnn_elm, cnn_X_train, Y_train, type='cnn_train')
    #test(cnn_elm, cnn_X_val, Y_val, type='cnn_val')