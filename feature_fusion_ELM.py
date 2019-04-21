__author__ = 'Administrator'
import hpelm
import numpy as np
from load_data import hog_X_train, hog_X_val, Y_train, Y_val, hist_X_train, hist_X_val, cnn_X_train, cnn_X_val, \
    late_fusion_cnn_hog_train, late_fusion_hog_hist_train, late_fusion_cnn_hog_val, late_fusion_hog_hist_val,late_fusion_cnn_hist_val,late_fusion_cnn_hog_hist_val,\
    late_fusion_cnn_hist_train,late_fusion_cnn_hog_hist_train
import os,sys
import datetime
from confusion_matrix import get_sampleNumOfEachClass, confusion_matrix

hog_feature_size=hog_X_train.shape[1]
hist_size = hist_X_val.shape[1]
cnn_size = cnn_X_train.shape[1]
category_num=8


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
    np.save('./con_mat2.npy', con_mat)

'''''
---------------------hog cnn elm-------------------------

cnn_hog_elm = hpelm.HPELM(inputs = late_fusion_cnn_hog_train.shape[1], outputs=category_num)
cnn_hog_elm.add_neurons(hidden_num, 'sigm')
print (late_fusion_cnn_hog_train.shape[0])
print (cad_y_train.shape[0])
cnn_hog_elm.train(late_fusion_cnn_hog_train, cad_y_train)
test(cnn_hog_elm, late_fusion_cnn_hog_train, cad_y_train, type='cnn+hog train')
test(cnn_hog_elm, late_fusion_cnn_hog_val, cad_y_val, type='cnn+hog val')
'''''
'''''
---------------------hog hist elm-------------------------

hog_hist_elm = hpelm.HPELM(inputs = late_fusion_hog_hist_train.shape[1], outputs=category_num)
hog_hist_elm.add_neurons(hidden_num, 'sigm')
hog_hist_elm.train(late_fusion_hog_hist_train, Y_train)
test(hog_hist_elm, late_fusion_hog_hist_train, Y_train, type='hog+hist train')
test(hog_hist_elm, late_fusion_hog_hist_val, Y_val, type='hog+hist val')
'''''

'''''
---------------------cnn hist elm-------------------------

cnn_hist_elm = hpelm.HPELM(inputs = late_fusion_cnn_hist_train.shape[1], outputs=category_num)
cnn_hist_elm.add_neurons(hidden_num, 'sigm')
cnn_hist_elm.train(late_fusion_cnn_hist_train, Y_train)
test(cnn_hist_elm, late_fusion_cnn_hist_train, Y_train, type='cnn+hist train')
test(cnn_hist_elm, late_fusion_cnn_hist_val, Y_val, type='cnn+hist val')
'''''

'''''
---------------------cnn hog hist elm-------------------------
'''''

hidden_num_s = [4096]
times = []
for hidden_num in hidden_num_s:
    cnn_hist_elm = hpelm.HPELM(inputs=late_fusion_cnn_hog_hist_train.shape[1], outputs=category_num)
    cnn_hist_elm.add_neurons(hidden_num, 'sigm')

    starttime = datetime.datetime.now()
    cnn_hist_elm.train(late_fusion_cnn_hog_hist_train, Y_train)
    endtime = datetime.datetime.now()
    now_time = (endtime - starttime).seconds
    times.append(now_time)
    test(cnn_hist_elm, late_fusion_cnn_hog_hist_train, Y_train, type='cnn+hog+hist train')
    test(cnn_hist_elm, late_fusion_cnn_hog_hist_val, Y_val, type='cnn+hog+hist val')
    cnn_hist_elm = 0
print times
