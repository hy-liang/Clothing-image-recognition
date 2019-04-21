__author__ = 'Administrator'

from load_data import cad_cnn_X_train, cad_y_train, cad_cnn_X_val, cad_y_val, cad_hog_X_train, cad_hog_X_val, late_fusion_cnn_hog_val, late_fusion_cnn_hog_train, cad_hist_X_train, cad_hist_X_val,\
    late_fusion_cnn_hog_hist_train,late_fusion_cnn_hog_hist_val
import hpelm
import numpy as np
from sklearn import  preprocessing
import os.path

types = {0:'cnn', 1:'hog', 2:'hist'}
features_train = [cad_cnn_X_train, cad_hog_X_train, cad_hist_X_train]
features_val = [cad_cnn_X_val, cad_hog_X_val, cad_hist_X_val]

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

X_train = combine_3_feature(cad_cnn_X_train, cad_hog_X_train, cad_hist_X_train)
X_val = combine_3_feature(cad_cnn_X_val, cad_hog_X_val, cad_hist_X_val)
train_sample_num = X_train.shape[0]
input_size = X_train.shape[1]

def test(elm, X, Y):
    predict = elm.predict(X)
    right_weight = 0.0
    counter = 0.0
    for r,t,i in zip(predict, Y, range(train_sample_num)):
        if np.argmax(r)==np.argmax(t):
            right_weight+=sample_weight[i]
            counter+=1
    print('right_weight:'+str(right_weight))
    weight = right_weight/np.sum(sample_weight)
    print ('right rate:'+str(counter/X.shape[0]))
    return weight

def load_sample_weight():
    if os.path.exists('./data/adapt_ELM_sample_weight.npy'):
        sample_weight = np.load('./data/adapt_ELM_sample_weight.npy')
        return sample_weight
    else:
        sample_weight = [1 for i in range(train_sample_num)]
        return sample_weight

def test_update_weight(elm):
    wrong_x = []
    wrong_y = []
    predict = elm.predict(X_train)
    for r,t,i in zip(predict,cad_y_train,range(train_sample_num)):
        if not np.argmax(r)==np.argmax(t):
            sample_weight[i] = sample_weight[i]+5.0/K
            wrong_x.append(X_train[i])
            wrong_y.append(t)

    return wrong_x, wrong_y

def get_train_data(k):
    x = []
    y = []
    if k==0:
        return X_train, cad_y_train
    else:
        elm = hpelm.HPELM(inputs=input_size, outputs=8)
        elm.add_neurons(hidden_num, 'sigm')
        elm.load('./data/adapt_elm'+str(K)+'/elm_'+str(k-1))

        new_x, new_y = test_update_weight(elm)
        assert new_x.__len__()==new_y.__len__()
        wrong_num = new_x.__len__()
        s_num = 0.8*train_sample_num - wrong_num
        if s_num>0:
            rand = np.random.randint(0, train_sample_num, size=int(s_num))
            for i in rand:
                new_x.append(X_train[i])
                new_y.append(cad_y_train[i])

        new_x = np.array(new_x)
        new_y = np.array(new_y)
        return new_x, new_y


def train():
    for k in range(K):
        X,Y = get_train_data(k)
        elm = hpelm.HPELM(inputs=input_size, outputs=7)
        elm.add_neurons(hidden_num, 'sigm')
        elm.train(X,Y)
        weight = test(elm, X_train, cad_y_train)
        test(elm, X_val, cad_y_val)
        elm_weight.append(weight)
        print ('elm_'+str(k)+' weight:'+str(weight))
        elm.save('./data/adapt_elm'+str(K)+ '/elm_'+str(k))

def test_adapt_elm(X,Y):
    result = []
    for k in range(K):
        elm = hpelm.HPELM(inputs=input_size, outputs=7)
        elm.add_neurons(hidden_num, 'sigm')
        elm.load('./data/adapt_elm'+str(K)+'/elm_'+str(k))

        predict = elm.predict(X)*elm_weight[k]
        if k==0:
            result = predict
        else:
            result = result + predict
    counter = 0.0
    for r,t in zip(result, Y):
        if np.argmax(r)==np.argmax(t):
            counter+=1
    right_rate = counter/X.shape[0]
    print('adapt elm right rate:'+str(right_rate))

Ks = [10]
hidden_num_s = [512]
for K in Ks:
    for hidden_num in hidden_num_s:
        elm_weight = []
        print('hidden_num:', hidden_num)
        print('K:', K)

        sample_weight = load_sample_weight()

        train()
        test_adapt_elm(X_train, cad_y_train)
        test_adapt_elm(X_val, cad_y_val)
