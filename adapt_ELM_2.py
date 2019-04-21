__author__ = 'Administrator'

from load_data import cnn_X_train, Y_train, cnn_X_val, Y_val, hog_X_train, hog_X_val, late_fusion_cnn_hog_val, late_fusion_cnn_hog_train, hist_X_train, hist_X_val
import hpelm
import numpy as np
from sklearn import  preprocessing
import os.path

types = {0:'cnn', 1:'hog', 2:'hist'}
features_train = [cnn_X_train, hog_X_train, hist_X_train]
features_val = [cnn_X_val, hog_X_val, hist_X_val]
input_sizes = [cnn_X_train.shape[1], hog_X_train.shape[1], hist_X_train.shape[1]]

K = 5
hidden_num = 128
train_sample_num = features_train[0].shape[0]

def test(elm,ty):
    X = features_train[ty]
    Y = Y_train
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

def test_update_weight(elm, ty):
    wrong_x = []
    wrong_y = []
    predict = elm.predict(features_train[ty])
    for r,t,j in zip(predict,Y_train,range(train_sample_num)):
        if not np.argmax(r)==np.argmax(t):
            sample_weight[j] = sample_weight[j]+1.0/K
            wrong_x.append(features_train[ty][j])
            wrong_y.append(t)

    return wrong_x, wrong_y

def get_train_data(k,ty):
    x = []
    y = []
    X_train = features_train[ty]
    if k==0:
        return X_train, Y_train
    else:
        elm = hpelm.HPELM(inputs=X_train.shape[1], outputs=8)
        elm.add_neurons(hidden_num, 'sigm')
        elm.load('./data/adapt_elm/elm_'+str(ty)+'_'+str(k-1))

        new_x, new_y = test_update_weight(elm, ty)
        assert new_x.__len__()==new_y.__len__()
        wrong_num = new_x.__len__()
        s_num = 0.8*train_sample_num - wrong_num
        if s_num>0:
            rand = np.random.randint(0, train_sample_num, size=int(s_num))
            for i in rand:
                new_x.append(X_train[i])
                new_y.append(Y_train[i])

        new_x = np.array(new_x)
        new_y = np.array(new_y)
        return new_x, new_y


def train():
    for ty in range(3):
        elm_weight = []
        for k in range(K):
            X,Y = get_train_data(k, ty)
            elm = hpelm.HPELM(inputs=features_train[ty].shape[1], outputs=8)
            elm.add_neurons(hidden_num, 'sigm')
            elm.train(X,Y)
            weight = test(elm, ty)
            elm_weight.append(weight)
            print ('elm_'+str(k)+' weight:'+str(weight))
            elm.save('./data/adapt_elm/elm_'+str(ty)+'_'+str(k))
        np.save('./data/adapt_elm/weight_'+str(ty)+'.npy', elm_weight)

def test_adapt_elm_feature_fusion(features, Y):
   result = []
   for i in range(3):
       weights = np.load('./data/adapt_elm/weight_'+str(i)+'.npy')
       for k in range(K):
           elm = hpelm.ELM(inputs=input_sizes[i], outputs=8)
           elm.load('./data/adapt_elm/elm_'+str(i)+'_'+str(k))
           predict = elm.predict(features[i])*weights[k]
           if i==0 and k==0:
               result = predict
           else:
               result = result + predict
   counter = 0.0
   for r,t in zip(result, Y):
       if np.argmax(r)==np.argmax(t):
           counter+=1
   right_rate = counter/features[0].shape[0]
   print('adapt elm right rate:'+str(right_rate))



print('hidden_num:',hidden_num)
print('K:', K)

sample_weight = load_sample_weight()
train()
test_adapt_elm_feature_fusion(features_train, Y_train)
test_adapt_elm_feature_fusion(features_val, Y_val)

