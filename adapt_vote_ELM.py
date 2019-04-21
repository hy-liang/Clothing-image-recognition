__author__ = 'Administrator'

from load_data import cnn_X_train, Y_train, cnn_X_val, Y_val
import hpelm
import numpy as np

train_sample_num = cnn_X_train.shape[0]
K = 5
sample_weight = [1 for i in range(train_sample_num)]
hidden_num = 512
input_size = cnn_X_train.shape[1]
elm_weight = []

def test(elm):
    predict = elm.predict(cnn_X_train)
    right_weight = 0.0
    counter = 0.0
    for r,t,i in zip(predict, Y_train, range(train_sample_num)):
        if np.argmax(r)==np.argmax(t):
            right_weight+=sample_weight[i]
            counter+=1
    print('right_weight:'+str(right_weight))
    weight = right_weight/np.sum(sample_weight)
    print ('right rate:'+str(counter/train_sample_num))
    return weight

def test_update_weight(elm):
    wrong_x = []
    wrong_y = []
    predict = elm.predict(cnn_X_train)
    for r,t,i in zip(predict,Y_train,range(train_sample_num)):
        if np.argmax(r)==np.argmax(t):
            sample_weight[i] = sample_weight[i]/2.0
        else:
            wrong_x.append(cnn_X_train[i])
            wrong_y.append(t)

    return wrong_x, wrong_y

def get_train_data(k):
    x = []
    y = []
    if k==0:
        return cnn_X_train, Y_train
    else:
        elm = hpelm.HPELM(inputs=input_size, outputs=8)
        elm.add_neurons(hidden_num, 'sigm')
        elm.load('./data/adapt_elm/elm_'+str(k-1))

        new_x, new_y = test_update_weight(elm)
        assert new_x.__len__()==new_y.__len__()
        wrong_num = new_x.__len__()
        rand = np.random.randint(0, train_sample_num, size=int(wrong_num))
        for i in rand:
            new_x.append(cnn_X_train[i])
            new_y.append(Y_train[i])

        new_x = np.array(new_x)
        new_y = np.array(new_y)
        return new_x, new_y


def train():
    for k in range(K):
        X,Y = get_train_data(k)
        elm = hpelm.HPELM(inputs=input_size, outputs=8)
        elm.add_neurons(hidden_num, 'sigm')
        elm.train(X,Y)
        weight = test(elm)
        elm_weight.append(weight)
        print ('elm_'+str(k)+' weight:'+str(weight))
        print('sum:'+str(np.sum(sample_weight)))
        elm.save('./data/adapt_elm/elm_'+str(k))

def test_adapt_vote_elm(X,Y):
    result = np.zeros((X.shape[0], 8))
    for k in range(K):
        elm = hpelm.HPELM(inputs=input_size, outputs=8)
        elm.add_neurons(hidden_num, 'sigm')
        elm.load('./data/adapt_elm/elm_'+str(k))

        predict = elm.predict(X)
        max = np.argmax(predict, axis=1)
        for n in range(X.shape[0]):
            result[n][max[n]]+=1

    counter = 0.0
    for r,t in zip(result, Y):
        if np.argmax(r)==np.argmax(t):
            counter+=1
    right_rate = counter/X.shape[0]
    print('adapt elm right rate:'+str(right_rate))


train()
test_adapt_vote_elm(cnn_X_train, Y_train)
test_adapt_vote_elm(cnn_X_val, Y_val)
