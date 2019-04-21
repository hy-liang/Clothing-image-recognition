__author__ = 'Administrator'

from keras.datasets import cifar10
from keras import utils
from skimage import exposure
from skimage import color
import numpy as np
import hpelm

def test(elm, x, y, type=''):
    sample_num = x.shape[0]
    print( 'predicting!')
    predict = elm.predict(x)
    counter = .0
    print('-------------------------------------------\n testing')
    for r,t in zip(predict, y):
        if np.argmax(r)==np.argmax(t):
            counter+=1
    print(type+'right number:', counter)
    print(type+' right rate:', counter/sample_num)

(X_train, Y_train), (X_test , Y_test) = cifar10.load_data()
X_train = X_train.transpose(0,2,3,1)
X_test = X_test.transpose(0,2,3,1)
Y_train = utils.to_categorical(Y_train)
Y_test = utils.to_categorical(Y_test)

hist_train = []
for img in X_train:
    img =  color.rgb2gray(img)
    hist = exposure.histogram(img)
    hist_train.append(np.array(hist[0]))
hist_train = np.array(hist_train)

elm = hpelm.HPELM(inputs = hist_train.shape[1], outputs=10)
elm.add_neurons(512, 'sigm')
elm.train(hist_train, Y_train)

test(elm, hist_train, Y_train, type='cifar10_train')
