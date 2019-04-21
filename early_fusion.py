__author__ = 'Administrator'

from sklearn import preprocessing
from load_data import  cnn_X_train, cnn_X_val, hog_X_train, hog_X_val, hist_X_val, hist_X_train, Y_train, Y_val
import hpelm
import numpy as np
from keras.layers import InputLayer, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

hidden_num = 256
this_type = 'cnn+hist'
print('hidden_num:',hidden_num, 'type:', this_type)

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

def get_MLP_model(my_input_shape):
    model = Sequential()

    model.add(InputLayer(input_shape=my_input_shape))
    model.add(Dense(hidden_num, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    return model

sgd = SGD(lr=0.01, momentum=0.95, decay=0.00018, nesterov=True)

val = combine_3_feature(cnn_X_val, hist_X_val, hog_X_val)
early_fusion_feature =combine_3_feature(cnn_X_train, hist_X_train, hog_X_train)
model = get_MLP_model((early_fusion_feature.shape[1],))
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(early_fusion_feature, Y_train, batch_size=50 ,epochs=50)
print(model.test_on_batch(val, Y_val))

'''''
elm = hpelm.HPELM(inputs=early_fusion_feature.shape[1], outputs=8)
elm.add_neurons(hidden_num, func='sigm')
elm.train(early_fusion_feature, Y_train)
test(elm, early_fusion_feature, Y_train, type=this_type+ 'train '+str(hidden_num))

fusion_val = combine_feature(hog_X_train, cnn_X_train)
test(elm, fusion_val, Y_val, type=this_type + 'val '+str(hidden_num))
'''''