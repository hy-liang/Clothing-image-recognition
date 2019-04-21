__author__ = 'Administrator'

__author__ = 'Administrator'

from keras.models import  Model,Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import SGD
from load_data import hog_X_train, hog_X_val, Y_train, Y_val, hist_X_train, hist_X_val, cnn_X_train, cnn_X_val, Small_X_train, Small_X_val
import os,sys
from sklearn import preprocessing
import numpy as np
import datetime

hog_feature_size=hog_X_train.shape[1]
hist_size = hist_X_val.shape[1]
cnn_size = cnn_X_train.shape[1]

def get_MLP_model(my_input_shape):
    model = Sequential()

    model.add(InputLayer(input_shape=my_input_shape))
    model.add(Dense(hidden_num, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_num, activation='softmax'))

    return model

hidden_num = 2048
output_num = 8

sgd = SGD(lr=0.01, momentum=0.95, decay=0.00018, nesterov=True)

print('hidden num'+str(hidden_num))

'''''
cnn_mlp
'''''
cnn_mlp = get_MLP_model((cnn_size,))

cnn_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
starttime = datetime.datetime.now()
cnn_train_history = cnn_mlp.fit(cnn_X_train, Y_train, batch_size=50, epochs=50,validation_data=[cnn_X_val, Y_val])
endtime = datetime.datetime.now()
print('time:'+str((endtime - starttime).microseconds))
print('history:', cnn_train_history.history)
acc = cnn_train_history.history['acc']
loss = cnn_train_history.history['loss']
val_acc = cnn_train_history.history['val_acc']
val_loss = cnn_train_history.history['val_loss']
np.save('./train_result/cnn_mlp_acc'+str(hidden_num)+'.npy', acc)
np.save('./train_result/cnn_mlp_loss'+str(hidden_num)+'.npy', loss)
np.save('./train_result/cnn_mlp_val_acc'+str(hidden_num)+'.npy', val_acc)
np.save('./train_result/cnn_mlp_val_loss'+str(hidden_num)+'.npy', val_loss)
print('cnn test:', cnn_mlp.test_on_batch(cnn_X_val, Y_val))


'''''
hog_mlp

hog_mlp = get_MLP_model((hog_feature_size, ))
hog_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
hog_train_history = hog_mlp.fit(preprocessing.scale(cad_hog_X_train), cad_y_train, batch_size=50, epochs=50, validation_data=[preprocessing.scale(cad_hog_X_val), cad_y_val])
print ('history:', hog_train_history.history)
acc = hog_train_history.history['acc']
loss = hog_train_history.history['loss']
val_acc = hog_train_history.history['val_acc']
val_loss = hog_train_history.history['val_loss']
np.save('./train_result/hog_mlp_acc'+str(hidden_num)+'.npy', acc)
np.save('./train_result/hog_mlp_loss'+str(hidden_num)+'.npy', loss)
np.save('./train_result/hog_mlp_val_acc'+str(hidden_num)+'.npy', val_acc)
np.save('./train_result/hog_mlp_val_loss'+str(hidden_num)+'.npy', val_loss)
print (hog_mlp.test_on_batch(preprocessing.scale(cad_hog_X_val), cad_y_val))
'''''

'''''
hist_mlp

hist_mlp = get_MLP_model((hist_size,))
hist_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
hist_train_history = hist_mlp.fit(preprocessing.scale(hist_X_train), Y_train, batch_size=50, epochs=50, validation_data=[preprocessing.scale(hist_X_val), Y_val])
acc = hist_train_history.history['acc']
loss = hist_train_history.history['loss']
val_acc = hist_train_history.history['val_acc']
val_loss = hist_train_history.history['val_loss']
np.save('./train_result/hist_mlp_acc'+str(hidden_num)+'.npy', acc)
np.save('./train_result/hist_mlp_loss'+str(hidden_num)+'.npy', loss)
np.save('./train_result/hist_mlp_val_acc'+str(hidden_num)+'.npy', val_acc)
np.save('./train_result/hist_mlp_val_loss'+str(hidden_num)+'.npy', val_loss)
print ('history:', hist_train_history.history)
print (hist_mlp.test_on_batch(preprocessing.scale(hist_X_val), Y_val))

hidden_num = hidden_num*2
'''''
'''''
picture mlp

picture_mlp = get_MLP_model((7500,))
picture_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
picture_mlp_history = picture_mlp.fit(Small_X_train.reshape(132887, 7500), Y_train, batch_size=50, epochs=30, validation_data=[Small_X_val.reshape(25406, 7500), Y_val])
acc = picture_mlp_history.history['acc']
loss = picture_mlp_history.history['loss']
val_acc = picture_mlp_history.history['val_acc']
val_loss = picture_mlp_history.history['val_loss']
np.save('./train_result/picture_mlp_acc'+str(hidden_num)+'.npy', acc)
np.save('./train_result/picture_mlp_loss'+str(hidden_num)+'.npy', loss)
np.save('./train_result/picture_mlp_val_acc'+str(hidden_num)+'.npy', val_acc)
np.save('./train_result/picture_mlp_val_loss'+str(hidden_num)+'.npy', val_loss)
print ('history:', picture_mlp_history.history)
print (picture_mlp_history.test_on_batch(Small_X_val.reshape(25406, 7500), Y_val))
'''''