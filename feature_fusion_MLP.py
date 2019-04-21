__author__ = 'Administrator'

from keras.models import  Model,Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import SGD
from load_data import hog_X_train, hog_X_val, Y_train, Y_val, hist_X_train, hist_X_val, cnn_X_train, cnn_X_val, img_X_val, img_X_train,\
    late_fusion_cnn_hog_train, late_fusion_hog_hist_train, late_fusion_cnn_hog_val, late_fusion_hog_hist_val,late_fusion_cnn_hist_val,late_fusion_cnn_hog_hist_val,\
    late_fusion_cnn_hist_train,late_fusion_cnn_hog_hist_train
import os,sys
from sklearn import preprocessing
import numpy as np
import datetime

hog_feature_size=hog_X_train.shape[1]
hist_size = hist_X_val.shape[1]
cnn_size = cnn_X_train.shape[1]


output_num = 8
def get_MLP_model(my_input_shape, hidden_num):
    model = Sequential()

    model.add(InputLayer(input_shape=my_input_shape))
    model.add(Dense(hidden_num, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_num, activation='softmax'))

    return model

sgd = SGD(lr=0.01, momentum=0.95, decay=0.00018, nesterov=True)
'''''
print('cnn_hog_late_fusion')
late_fusion_cnn_hog_mlp = get_MLP_model((late_fusion_cnn_hog_train.shape[1],))
late_fusion_cnn_hog_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
late_fusion_cnn_hog_train_history = late_fusion_cnn_hog_mlp.fit(late_fusion_cnn_hog_train, Y_train, batch_size=50, epochs=50, validation_data=[late_fusion_cnn_hog_val, Y_val])
print ('history:', late_fusion_cnn_hog_train_history.history)
acc = late_fusion_cnn_hog_train_history.history['acc']
loss = late_fusion_cnn_hog_train_history.history['loss']
val_acc = late_fusion_cnn_hog_train_history.history['val_acc']
val_loss = late_fusion_cnn_hog_train_history.history['val_loss']
np.save('./train_result/late_fusion_cnn_hog_acc'+str(hidden_num)+'.npy', acc)
np.save('./train_result/late_fusion_cnn_hog_loss'+str(hidden_num)+'.npy', loss)
np.save('./train_result/late_fusion_cnn_hog_val_acc'+str(hidden_num)+'.npy', val_acc)
np.save('./train_result/late_fusion_cnn_hog_val_loss'+str(hidden_num)+'.npy', val_loss)
print('cnn_hog_late_fusion test:', late_fusion_cnn_hog_mlp.test_on_batch(late_fusion_cnn_hog_val, Y_val))
'''''

'''''
print('hog_hist_late_fusion')
late_fusion_hog_hist_mlp = get_MLP_model((late_fusion_hog_hist_train.shape[1],))
late_fusion_hog_hist_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
late_fusion_hog_hist_train_history = late_fusion_hog_hist_mlp.fit(late_fusion_hog_hist_train, Y_train, batch_size=50, epochs=50, validation_data=[late_fusion_hog_hist_val, Y_val])
print ('history:', late_fusion_hog_hist_train_history.history)
acc = late_fusion_hog_hist_train_history.history['acc']
loss = late_fusion_hog_hist_train_history.history['loss']
val_acc = late_fusion_hog_hist_train_history.history['val_acc']
val_loss = late_fusion_hog_hist_train_history.history['val_loss']
np.save('./train_result/late_fusion_hog_hist_acc'+str(hidden_num)+'.npy', acc)
np.save('./train_result/late_fusion_hog_hist_loss'+str(hidden_num)+'.npy', loss)
np.save('./train_result/late_fusion_hog_hist_val_acc'+str(hidden_num)+'.npy', val_acc)
np.save('./train_result/late_fusion_hog_hist_val_loss'+str(hidden_num)+'.npy', val_loss)
print('late_fusion_hog_hist_test:', late_fusion_hog_hist_mlp.test_on_batch(late_fusion_hog_hist_val, Y_val))
'''''

'''''
print('cnn_hist_late_fusion')
late_fusion_cnn_hist_mlp = get_MLP_model((late_fusion_cnn_hist_train.shape[1],))
late_fusion_cnn_hist_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
late_fusion_cnn_hist_train_history = late_fusion_cnn_hist_mlp.fit(late_fusion_cnn_hist_train, Y_train, batch_size=50, epochs=50, validation_data=[late_fusion_cnn_hist_val, Y_val])
print ('history:', late_fusion_cnn_hist_train_history.history)
acc = late_fusion_cnn_hist_train_history.history['acc']
loss = late_fusion_cnn_hist_train_history.history['loss']
val_acc = late_fusion_cnn_hist_train_history.history['val_acc']
val_loss = late_fusion_cnn_hist_train_history.history['val_loss']
np.save('./train_result/late_fusion_cnn_hist_acc'+str(hidden_num)+'.npy', acc)
np.save('./train_result/late_fusion_cnn_hist_loss'+str(hidden_num)+'.npy', loss)
np.save('./train_result/late_fusion_cnn_hist_val_acc'+str(hidden_num)+'.npy', val_acc)
np.save('./train_result/late_fusion_cnn_hist_val_loss'+str(hidden_num)+'.npy', val_loss)
print('late_fusion_cnn_hist_test:', late_fusion_cnn_hist_mlp.test_on_batch(late_fusion_cnn_hist_val, Y_val))
'''''

print('cnn_hog_hist_late_fusion')
hidden_num_s = [2048]
times = []
for hidden_num in hidden_num_s:
    late_fusion_cnn_hog_hist_mlp = get_MLP_model((late_fusion_cnn_hog_hist_train.shape[1],), hidden_num)
    late_fusion_cnn_hog_hist_mlp.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    starttime = datetime.datetime.now()
    late_fusion_cnn_hog_hist_train_history = late_fusion_cnn_hog_hist_mlp.fit(late_fusion_cnn_hog_hist_train, Y_train,batch_size=50, epochs=50,validation_data=[late_fusion_cnn_hog_hist_val, Y_val])
    endtime = datetime.datetime.now()
    now_time = (endtime - starttime).seconds
    times.append(now_time)
    print ('history:', late_fusion_cnn_hog_hist_train_history.history)
    acc = late_fusion_cnn_hog_hist_train_history.history['acc']
    loss = late_fusion_cnn_hog_hist_train_history.history['loss']
    val_acc = late_fusion_cnn_hog_hist_train_history.history['val_acc']
    val_loss = late_fusion_cnn_hog_hist_train_history.history['val_loss']
    np.save('./train_result/late_fusion_cnn_hog_hist_acc' + str(hidden_num) + '.npy', acc)
    np.save('./train_result/late_fusion_cnn_hog_hist_loss' + str(hidden_num) + '.npy', loss)
    np.save('./train_result/late_fusion_cnn_hog_hist_val_acc' + str(hidden_num) + '.npy', val_acc)
    np.save('./train_result/late_fusion_cnn_hog_hist_val_loss' + str(hidden_num) + '.npy', val_loss)
    print('cnn_hog_hist_late_fusion test:', late_fusion_cnn_hog_hist_mlp.test_on_batch(late_fusion_cnn_hog_hist_val, Y_val))