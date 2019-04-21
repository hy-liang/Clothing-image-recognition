import sys
import os

#sys.path.append('../')
from Small_Sample_CNN_model import Small_Sample_CNN
from keras.optimizers import SGD
from load_data import Small_X_train, Y_train, Small_X_val, Y_val
from keras.models import Model
import numpy as np
#os.chdir(os.path.split(os.path.abspath(__file__))[0])

def get_CNN_Category_model():
    model = Small_Sample_CNN()
    model.load_weights('./data/Small_Sample_CNN_model')
    return model

def train():
    hist = CNN_Category_model.fit(Small_X_train, Y_train, nb_epoch=20, batch_size=50, validation_data=[Small_X_val, Y_val])
    acc = hist.history['acc']
    loss = hist.history['loss']
    val_acc = hist.history['val_acc']
    val_loss = hist.history['val_loss']
    np.save('./train_result/cnn_acc3.npy', np.array(acc))
    np.save('./train_result/cnn_loss3.npy', np.array(loss))
    np.save('./train_result/cnn_val_acc3.npy', np.array(val_acc))
    np.save('./train_result/cnn_val_loss3.npy', np.array(val_loss))
    CNN_Category_model.save_weights('./data/Small_Sample_CNN_model')


CNN_Category_model = get_CNN_Category_model()
#CNN_Category_model.summary()
#set parameters for compile
#sgd = SGD(lr=0.0001, momentum=0.95, decay=0.000003, nesterov=True)
#CNN_Category_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



#train()
