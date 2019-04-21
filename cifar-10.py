__author__ = 'Administrator'

from cifar_CNN import My_CNN
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage import transform
from load_data import Small_X_train, Small_X_val

(X_train,Y_train), (X_test, Y_test) = cifar10.load_data()
print (X_train.shape, Y_train.shape)
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
X_train = X_train/255.0
X_test = X_test/255.0

def resize_x():
    X = []
    for x in Small_X_val:
        x2 = transform.resize(x, (32, 32))
        print( x2.shape)

resize_x()

my_CNN = My_CNN()
sgd = SGD(lr=0.01, momentum=0.95, decay=0.000198, nesterov=True)

my_CNN.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

my_CNN.fit(X_train, Y_train, batch_size=50, epochs=50)





