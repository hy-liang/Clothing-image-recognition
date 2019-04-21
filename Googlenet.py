from keras.layers.core import Dense, Dropout, Reshape
from keras.models import Model, Sequential
from load_data import Small_X_train, Small_X_val, Y_train, Y_val
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D

def vgg_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(50, 50, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    print ('weight loaded')

    vgg_out = model.output
    vgg_out = Reshape((-1, ))(vgg_out)

    fc1 = Dense(512, activation='relu')(vgg_out)
    fc1 = Dropout(0.5)(fc1)
    '''''
    fc2 = Dense(256, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)
    '''''
    model_out = Dense(8, activation='softmax')(fc1)

    for layer in model.layers[0:17]:
        print(layer.name)
        layer.trainable = False

    my_model = Model(input=model.input, output=model_out)


    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return my_model

model = vgg_model()
model.fit(Small_X_train, Y_train, batch_size=64, epochs=50, 
	validation_data=(Small_X_val, Y_val), shuffle=True)