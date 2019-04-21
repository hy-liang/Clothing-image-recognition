from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Activation

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./5.jpg')
batch_img = np.expand_dims(img, axis=0)


model = Sequential()
model.add(Convolution2D(3,3,3, input_shape=(50, 50, 3)))

conv_out = model.predict(batch_img)

out = np.squeeze(conv_out, axis=0)

plt.imshow(out[:,:,2])
plt.show()