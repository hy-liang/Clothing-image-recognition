from __future__ import division
__author__ = 'devilc'

from skimage import exposure, io, transform
import skimage.color as color
import matplotlib.pyplot as plt
import numpy as np
import cv2

#img = io.imread('5.jpg')
#resize_img = transform.resize(img, (50, 50))


#grayimg = color.rgb2gray(img)

grayimg = cv2.cvtColor(cv2.imread("./5.jpg"),
                   cv2.COLOR_BGR2GRAY)
print (grayimg)
#hist = exposure.histogram(img)[0]
#print hist
#print hist.shape
arr = grayimg.flatten()

print (arr)

plt.figure()
#plt.imshow(grayimg, cmap='gray')
n, bins, patches = plt.hist(arr, bins=256, normed=0,edgecolor='None',facecolor='black',)
plt.xlabel('grey level')
plt.ylabel('num')
plt.show()