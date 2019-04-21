import os

import numpy as np

from skimage import io
from skimage import transform
from data_prepare import category_name


def resize_sample():
    for c in range(8):
        path  = './data/small_category_val/'+category_name[c]
        for dirpath,dirnames,filenames in os.walk(path):
            for file in filenames:
                fullpath=os.path.join(dirpath, file)
                img = io.imread(fullpath)
                img = transform.resize(img, (50, 50))
                print fullpath
		io.imsave(fullpath, img)

def save_as_npy():
    X=[]
    Y=[]
    for c in range(8):
        path  = './data/small_category_val/'+category_name[c]
        for dirpath,dirnames,filenames in os.walk(path):
            for file in filenames:
                fullpath=os.path.join(dirpath, file)
                img = io.imread(fullpath)
                X.append(img)
    		Y.append(c)
		print fullpath
    np.save('./data/Small_X_val.npy', np.array(X))
    np.save('./data/Small_Y_val.npy',np.array(Y))

#resize_sample()
save_as_npy()

