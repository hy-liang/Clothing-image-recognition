import numpy as np
import cv2

import matplotlib.pyplot as plt

import skimage.feature as ft
'''''
cell_size = (6, 6)
block_size = (3, 3)
nbins = 8

img = cv2.cvtColor(cv2.imread("./3.jpg"),
                   cv2.COLOR_BGR2GRAY)

hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
hog_feats = hog.compute(img)\
               .reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               .transpose((1, 0, 2, 3, 4))

gradients = np.zeros((n_cells[0], n_cells[1], nbins))

# count cells (border cells appear less often across overlapping groups)
cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

for off_y in range(block_size[0]):
    for off_x in range(block_size[1]):
        gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                  off_x:n_cells[1] - block_size[1] + off_x + 1] += \
            hog_feats[:, :, off_y, off_x, :]
        cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                   off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

# Average gradients
gradients /= cell_count

# Preview
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()

bin = 5  # angle is 360 / nbins * direction
plt.pcolor(gradients[:, :, bin])
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()
'''''

from skimage import io, transform
from skimage import color

img = io.imread('./5.jpg')
plt.figure()

GrayImage = color.rgb2gray(img)

print(img.shape)

print(img.shape)
feature, hog_img = ft.hog(GrayImage, orientations=8, pixels_per_cell=(5, 5), cells_per_block=(2,2), block_norm='L2', visualise=True)
plt.imshow(hog_img, cmap='gray')
plt.grid(True, linestyle = "--", color = "w", linewidth = "1")
plt.show()
