__author__ = 'Administrator'

import matplotlib.pyplot as plt
import numpy as np

cnn_acc = np.load('./train_result/cnn_acc.npy')
cnn_loss = np.load('./train_result/cnn_loss.npy')

plt.plot(cnn_acc)