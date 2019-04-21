import numpy as np

f=open('print.txt', 'w+')
a1 = np.load('con_mat.npy')
a2 = np.load('fusion_con_mat.npy')
a3 = np.load('cnn_con_mat.npy')

for i in a1:
    for j in i:
        f.write(str(j)+'\t')
    f.write('\n')
for i in a2:
    for j in i:
        f.write(str(j)+'\t')
    f.write('\n')
for i in a3:
    for j in i:
        f.write(str(j)+'\t')
    f.write('\n')