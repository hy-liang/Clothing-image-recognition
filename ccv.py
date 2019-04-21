# -*- coding:utf-8 -*-
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plot

def QuantizeColor(img, n=64):
  div = 256//n
  rgb = cv2.split(img)
  q = []
  for ch in rgb:
    vf = np.vectorize(lambda x, div: int(x//div)*div)
    quantized = vf(ch, div)
    q.append(quantized.astype(np.uint8))
  d_img = cv2.merge(q)
  return d_img


def ccv(src, tau=0, n=64):
  img = src.copy()
  row, col, channels = img.shape
  print row,col,channels
  if not col == 300:
    aspect = 300.0/col
    img = cv2.resize(img, None, fx=aspect, fy=aspect, interpolation = cv2.INTER_CUBIC)
  row, col, channels = img.shape
  # blur
  img = cv2.GaussianBlur(img, (3,3),0)
  # quantize color
  img = QuantizeColor(img, n)
  bgr = cv2.split(img)
  #bgr = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
  if tau == 0:
    tau = row*col * 0.1
  alpha = np.zeros(n)
  beta = np.zeros(n)
  # labeling
  for i,ch in enumerate(bgr):
    ret,th = cv2.threshold(ch,130,255,0)
    ret, labeled, stat, centroids = cv2.connectedComponentsWithStats(th, None, cv2.CC_STAT_AREA, None, connectivity=8)
    # generate ccv
    areas = [[v[4],label_idx] for label_idx,v in enumerate(stat)]
    coord = [[v[0],v[1]] for label_idx,v in enumerate(stat)]
    # Counting
    for a,c in zip(areas,coord):
      area_size = a[0]
      x,y = c[0], c[1]
      if (x < ch.shape[1]) and (y < ch.shape[0]):
        bin_idx = int(ch[y,x]//(256//n))
        if area_size >= tau:
          alpha[bin_idx] = alpha[bin_idx] + area_size
        else:
          beta[bin_idx] = beta[bin_idx] + area_size
  return alpha, beta

def ccv_plot(img, alpha, beta, n=64):
  import matplotlib.pyplot as plt
  X = [x for x in range(n*2)]
  Y = alpha.tolist()+beta.tolist()
  with open('ccv.csv','w') as f:
    f.write(str(Y))
  im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.subplot(2,1,1)
  plt.imshow(im)
  plt.subplot(2,1,2)
  plt.bar(X, Y, align='center')
  #plt.yscale('log')
  #plt.xticks(X, (['alpha']*n)+(['beta']*n))
  plt.show()  

'''''
if __name__ == '__main__':
  argvs = sys.argv
  argc = len(argvs)
  img = cv2.imread(argvs[1])
  n = int(argvs[2])
  alpha, beta = ccv(img, tau=0,n=n)
  CCV = alpha.tolist()+beta.tolist()
  #assert(sum(CCV) == img.size)
  assert(n == len(alpha) and n == len(beta))
  ccv_plot(img, alpha, beta, n)
'''''

img = cv2.imread('./test_images/002_0005.jpg')
hist = cv2.calcHist(img,[0], None, [256], [0.0, 255.0])
plot.plot(hist)
plot.show()
print hist
'''''
n = 64
alpha, beta = ccv(img, tau=0, n=n)
CCV = alpha.tolist() + beta.tolist()
# assert(sum(CCV) == img.size)
assert (n == len(alpha) and n == len(beta))
ccv_plot(img, alpha, beta, n)
'''''