import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2


from   skimage.feature import local_binary_pattern


def sklbp(img,hdiv=1, vdiv=1, mapping='nri_uniform',norm=False,names=False):
  if mapping == 'nri_uniform':
    n_bins = 59
    st     = 'LBP'
  else:
    n_bins = 10
    st     = 'LBPri'
  (nv,nh) = (vdiv,hdiv)
  nn  = int(np.fix(img.shape[0]/nv))
  mm  = int(np.fix(img.shape[1]/nh))
  k = 0
  for r in range(0,img.shape[0] - nn+1, nn):
    for c in range(0,img.shape[1] - mm+1, mm):
      w = img[r:r+nn,c:c+mm]
      lbp = local_binary_pattern(w,8,1,mapping)
      (xrc, _) = np.histogram(lbp.ravel(), bins=n_bins,range=(0, n_bins))
      if k==0:
        X = xrc
        k = 1
      else:
        X = np.concatenate((X,xrc))
  if norm:
      X = X/np.linalg.norm(X)
  if names==True:
    Xn = []
    for k in range(n_bins):
      Xn.append(st+'-'+str(k))
    return X,Xn
  else:
    return X




