import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2


from   skimage.feature import local_binary_pattern
from   skimage.filters import gabor_kernel
from   scipy import ndimage as ndi

def lbp(img,hdiv=1, vdiv=1, mapping='nri_uniform',norm=False,names=False):
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
    for i in range(vdiv):
      for j in range(hdiv):
        for k in range(n_bins):
          Xn.append(st+'('+str(i)+','+str(j)+')-'+str(k))
    return X,Xn
  else:
    return X


def haralick(img,hdiv=1, vdiv=1, distance=1,norm=False,names=False):
  (nv,nh) = (vdiv,hdiv)
  nn  = int(np.fix(img.shape[0]/nv))
  mm  = int(np.fix(img.shape[1]/nh))
  k = 0
  fst = ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']
  for r in range(0,img.shape[0] - nn+1, nn):
    for c in range(0,img.shape[1] - mm+1, mm):
      w  = img[r:r+nn,c:c+mm]
      g  = skimage.feature.graycomatrix(w, [distance], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
      x0 = skimage.feature.graycoprops(g, fst[0])
      x1 = skimage.feature.graycoprops(g, fst[1])
      x2 = skimage.feature.graycoprops(g, fst[2])
      x3 = skimage.feature.graycoprops(g, fst[3])
      x4 = skimage.feature.graycoprops(g, fst[4])
      x5 = skimage.feature.graycoprops(g, fst[5])
      #haralick = [np.mean(x1),np.mean(x2),np.mean(x3),np.mean(x4),np.mean(x5),np.mean(x6)]
      haralick = np.concatenate((x0,x1,x2,x3,x4,x5), axis=1)
      if k==0:
        X = haralick[0]
        k = 1
      else:
        X = np.concatenate((X,haralick))
  if norm:
    X = X/np.linalg.norm(X)

  if names==True:
    Xn = []
    for i in range(vdiv):
      for j in range(hdiv):
        for k in range(6):
          Xn.append('Haralick('+str(i)+','+str(j)+')-'+fst[k])
    return X,Xn
  else:
    return X


  return x


def gabor(image,angles=4,sigmas=(1,3),frequencies=(0.05, 0.25)):
  kernels = []
  for theta in range(angles):
    theta = theta / 4. * np.pi
    for sigma in sigmas:
        for frequency in frequencies:
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
  feats = np.zeros((len(kernels), 2), dtype=np.double)
  for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
  X = feats.reshape(len(kernels)*2,)
  return X


