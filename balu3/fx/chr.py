import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
from   skimage.feature import local_binary_pattern
from   skimage.filters import gabor_kernel
from   scipy import ndimage as ndi
from   scipy.fftpack import dct
from   scipy.stats import kurtosis,skew
from   scipy.ndimage.morphology import binary_dilation as imdilate
from   balu3.im.proc import fspecial, im_grad

def basicint(image, region=None, *, mask=15, names=False):
    if region is None:
        region = np.ones(shape=image.shape, dtype=int)
    
    r_perim = skimage.segmentation.find_boundaries(region,mode='inner')
    region = region.astype(bool)

    image = image.astype(float)

    kernel = fspecial('gaussian',mask,mask/8.5)

    im1, _ = im_grad(image, kernel)
    im2, _ = im_grad(im1  , kernel)

    if not region.all():
        boundary_gradient = np.abs(im1[r_perim]).mean()
    else:
        boundary_gradient = -1

    useful_img = image[region]

    intensity_mean     = useful_img.mean()
    intensity_std      = useful_img.std(ddof=1)
    intensity_kurtosis = kurtosis(useful_img, fisher=False)
    intensity_skewness = skew(useful_img)
    mean_laplacian     = im2[region].mean()

    X                  = np.array([intensity_mean,
                             intensity_std,
                             intensity_kurtosis,
                             intensity_skewness,
                             mean_laplacian,
                             boundary_gradient])
    if names:
      Xn = ['Intensity Mean',
            'Intensity StdDev',
            'Intensity Kurtosis',
            'Intensity Skewness',
            'Mean Laplacian',
            'Mean Boundary Gradient']
      return X,Xn
    else:
      return X



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



def gabor(img,hdiv=1, vdiv=1,angles=4,sigmas=(1,3),frequencies=(0.05, 0.25),norm=False):
  kernels = []
  for theta in range(angles):
    theta = theta / 4. * np.pi
    for sigma in sigmas:
        for frequency in frequencies:
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
  n       = len(kernels)
  (nv,nh) = (vdiv,hdiv)
  nn  = int(np.fix(img.shape[0]/nv))
  mm  = int(np.fix(img.shape[1]/nh))
  k = 0
  feats = np.zeros((len(kernels), 2), dtype=np.double)
  X = np.zeros((n*nh*nv*2,))
  t = 0
  for r in range(0,img.shape[0] - nn+1, nn):
    for c in range(0,img.shape[1] - mm+1, mm):
      w  = img[r:r+nn,c:c+mm]
      for j in range(n): #,kernel in enumerate(kernels):
        kernel = kernels[j]
        filtered = ndi.convolve(w, kernel, mode='wrap')
        X[t]   = filtered.mean()
        X[t+1] = filtered.var()
        t = t+2
  if norm:
    X = X/np.linalg.norm(X)
  return X

def fourier(I,region=None,Nfourier=64,Mfourier=64,nfourier=4,mfourier=4):
    if region is None:
        region = np.ones_like(I)
    I[region == 0] = 0
    Im       = cv2.resize(I,(Nfourier,Mfourier))
    FIm      = np.fft.fft2(Im)
    Y        = FIm[0:int(Nfourier/2),0:int(Mfourier/2)]
    x        = np.abs(Y)
    F        = cv2.resize(x,(nfourier,mfourier))
    f        = np.reshape(F,(nfourier*mfourier,))
    x        = np.angle(Y)
    A        = cv2.resize(x,(nfourier,mfourier))
    a        = np.reshape(A,(nfourier*mfourier,))
    X        = np.concatenate((f,a))
    return X

def dct(I,region=None,Ndct=64,Mdct=64,ndct=4,mdct=4):
    if region is None:
        region = np.ones_like(I)
    I[region == 0] = 0
    Im       = cv2.resize(I,(Ndct,Mdct))
    FIm      = dct(Im)
    Y        = FIm[0:int(Ndct/2),0:int(Mdct/2)]
    x        = np.abs(Y)
    F        = cv2.resize(x,(ndct,mdct))
    X        = np.reshape(F,(ndct*mdct,))
    return X

def clp(img):

    # indices of the pixels of the 8 profiles of CLP (each profile is a line of 32 pixels)
    # C[2*k,:]  : coordinate i of profile k, for k=0...7
    # C[2*k+1,:]: coordinate j of profile k, for k=0...7
    C = np.array([
        [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [8,8,9,9,10,10,11,11,12,12,13,14,14,15,15,16,16,17,17,18,18,19,20,20,21,21,22,22,23,23,24,25],
        [8,8,9,9,10,10,11,11,12,12,13,14,14,15,15,16,16,17,17,18,18,19,20,20,21,21,22,22,23,23,24,25],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        [25,24,23,23,22,22,21,21,20,20,19,18,18,17,17,16,16,15,15,14,14,13,12,12,11,11,10,10,9,9,8,8],
        [25,24,23,23,22,22,21,21,20,20,19,18,18,17,17,16,16,15,15,14,14,13,12,12,11,11,10,10,9,9,8,8],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]])


    ng = 32
    X = np.array(np.zeros((8,ng)))
    I = cv2.resize(img, (ng, ng))
    
    for i in range(8):
        k = i*2
        for j in range(ng):
            X[i,j] = I[C[k,j],C[k+1,j]]*255
        #plt.plot(range(32),X[i,:])
    #plt.show()
    d = np.abs(X[:,0]-X[0:,31])
    k = np.argmin(d)
    y = X[k,:]
    #plt.plot(range(32),y)
    #plt.show()
    Po  = y/y[0]
    Q   = rampefree(Po)
    Qm  = np.average(Q)
    Qd  = np.max(Q)-np.min(Q)
    Qd1 = np.log(Qd+1)
    Qd2 = 2*Qd/(Po[0]+Po[ng-1])
    Qs  = np.std(Q)
    Qf  = np.abs(np.fft.fft(Q))
    Qf  = Qf[range(1,8)]
    X = [Qm, Qs, Qd, Qd1, Qd2, Qf[0], Qf[1], Qf[2], Qf[3], Qf[4], Qf[5], Qf[6]]
    return X


def rampefree(x):
    k = len(x)-1
    m = (x[k]-x[0])/k
    b = x[0]
    y = x - range(k+1)*m - b
    return y

def contrast(img,region=None,neihbors=2):
    img = img*255
    if region is None:
        region = np.ones_like(img)
    
    R = region==1
    Rn = R
    for i in range(neihbors):
        Rn = imdilate(Rn)

    Rn = np.bitwise_and(Rn,~R)

    if np.max(Rn)==1:
        Ir = img*region
        In = img*Rn
        MeanGr = np.average(Ir)
        MeanGn = np.average(In)
        K1 = (MeanGr-MeanGn)/MeanGn            # contrast after Kamm, 1999
        K2 = (MeanGr-MeanGn)/(MeanGr+MeanGn)   # modulation after Kamm, 1999
        K3 = np.log(MeanGr/MeanGn)                # film-contrast after Kamm, 1999
    else:
        K1 = -1        
        K2 = -1        
        K3 = -1

    (nI,mI) = img.shape

    n1 = int(round(nI/2)+1)
    m1 = int(round(mI/2)+1)

    P1 = img[n1,:]    # Profile in i-Direction
    P2 = img[:,m1]    # Profile in j-Direction
    Q1 = rampefree(P1)
    Q2 = rampefree(P2)
    Q = np.concatenate((Q1,Q2))

    Ks = np.std(Q)
    K  = np.log(np.max(Q)-np.min(Q)+1)
        
    X  = [K1,K2,K3,Ks,K]
    return X


