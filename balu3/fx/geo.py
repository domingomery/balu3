import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
from   scipy.ndimage import binary_fill_holes
from   skimage.segmentation import find_boundaries




def basicgeo(R,names=False):
  # center of mass
  ij = np.argwhere(R)
  ii = ij[:,0]
  jj = ij[:,1]
  i_m = np.mean(ii)
  j_m = np.mean(jj)

  # height
  h = np.max(ii)-np.min(ii)+1
  
  # width
  w = np.max(jj)-np.min(jj)+1

  # area
  area = np.sum(R)

  # perimeter
  # https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.find_boundaries
  B = skimage.segmentation.find_boundaries(R,mode='inner')
  perimeter = np.sum(B)


  # regionprops
  # https://scikit-image.org/docs/stable/api/skimage.measure.html#regionprops
  props        = skimage.measure.regionprops(R)[0]

  #perimeter    = props.perimeter
  roundness    = 4*area*np.pi/perimeter/perimeter
  solidity     = props.solidity
  euler_number = props.euler_number
  eq_diameter  = props.equivalent_diameter_area
  axis_major   = props.axis_major_length
  axis_minor   = props.axis_minor_length
  orientation  = props.orientation
  extent       = props.extent
  eccentricity = props.eccentricity
  area_convex  = props.area_convex

  # roundness
  #roundness    = 4*area*np.pi/perimeter/perimeter

  X = [i_m,j_m,h,w,area,perimeter,roundness,
       euler_number,eq_diameter,axis_major,axis_minor,
       orientation,solidity,extent,
       eccentricity,area_convex]
  if names:
    Xn = ['i_m','j_m','height','width','area','perimeter','roundness',
       'euler_number','equivalent_diameter','major_axis','minor_axis',
       'orientation','solidity','extent',
       'eccentricity','convex_area_convex']
    return X,Xn
  else:
    return X




def hugeo(R,names=False):
  # regionprops
  # https://scikit-image.org/docs/stable/api/skimage.measure.html#regionprops
  props = skimage.measure.regionprops(R)[0]

  X = props.moments_hu

  if names:
    Xn = ['Hu-moment-1','Hu-moment-2','Hu-moment-3','Hu-moment-4','Hu-moment-5','Hu-moment-6','Hu-moment-7']
    return X,Xn
  else:
    return X

def flusser(R,names=False):
  moments = cv2.moments(R, True)
  u00,u20, u11, u02, u30, u21, u12, u03 = moments['m00'], moments['mu20'], moments['mu11'], moments['mu02'], moments['mu30'], moments['mu21'], moments['mu12'], moments['mu03']
  I1 = (u20*u02-u11**2)/u00**4 ;
  I2 = (u30**2*u03**2-6*u30*u21*u12*u03+4*u30*u12**3+4*u21**3*u03-3*u21**2*u12**2)/u00**10;
  I3 = (u20*(u21*u03-u12**2)-u11*(u30*u03-u21*u12)+u02*(u30*u12-u21**2))/u00**7;
  I4 = (u20**3*u03**2-6*u20**2*u11*u12*u03-6*u20**2*u02*u21*u03+9*u20**2*u02*u12**2 + 12*u20*u11**2*u21*u03+6*u20*u11*u02*u30*u03-18*u20*u11*u02*u21*u12-8*u11**3*u30*u03- 6*u20*u02**2*u30*u12+9*u20*u02**2*u21+12*u11**2*u02*u30*u12-6*u11*u02**2*u30*u21+u02**3*u30**2)/u00**11;
  X = [I1,I2,I3,I4]
  if names:
    Xn = ['Flusser-1','Flusser-2','Flusser-3','Flusser-4']
    return X,Xn
  else:
    return X



def efd_descriptors(contour, order=10, normalize=True):
    # From https://github.com/hbldh/pyefd
  
    """Calculate elliptical Fourier descriptors for a contour.
    :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
    :param int order: The order of Fourier coefficients to calculate.
    :param bool normalize: If the coefficients should be normalized;
        see references for details.
    :param bool return_transformation: If the normalization parametres should be returned.
        Default is ``False``.
    :return: A ``[order x 4]`` array of Fourier coefficients and optionally the
        transformation parametres ``scale``, ``psi_1`` (rotation) and ``theta_1`` (phase)
    :rtype: ::py:class:`numpy.ndarray` or (:py:class:`numpy.ndarray`, (float, float, float))
    """
    dxy = np.diff(contour, axis=1)
    dxy = dxy.reshape(dxy.shape[1],2)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    t = np.concatenate([([0.0]), np.cumsum(dt)])
    T = t[-1]

    phi = (2 * np.pi * t) / T

    orders = np.arange(1, order + 1)
    consts = T / (2 * orders * orders * np.pi * np.pi)
    phi = phi * orders.reshape((order, -1))

    d_cos_phi = np.cos(phi[:, 1:]) - np.cos(phi[:, :-1])
    d_sin_phi = np.sin(phi[:, 1:]) - np.sin(phi[:, :-1])

    a = consts * np.sum((dxy[:, 0] / dt) * d_cos_phi, axis=1)
    b = consts * np.sum((dxy[:, 0] / dt) * d_sin_phi, axis=1)
    c = consts * np.sum((dxy[:, 1] / dt) * d_cos_phi, axis=1)
    d = consts * np.sum((dxy[:, 1] / dt) * d_sin_phi, axis=1)

    coeffs = np.concatenate(
        [
            a.reshape((order, 1)),
            b.reshape((order, 1)),
            c.reshape((order, 1)),
            d.reshape((order, 1)),
        ],
        axis=1,
    )

    if normalize:
      theta_1 = 0.5 * np.arctan2(
        2 * ((coeffs[0, 0] * coeffs[0, 1]) + (coeffs[0, 2] * coeffs[0, 3])),
        (
            (coeffs[0, 0] ** 2)
            - (coeffs[0, 1] ** 2)
            + (coeffs[0, 2] ** 2)
            - (coeffs[0, 3] ** 2)
        ),
      )
      # Rotate all coefficients by theta_1.
      for n in range(1, coeffs.shape[0] + 1):
        coeffs[n - 1, :] = np.dot(
            np.array(
                [
                    [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                    [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                ]
            ),
            np.array(
                [
                    [np.cos(n * theta_1), -np.sin(n * theta_1)],
                    [np.sin(n * theta_1), np.cos(n * theta_1)],
                ]
            ),
        ).flatten()

      # Make the coefficients rotation invariant by rotating so that
      # the semi-major axis is parallel to the x-axis.
      psi_1 = np.arctan2(coeffs[0, 2], coeffs[0, 0])
      psi_rotation_matrix = np.array(
        [[np.cos(psi_1), np.sin(psi_1)], [-np.sin(psi_1), np.cos(psi_1)]]
      )
      # Rotate all coefficients by -psi_1.
      for n in range(1, coeffs.shape[0] + 1):
        coeffs[n - 1, :] = psi_rotation_matrix.dot(
            np.array(
                [
                    [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                    [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                ]
            )
        ).flatten()

      size = coeffs[0, 0]
      # Obtain size-invariance by normalizing.
      coeffs /= np.abs(size)
    X = coeffs.reshape(4*order,)

    return X

def efourierdes(R, order=10, names=False):
  # Elliptic Fourier Descriptors from https://github.com/hbldh/pyefd
  R8h    = binary_fill_holes(R).astype(np.uint8)
  contours, hierarchy = cv2.findContours(R8h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  X = efd_descriptors(contours, order=order)

  if names==True:
    Xn = []
    for k in range(order):
      Xn.append('Fourierdes-a-'+str(k))
      Xn.append('Fourierdes-b-'+str(k))
      Xn.append('Fourierdes-c-'+str(k))
      Xn.append('Fourierdes-d-'+str(k))
    return X,Xn
  else:
    return X

from   skimage.segmentation import find_boundaries
from   scipy.ndimage import binary_fill_holes


def fourierdes(R, n_des=16, names=False):

    R8h    = binary_fill_holes(R).astype(np.uint8)
    contour, hierarchy = cv2.findContours(R8h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = np.array(contour)
    B = x.reshape(x.shape[1],2)
    V = B[:, 0] + 1j * B[:, 1]
    m = B.shape[0]

    r = np.zeros(m, dtype=complex)
    phi = np.zeros(m)
    dphi = np.zeros(m)
    l = np.zeros(m)
    dl = np.zeros(m)

    r[0] = V[0] - V[m-1]
    r[1:] = V[1:] - V[:m-1]

    dl = np.abs(r)
    phi = np.angle(r)

    dphi[:m-1] = np.mod(phi[1:] - phi[:m-1] + np.pi, 2 * np.pi) - np.pi
    dphi[m-1] = np.mod(phi[0] - phi[m-1] + np.pi, 2 * np.pi) - np.pi

    l[0] = dl[0]
    for k in range(1, m):
        l[k] = l[k-1] + dl[k]

    l = l * (2 * np.pi / l[m-1])
    descriptors = np.zeros(n_des)

    for n in range(1, n_des + 1):
        an = (dphi * np.sin(l * n)).sum()
        bn = (dphi * np.cos(l * n)).sum()
        an = -an / n / np.pi
        bn = bn / n / np.pi
        imagi = an + 1j * bn
        descriptors[n-1] = np.abs(imagi)

    X = descriptors

    if names:
        return np.array([f'Fourier-des {n+1:>2d}' for n in range(n_des)]), descriptors
    return X




def fit_ellipse(x,y):
    # Fitzgibbon, A.W., Pilu, M., and Fischer R.B., 
    # Direct least squares fitting of ellipses, 1996
    x        = x[:,None]
    y        = y[:,None]
    D        = np.hstack([x*x,x*y,y*y,x,y,np.ones(x.shape)])
    S        = np.dot(D.T,D)
    C        = np.zeros([6,6])
    C[0,2]   = C[2,0] = 2
    C[1,1]   = -1
    E,V      = np.linalg.eig(np.dot(np.linalg.inv(S),C))
    n        = np.argmax(E)
    s        = V[:,n]
    a        = s[0]
    b        = s[1]/2.
    c        = s[2]
    d        = s[3]/2.
    f        = s[4]/2.
    g        = s[5]
    dd       = b*b-a*c
    cx       = (c*d-b*f)/dd
    cy       = (a*f-b*d)/dd
    alpha    = 0.5*np.arctan(2*b/(a-c))*180/np.pi
    up       = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1    = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2    = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    a        = np.sqrt(abs(up/down1))
    b        = np.sqrt(abs(up/down2))
    area     = np.pi*a*b

    if b>a:
        ecc  = a/b
    else:
        ecc  = b/a

    features = [cx,cy,a,b,alpha,ecc,area]

    return features

def fitellipse(R,names=False):
    E        = find_boundaries(R, mode='outer').astype(np.uint8)
    # E        = bwperim(R)
    data     = np.argwhere(E==True)
    y        = data[:,0]
    x        = data[:,1]
    X       = fit_ellipse(x,y)
    if names:
       Xn = ['cx','cy','a','b','alpha','ecc','area']
       return X,Xn
    else:
       return X
    return features

def gupta(R, names=False):

    B = skimage.segmentation.find_boundaries(R,mode='inner')
    i_perim, j_perim = np.where(B.astype(bool))
    im_perim = i_perim + j_perim * 1j
    ix = i_perim.mean()
    jx = j_perim.mean()
    centre = ix + jx * 1j
    z = np.abs(im_perim - centre)
    m1 = z.mean()

    mur1 = z - m1
    mur2 = mur1 * mur1
    mur3 = mur1 * mur2
    mur4 = mur2 * mur2

    mu2 = mur2.mean()
    mu3 = mur3.mean()
    mu4 = mur4.mean()

    F1 = (mu2 ** .5) / m1
    F2 = mu3 / (mu2 * (mu2 ** .5))
    F3 = mu4 / mu2 ** 2

    X = np.array([F1, F2, F3])

    if names:
      Xn = ['Gupta-1','Gupta-2','Gupta-3']
      return X,Xn
    else:
      return X
