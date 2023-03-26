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


def efd_descriptors(contour, order=10, normalize=True):
  
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

def fourierdes(R, order=10, names=False):
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
