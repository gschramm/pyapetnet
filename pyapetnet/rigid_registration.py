import numpy as np
import math

from numba import njit, prange

#-------------------------------------------------------------------------------
@njit(parallel = True)
def jointhisto_3dvols(x, y, nbins = 40, normalize = True):
  """
  Calculate the joint histogram between two 3d volumes.
  E.g. useful for rigid registration of two 3d volumes with 
  using mutual information.
  
  The implementaion is optimized for a multi cpu system using
  numba.

  The left edges of the bins are at np.arange(nbins) * binwidth + jmin
  where jmin is the joint minimum between x and y and 
  binwidth = (jmax - jmin) / (nbins - 1).

  Inputs
  ------

  x ... first 3d numpy array
  
  y ... second 3d numpy arry

  
  Keyword arguments
  -----------------

  nbins     ... number of bins in the joint histogram (default 40)

  normalize ... divide the counts in each bin by the number of data points (default True)

  Output
  ------

  a 2d numpy array of shape (nbins, nbins) containing the joint histogram
  """

  xmin = x.min()
  xmax = x.max()

  ymin = y.min()
  ymax = y.max()
  
  xbinwidth = (xmax - xmin) / (nbins - 1)
  ybinwidth = (ymax - ymin) / (nbins - 1)

  n0, n1, n2 = x.shape
  ntot       = n0*n1*n2

  # we create n0 temporary 2d histograms to avoid race conditions
  tmp    = np.zeros((n0, nbins, nbins), dtype = np.uint64)
  # the dtype of the joint histo is float64 because we might want to normalize it
  jhisto = np.zeros((nbins, nbins), dtype = np.float64)

  for i in prange(n0):
    for j in range(n1):
      for k in range(n2):
        tmp[i,math.floor((x[i,j,k]-xmin)/xbinwidth),math.floor((y[i,j,k]-ymin)/ybinwidth)] += 1

  for j in prange(nbins):
    for k in range(nbins):
      for i in range(n0):
        if normalize:
          jhisto[j,k] += tmp[i,j,k] / ntot
        else:
          jhisto[j,k] += tmp[i,j,k]

  return jhisto

#-------------------------------------------------------------------------------
@njit(parallel = True)
def aff_transform(volume, aff_mat, output_shape, trilin = True, cval = 0., os0 = 1, os1 = 1, os2 = 1):
  """
  Calculate an affine transformation of a 3D volume
  in parallel (using numba's njit).

  The purpose of the function is to reproduce the results
  of scipy.ndimage.affine_transform (with order = 1 and prefilter = False) 
  in a faster way on a multi CPU system.

  Inputs
  ------

  volume       ... a 3D numpy array

  aff_mat      ... a 4x4 affine transformation matrix

  output_shape ... a tuple containing the shape of the output volume
  
  Keyword arguments
  -----------------

  trilin      ... (bool) whether to use trilinear interpolation (default True)

  cval        ... (float) value of "outside" voxels (default 0)  

  os0,os1,os2 ... oversampling factors along the 0, 1, and 2 axis (default 1 -> no oversampling)

  Returns
  -------

  A transformed volume (3d numpy array)
  """

  # the dimensions of the output volume
  n0, n1, n2          = output_shape

  # the dimenstion of the input volume
  n0_in, n1_in, n2_in = volume.shape

  # the sizes of the temporary oversampled array
  # the oversampling is needed in case we go from
  # small voxels to big voxels
  n0_os = n0*os0
  n1_os = n1*os1
  n2_os = n2*os2

  os_output_volume = np.zeros((n0_os, n1_os, n2_os))
  if cval != 0: os_output_volume += cval

  for i in prange(n0_os):
    for j in range(n1_os):
      for k in range(n2_os):
        tmp_x = aff_mat[0,0]*i/os0 + aff_mat[0,1]*j/os1 + aff_mat[0,2]*k/os2 + aff_mat[0,3]
        tmp_y = aff_mat[1,0]*i/os0 + aff_mat[1,1]*j/os1 + aff_mat[1,2]*k/os2 + aff_mat[1,3]
        tmp_z = aff_mat[2,0]*i/os0 + aff_mat[2,1]*j/os1 + aff_mat[2,2]*k/os2 + aff_mat[2,3]

        if trilin:
          # trilinear interpolatio mode
          # https://en.wikipedia.org/wiki/Trilinear_interpolation
          x0 = math.floor(tmp_x)  
          x1 = math.ceil(tmp_x)  
          y0 = math.floor(tmp_y)  
          y1 = math.ceil(tmp_y)  
          z0 = math.floor(tmp_z)  
          z1 = math.ceil(tmp_z)  

          if (x0 >= 0) and (x1 < n0_in) and (y0 >= 0) and (y1 < n1_in) and (z0 >= 0) and (z1 < n2_in):
            xd = (tmp_x - x0)
            yd = (tmp_y - y0)
            zd = (tmp_z - z0)

            c00 = volume[x0,y0,z0]*(1 - xd) + volume[x1,y0,z0]*xd
            c01 = volume[x0,y0,z1]*(1 - xd) + volume[x1,y0,z1]*xd
            c10 = volume[x0,y1,z0]*(1 - xd) + volume[x1,y1,z0]*xd
            c11 = volume[x0,y1,z1]*(1 - xd) + volume[x1,y1,z1]*xd

            c0 = c00*(1 - yd) + c10*yd
            c1 = c01*(1 - yd) + c11*yd
 
            os_output_volume[i,j,k] = c0*(1 - zd) + c1*zd

        else:
          # no interpolation mode
          x = round(tmp_x)
          y = round(tmp_y)
          z = round(tmp_z)

          if ((x >= 0) and (x < n0_in) and (y >= 0) and (y < n1_in) and (z >= 0) and (z < n2_in)):
            os_output_volume[i,j,k] = volume[x,y,z]

  if os0 == 1 and os1 == 1 and os2 == 1:
    # case that were was no oversampling
    output_volume = os_output_volume
  else:
    output_volume = np.zeros((n0, n1, n2))
    # case with oversampling, we have to average neighbors
    for i in prange(n0):
      for j in range(n1):
        for k in range(n2):
          for ii in range(os0):
            for jj in range(os1):
              for kk in range(os2):
                output_volume[i,j,k] += os_output_volume[i*os0 + ii, j*os1 + jj, k*os2 + kk] / (os0*os1*os2)

  return output_volume

#----------------------------------------------------------------------


def aff(params, origin = None):
  
  t001 = params[3]
  t100 = params[4]
  t010 = params[5]

  # set up matrix for rotation around 001 axis
  P001 = np.zeros((3,3))
  P001[0,0] =  math.cos(t001)
  P001[1,1] =  math.cos(t001)
  P001[0,1] = -math.sin(t001)
  P001[1,0] =  math.sin(t001)
  P001[2,2] = 1

  P100 = np.zeros((3,3))
  P100[1,1] =  math.cos(t100)
  P100[2,2] =  math.cos(t100)
  P100[1,2] = -math.sin(t100)
  P100[2,1] =  math.sin(t100)
  P100[0,0] = 1

  P010 = np.zeros((3,3))
  P010[0,0] =  math.cos(t010)
  P010[2,2] =  math.cos(t010)
  P010[0,2] =  math.sin(t010)
  P010[2,0] = -math.sin(t010)
  P010[1,1] = 1

  R = np.eye(4)
  R[:3,:3] = (P001 @ P100 @ P010)

  if origin is not None:
    T = np.eye(4) 
    T[:-1,-1] -= origin
    R = np.linalg.inv(T) @ (R @ T)
  
  if len(params) == 6:
    TR = np.eye(4) 
    TR[0,-1] = params[0]
    TR[1,-1] = params[1]
    TR[2,-1] = params[2]
    R = R @ TR
  
  return R
  
#---------------------------------------------------------------- 
def neg_mutual_information(x, y, nbins = 40, norm = True):
  p_xy  = jointhisto_3dvols(x, y, nbins = nbins, normalize = True)
  ixy   = np.where(p_xy > 0) 

  # calculate the outer product of the marginal distributions
  p_x     = p_xy.sum(axis = 1)
  p_y     = p_xy.sum(axis = 0)

  if norm:
    # normalized mututal information 
    # Studholme et al: Pattern Recognition 32 (1999) 71 86
    # has 1:1 corespondance to ECC defined by Maes et al

    ix = np.where(p_x > 0)    
    iy = np.where(p_y > 0)    

    mi = -(((p_x[ix]*np.log(p_x[ix])).sum() +  (p_y[iy]*np.log(p_y[iy])).sum()) / 
           (p_xy[ixy]*np.log(p_xy[ixy])).sum())

  else:
    # conventional mutual information
    # Maes et al: IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 16, NO. 2, APRIL 1997
    p_x_p_y = np.outer(p_x, p_y)
    mi = -(p_xy[ixy] * np.log(p_xy[ixy]/p_x_p_y[ixy])).sum()
 
  return mi

#---------------------------------------------------------------- 
def squared_euclidean_dist(x,y):
  return np.sqrt(((x-y)**2).sum())

#---------------------------------------------------------------- 
def regis_cost_func(params, img_fix, img_float, verbose = False,
                    rotate = True, metric = neg_mutual_information):

  if rotate:
    af  = aff(params, origin = np.array(img_float.shape)/2)
    m   = metric(img_fix, aff_transform(img_float, af, img_float.shape))
  else:
    af = np.eye(4)
    af[:-1,-1] = params
    m  = metric(img_fix, aff_transform(img_float, af, img_float.shape))

  if verbose:
    print(params)
    print(m)
    print('')

  return m

#-------------------------------------------------------------------------------
@njit(parallel = True)
def backward_3d_warp(volume, d0, d1, d2, trilin = True, cval = 0., os0 = 1, os1 = 1, os2 = 1):
  """
  Calculate an affine transformation of a 3D volume
  in parallel (using numba's njit).

  The purpose of the function is to reproduce the results
  of scipy.ndimage.affine_transform (with order = 1 and prefilter = False) 
  in a faster way on a multi CPU system.

  Inputs
  ------

  volume       ... a 3D numpy array containing the image (volume)

  d0, d1, d2   ... the 3 3D numpy arrays that contain the 3 components of the deformation field 

  
  Keyword arguments
  -----------------

  trilin      ... (bool) whether to use trilinear interpolation (default True)

  Returns
  -------

  A transformed volume (3d numpy array)
  """
  # the dimensions of the output volume
  n0, n1, n2          = volume.shape

  # the dimenstion of the input volume
  n0_in, n1_in, n2_in = volume.shape

  # the sizes of the temporary oversampled array
  # the oversampling is needed in case we go from
  # small voxels to big voxels
  n0_os = n0*os0
  n1_os = n1*os1
  n2_os = n2*os2

  os_output_volume = np.zeros((n0_os, n1_os, n2_os))
  if cval != 0: os_output_volume += cval

  for i in prange(n0_os):
    for j in range(n1_os):
      for k in range(n2_os):
        tmp_x = i - d0[i,j,k] 
        tmp_y = j - d1[i,j,k]
        tmp_z = k - d2[i,j,k]

        if trilin:
          # trilinear interpolatio mode
          # https://en.wikipedia.org/wiki/Trilinear_interpolation
          x0 = math.floor(tmp_x)  
          x1 = math.ceil(tmp_x)  
          y0 = math.floor(tmp_y)  
          y1 = math.ceil(tmp_y)  
          z0 = math.floor(tmp_z)  
          z1 = math.ceil(tmp_z)  

          if (x0 >= 0) and (x1 < n0_in) and (y0 >= 0) and (y1 < n1_in) and (z0 >= 0) and (z1 < n2_in):
            xd = (tmp_x - x0)
            yd = (tmp_y - y0)
            zd = (tmp_z - z0)

            c00 = volume[x0,y0,z0]*(1 - xd) + volume[x1,y0,z0]*xd
            c01 = volume[x0,y0,z1]*(1 - xd) + volume[x1,y0,z1]*xd
            c10 = volume[x0,y1,z0]*(1 - xd) + volume[x1,y1,z0]*xd
            c11 = volume[x0,y1,z1]*(1 - xd) + volume[x1,y1,z1]*xd

            c0 = c00*(1 - yd) + c10*yd
            c1 = c01*(1 - yd) + c11*yd
 
            os_output_volume[i,j,k] = c0*(1 - zd) + c1*zd

        else:
          # no interpolation mode
          x = round(tmp_x)
          y = round(tmp_y)
          z = round(tmp_z)

          if ((x >= 0) and (x < n0_in) and (y >= 0) and (y < n1_in) and (z >= 0) and (z < n2_in)):
            os_output_volume[i,j,k] = volume[x,y,z]

  if os0 == 1 and os1 == 1 and os2 == 1:
    # case that were was no oversampling
    output_volume = os_output_volume
  else:
    output_volume = np.zeros((n0, n1, n2))
    # case with oversampling, we have to average neighbors
    for i in prange(n0):
      for j in range(n1):
        for k in range(n2):
          for ii in range(os0):
            for jj in range(os1):
              for kk in range(os2):
                output_volume[i,j,k] += os_output_volume[i*os0 + ii, j*os1 + jj, k*os2 + kk] / (os0*os1*os2)

  return output_volume
