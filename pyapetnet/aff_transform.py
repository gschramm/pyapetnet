import numpy as np
import math

from numba import njit, prange

#-------------------------------------------------------------------------------
@njit(parallel = True)
def aff_transform(volume, aff_mat, output_shape, trilin = True, cval = 0., os0 = 1, os1 = 1, os2 = 1):
  """ Affine transformation of a 3D volume in parallel (using numba's njit).

  Parameters
  ----------
  volume : a 3D numpy array
    contaning an image volume

  aff_mat : a 4x4 2D numpy array 
    affine transformation matrix

  output_shape : 3 element tuple 
    containing the shape of the output volume
  
  trilin : bool, optional
    whether to use trilinear interpolation (default True)

  cval : float, optional
    value of "outside" voxels (default 0)  

  os0, os1, os2 : int 
    oversampling factors along the 0, 1, and 2 axis (default 1 -> no oversampling)

  Returns
  -------
  3D numpy array
    a transformed volume

  Note
  ----
  The purpose of the function is to reproduce the results
  of scipy.ndimage.affine_transform (with order = 1 and prefilter = False) 
  in a faster way on a multi CPU system.
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
    for i in range(n0):
      for j in range(n1):
        for k in range(n2):
          for ii in range(os0):
            for jj in range(os1):
              for kk in range(os2):
                output_volume[i,j,k] += os_output_volume[i*os0 + ii, j*os1 + jj, k*os2 + kk] / (os0*os1*os2)

  return output_volume

#----------------------------------------------------------------------

def kul_aff(params, origin = None):
  """ KUL affine transformation matrix

  Parameters
  ----------

   params : 6 element numpy array
     containing 3 translations and 3 rotation angles.
     The definition of the rotations is the following (purely historical):
       parms[3] ... rotation around 001 axis
       parms[4] ... rotation around 100 axis
       parms[5] ... rotation around 010 axis
     The order of the rotations is first 010, second 100, third 001
   
   origin : 3 element numpy array, optional
     containing the origin for the rotations (rotation center)
     The default None means origin = [0,0,0]
  """
  
  if len(params) > 3:
    t001 = params[3]
    t100 = params[4]
    t010 = params[5]
  else:
    t001 = 0 
    t100 = 0
    t010 = 0

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
  
  TR = np.eye(4) 
  TR[0,-1] = params[0]
  TR[1,-1] = params[1]
  TR[2,-1] = params[2]
  R = R @ TR
  
  return R
