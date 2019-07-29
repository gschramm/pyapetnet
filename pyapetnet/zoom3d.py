import numpy as np
import math

from numba import prange, njit

#---------------------------------------------------------------------------
@njit(parallel = True)
def upsample_3d_0(array, zoom, cval = 0):
  """Upsample a 3D array in the 0 (left-most) direction 

  Parameters
  ----------
  array : 3D numpy array
    array to be upsampled

  zoom  : float > 1
    zoom factor. a zoom factors of 2 means that the shape will double

  cval : float, optional
    constant value used for background 

  Returns
  -------
  3D numpy array
    the upsampled array
  """
  delta = 1./zoom

  # number of elements in array with big voxels
  nb,n1,n2 = array.shape

  # number of elements in arrray with small voxels
  ns = math.ceil(nb/delta)

  new_array = np.zeros((ns,n1,n2))
  if cval != 0: new_array += cval
 

  for j in prange(n1):
    for i in range(ns):
      for k in range(n2):
        # calculate the bin center of the small voxel array
        # in coordinates in the big voxel array
        ib_c = delta*(i - 0.5*ns + 0.5) + 0.5*nb - 0.5

        fl_l = math.floor(ib_c)
        fl_r = fl_l + 1

        if (fl_l >= 0):  
          a = array[fl_l,j,k] * (fl_r - ib_c)
        else:
          a = cval

        if (fl_r < nb): 
          b = array[fl_r,j,k] * (ib_c - fl_l)
        else:
          b = cval

        new_array[i,j,k] = a + b

  return new_array

#---------------------------------------------------------------------------
@njit(parallel = True)
def upsample_3d_1(array, zoom, cval = 0):
  """Upsample a 3D array in the 1 (middle) direction 

  Parameters
  ----------
  array : 3D numpy array
    array to be upsampled

  zoom  : float > 1
    zoom factor. a zoom factors of 2 means that the shape will double

  cval : float, optional
    constant value used for background 

  Returns
  -------
  3D numpy array
    the upsampled array
  """
  delta = 1./zoom

  # number of elements in array with big voxels
  n0,nb,n2 = array.shape

  # number of elements in arrray with small voxels
  ns = math.ceil(nb/delta)

  new_array = np.zeros((n0,ns,n2))
  if cval != 0: new_array += cval

  for i in prange(n0):
    for j in range(ns):
      for k in range(n2):
        # calculate the bin center of the small voxel array
        # in coordinates in the big voxel array
        ib_c = delta*(j - 0.5*ns + 0.5) + 0.5*nb - 0.5

        fl_l = math.floor(ib_c)
        fl_r = fl_l + 1

        if (fl_l >= 0):  
          a = array[i,fl_l,k] * (fl_r - ib_c)
        else:
          a = cval

        if (fl_r < nb): 
          b = array[i,fl_r,k] * (ib_c - fl_l)
        else:
          b = cval

        new_array[i,j,k] = a + b

  return new_array

#---------------------------------------------------------------------------
@njit(parallel = True)
def upsample_3d_2(array, zoom, cval = 0):
  """Upsample a 3D array in the 2 (right-most) direction 

  Parameters
  ----------
  array : 3D numpy array
    array to be upsampled

  zoom  : float > 1
    zoom factor. a zoom factors of 2 means that the shape will double

  cval : float, optional
    constant value used for background 

  Returns
  -------
  3D numpy array
    the upsampled array
  """
  delta = 1./zoom

  # number of elements in array with big voxels
  n0,n1,nb = array.shape

  # number of elements in arrray with small voxels
  ns = math.ceil(nb/delta)

  new_array = np.zeros((n0,n1,ns))
  if cval != 0: new_array += cval

  for i in prange(n0):
    for j in range(n1):
      for k in range(ns):
        # calculate the bin center of the small voxel array
        # in coordinates in the big voxel array
        ib_c = delta*(k - 0.5*ns + 0.5) + 0.5*nb - 0.5

        fl_l = math.floor(ib_c)
        fl_r = fl_l + 1

        if (fl_l >= 0):  
          a = array[i,j,fl_l] * (fl_r - ib_c)
        else:
          a = cval

        if (fl_r < nb): 
          b = array[i,j,fl_r] * (ib_c - fl_l)
        else:
          b = cval

        new_array[i,j,k] = a + b

  return new_array



#---------------------------------------------------------------------------
@njit(parallel = True)
def downsample_3d_0(array, zoom, cval = 0):
  """Downsample a 3D array in the 0 (left-most) direction 

  Parameters
  ----------
  array : 3D numpy array
    array to be upsampled

  zoom  : float < 1
    zoom factor. a zoom factors of 0.5 means that the shape will reduced by a factor of 2

  cval : float, optional
    constant value used for background 

  Returns
  -------
  3D numpy array
    the upsampled array
  """
  delta = 1./zoom

  # number of elements in array with small voxels
  ns,n1,n2 = array.shape

  # number of elements in arrray with big voxels
  nb = math.ceil(ns/delta)

  new_array = np.zeros((nb,n1,n2))
  if cval != 0: new_array += cval

  for j in prange(n1):
    for i in range(ns):
      for k in range(n2):
        ib_l = (1./delta) *(i     - 0.5*ns) + 0.5*nb
        ib_r = (1./delta) *(i + 1 - 0.5*ns) + 0.5*nb

        left_bin  = math.floor(ib_l)
        right_bin =  math.ceil(ib_r) - 1

        if left_bin == right_bin:
          new_array[left_bin,j,k] += array[i,j,k] / delta
        else:
          c = math.ceil(ib_l)
          new_array[left_bin,j,k]  += array[i,j,k]*(c - ib_l)
          new_array[right_bin,j,k] += array[i,j,k]*(ib_r - c)

  return new_array

#---------------------------------------------------------------------------
@njit(parallel = True)
def downsample_3d_1(array, zoom, cval = 0):
  """Downsample a 3D array in the 1 (middle) direction 

  Parameters
  ----------
  array : 3D numpy array
    array to be upsampled

  zoom  : float < 1
    zoom factor. a zoom factors of 0.5 means that the shape will reduced by a factor of 2

  cval : float, optional
    constant value used for background 

  Returns
  -------
  3D numpy array
    the upsampled array
  """
  delta = 1./zoom

  # number of elements in array with small voxels
  n0,ns,n2 = array.shape

  # number of elements in arrray with big voxels
  nb = math.ceil(ns/delta)

  new_array = np.zeros((n0,nb,n2))
  if cval != 0: new_array += cval

  for i in prange(n0):
    for j in range(ns):
      for k in range(n2):
        ib_l = (1./delta) *(j     - 0.5*ns) + 0.5*nb
        ib_r = (1./delta) *(j + 1 - 0.5*ns) + 0.5*nb

        left_bin  = math.floor(ib_l)
        right_bin =  math.ceil(ib_r) - 1

        if left_bin == right_bin:
          new_array[i,left_bin,k] += array[i,j,k] / delta
        else:
          c = math.ceil(ib_l)
          new_array[i,left_bin,k]  += array[i,j,k]*(c - ib_l)
          new_array[i,right_bin,k] += array[i,j,k]*(ib_r - c)

  return new_array

#---------------------------------------------------------------------------
@njit(parallel = True)
def downsample_3d_2(array, zoom, cval = 0):
  """Downsample a 3D array in the 2 (right-most) direction 

  Parameters
  ----------
  array : 3D numpy array
    array to be upsampled

  zoom  : float < 1
    zoom factor. a zoom factors of 0.5 means that the shape will reduced by a factor of 2

  cval : float, optional
    constant value used for background 

  Returns
  -------
  3D numpy array
    the upsampled array
  """
  delta = 1./zoom

  # number of elements in array with small voxels
  n0,n1,ns = array.shape

  # number of elements in arrray with big voxels
  nb = math.ceil(ns/delta)

  new_array = np.zeros((n0,n1,nb))
  if cval != 0: new_array += cval

  for i in prange(n0):
    for j in range(n1):
      for k in range(ns):
        ib_l = (1./delta) *(k     - 0.5*ns) + 0.5*nb
        ib_r = (1./delta) *(k + 1 - 0.5*ns) + 0.5*nb

        left_bin  = math.floor(ib_l)
        right_bin =  math.ceil(ib_r) - 1

        if left_bin == right_bin:
          new_array[i,j,left_bin] += array[i,j,k] / delta
        else:
          c = math.ceil(ib_l)
          new_array[i,j,left_bin]  += array[i,j,k]*(c - ib_l)
          new_array[i,j,right_bin] += array[i,j,k]*(ib_r - c)

  return new_array

#---------------------------------------------------------------------------

def zoom3d(vol, zoom, cval = 0):
  """Zoom (upsample or downsample) a 3d array along all axis.

  Parameters
  ----------
  vol :  3d numpy array
    volume to be zoomed

  zoom : float  or 3 element tuple/list/array of floats 
    the zoom factors along each axis.
    if a scalar is provided, the same zoom is applied
    along each axis.

  cval : float 
    constant value around the input array needed for boarder voxels (default 0)
 
  Returns
  -------
  3d numpy arrays
    zoomed version of the input array.

  Note
  ----
  This function is supposed to be similar to scipy.ndimage.zoom
  but much faster (parallel via numba) and better if the zoom factors 
  are < 1 (down sampling).
  """
  if not isinstance(zoom, (list, tuple, np.ndarray)):
    zoom = 3*[zoom]

  if zoom[0] > 1:
    vol = upsample_3d_0(vol, zoom[0], cval = cval)
  elif zoom[0] < 1:
    vol = downsample_3d_0(vol, zoom[0], cval = cval)

  if zoom[1] > 1:
    vol = upsample_3d_1(vol, zoom[1], cval = cval)
  elif zoom[1] < 1:
    vol = downsample_3d_1(vol, zoom[1], cval = cval)

  if zoom[2] > 1:
    vol = upsample_3d_2(vol, zoom[2], cval = cval)
  elif zoom[2] < 1:
    vol = downsample_3d_2(vol, zoom[2], cval = cval)

  return vol

#----------------------------------------------------------------
