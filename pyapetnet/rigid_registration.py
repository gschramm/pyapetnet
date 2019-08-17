import numpy as np
import math

from  numba import njit, prange

from .aff_transform import kul_aff, aff_transform
#-------------------------------------------------------------------------------
@njit(parallel = True)
def jointhisto_3dvols(x, y, nbins = 40, normalize = True):
  """Calculate the joint histogram between two 3d volumes.

  Parameters
  ----------
  x : 3D numpy array 
    first input

  y : 3D numpy array 
    second input
  
  nbins : int, optional
    number of bins in the joint histogram (default 40)

  normalize : bool, optional
    divide the counts in each bin by the number of data points (default True)

  Returns
  -------
  2D numpy array of shape (nbins, nbins) 
    containing the joint histogram

  Note
  ----
  E.g. useful for rigid registration of two 3d volumes with 
  using mutual information.
  
  The implementaion is optimized for a multi cpu system using
  numba.

  The left edges of the bins are at np.arange(nbins) * binwidth + jmin
  where jmin is the joint minimum between x and y and 
  binwidth = (jmax - jmin) / (nbins - 1).
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

 
#---------------------------------------------------------------- 
def neg_mutual_information(x, y, nbins = 40, norm = True):
  """Negative mutual information between two 3D volumes

  Parameters
  ----------
  x : 3D numpy array 
    first input

  y : 3D numpy array 
    second input
 
  nbins : int, optional
    number of bins in the joint histogram (default 40)
  
  norm: bool, optional
    whether to use normalized version of MI (default True) 

  Returns
  -------
  float
    containing the negative mutual information
  
  References
  ----------
  Maes et al: IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL. 16, NO. 2, APRIL 1997
  Studholme et al: Pattern Recognition 32 (1999) 71 86
  """
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
def regis_cost_func(params, img_fix, img_float, verbose = False,
                    rotate = True, metric = neg_mutual_information, pre_affine = None, metric_kwargs = {}):
  """Generic cost function for rigid registration

  Parameters
  ----------

  params : 3 or 6 element numpy array
    If rotate is False it contains the 3 translations
    If rotate is True  it contains the 3 translations and the 3 rotation angles
 
  img_fix : 3D numpy array
     containg the fixed (reference) volume

  img_float : 3D numpy array
     containg the floating (moving) volume

  verbose : bool, optional
    print verbose output (default False)

  rotate : bool, optional
    rotate volume as well on top of translations (6 degrees of freedom)  
 
  metric : function(img_fix, aff_transform(img_float, ...)) -> R, optional
    metric used to compare fixed and floating volume (default neg_mutual_information)

  pre_affine : 2D 4x4 numpy array
    affine transformation applied before doing the rotation and shifting

  metric_kwargs : dictionary
    key word arguments passed to the metric function

  Returns
  -------
  float
    the metric between the fixed and floating volume

  See Also
  --------
  kul_aff()
  """
  if rotate:
    p = params.copy()
  else:
    p = np.concatenate((params,np.zeros(3)))

  if pre_affine is not None:
    af  = pre_affine @ kul_aff(p, origin = np.array(img_fix.shape)/2)
  else:
    af  = kul_aff(params, origin = np.array(img_fix.shape)/2)

  m = metric(img_fix, aff_transform(img_float, af, img_fix.shape, cval = img_float.min()), **metric_kwargs)

  if verbose:
    print(params)
    print(m)
    print('')

  return m


