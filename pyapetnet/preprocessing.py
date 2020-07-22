import numpy as np
from scipy.ndimage import find_objects, gaussian_filter
from scipy.optimize import minimize

from .aff_transform import aff_transform, kul_aff
from .rigid_registration import regis_cost_func, neg_mutual_information
from .zoom3d import zoom3d

def align_inputs(pet_vol, mr_vol, pet_affine, mr_affine, coreg = True):
  """ calculate affine transformation matrix to map PET onto MR grid
  """
  # this is the affine that maps from the PET onto the MR grid
  pre_affine = np.linalg.inv(pet_affine) @ mr_affine

  if coreg: 
    reg_params = np.zeros(6)
    
    # (1) initial registration with downsampled arrays
    # define the down sampling factor
    dsf    = 3
    ds_aff = np.diag([dsf,dsf,dsf,1.])
    
    mr_vol_ds = aff_transform(mr_vol, ds_aff, np.ceil(np.array(mr_vol.shape)/dsf).astype(int))

    res = minimize(regis_cost_func, reg_params, 
                   args = (mr_vol_ds, pet_vol, True, True, neg_mutual_information, pre_affine @ ds_aff), 
                   method = 'Powell', 
                   options = {'ftol':1e-2, 'xtol':1e-2, 'disp':True, 'maxiter':20, 'maxfev':5000})
    
    reg_params = res.x.copy()
    # we have to scale the translations by the down sample factor since they are in voxels
    reg_params[:3] *= dsf
    
    # (2) registration with full arrays
    res = minimize(regis_cost_func, reg_params, 
                   args = (mr_vol, pet_vol, True, True, neg_mutual_information, pre_affine), 
                   method = 'Powell', 
                   options = {'ftol':1e-2, 'xtol':1e-2, 'disp':True, 'maxiter':20, 'maxfev':5000})
    reg_params = res.x.copy()

    regis_aff = pre_affine @ kul_aff(reg_params, origin = np.array(mr_vol.shape)/2)
  else:
    regis_aff = pre_affine.copy()

  return regis_aff

#-----------------------------------------------------------------------------------------------------

def preprocess_volumes(pet_vol, mr_vol, pet_affine, mr_affine, training_voxsize,
                       perc = 99.99, coreg = True, crop_mr = True, 
                       mr_ps_fwhm_mm = None, verbose = False):

  # get voxel sizes from affine matrices
  pet_voxsize = np.sqrt((pet_affine**2).sum(axis = 0))[:-1]
  mr_voxsize  = np.sqrt((mr_affine**2).sum(axis = 0))[:-1]

  # crop the MR if needed
  if crop_mr:
    bbox              = find_objects(mr_vol > 0.1*mr_vol.max(), max_label = 1)[0]
    mr_vol            = mr_vol[bbox]
    crop_origin       = np.array([x.start for x in bbox] + [1])
    mr_affine[:-1,-1] = (mr_affine @ crop_origin)[:-1]

  # post-smooth MR if needed
  if mr_ps_fwhm_mm is not None:
    print(f'post-smoothing MR with {mr_ps_fwhm_mm} mm')
    mr_vol = gaussian_filter(mr_vol, mr_ps_fwhm_mm / (2.35*mr_voxsize))

  # regis_aff is the affine transformation that maps from the PET to the MR grid
  # if coreg is False, it is simply deduced from the affine transformation
  # otherwise, rigid registration with mutual information is used
  regis_aff = align_inputs(pet_vol, mr_vol, pet_affine, mr_affine, coreg = coreg)

  # interpolate both volumes to the voxel size used during training
  zoomfacs = mr_voxsize / training_voxsize
  if not np.all(np.isclose(zoomfacs, np.ones(3))):
    if verbose: print('interpolationg input volumes to training voxel size')
    mr_vol_interpolated = zoom3d(mr_vol, zoomfacs)
    mr_affine = mr_affine @ np.diag(np.concatenate((1./zoomfacs,[1])))
  else:
    mr_vol_interpolated = mr_vol.copy()

  # this is the final affine that maps from the PET grid to interpolated MR grid 
  # using the small voxels used during training 
  pet_mr_interp_aff = regis_aff @ np.diag(np.concatenate((1./zoomfacs,[1])))

  if not np.all(np.isclose(pet_mr_interp_aff, np.eye(4))):
    pet_vol_mr_grid_interpolated = aff_transform(pet_vol, pet_mr_interp_aff, mr_vol_interpolated.shape, 
                                                 cval = pet_vol.min()) 
  else:
    pet_vol_mr_grid_interpolated = pet_vol.copy()
 
  # convert the input volumes to float32
  if not mr_vol_interpolated.dtype == np.float32: 
    mr_vol_interpolated  = mr_vol_interpolated.astype(np.float32)
  if not pet_vol_mr_grid_interpolated.dtype == np.float32: 
    pet_vol_mr_grid_interpolated = pet_vol_mr_grid_interpolated.astype(np.float32)

  # normalize the data: we divide the images by the specified percentile (more stable than the max)
  if verbose: print('\nnormalizing the input images')
  mr_max = np.percentile(mr_vol_interpolated, perc)
  mr_vol_interpolated /= mr_max
  
  pet_max = np.percentile(pet_vol_mr_grid_interpolated, perc)
  pet_vol_mr_grid_interpolated /= pet_max


  return pet_vol_mr_grid_interpolated, mr_vol_interpolated, mr_affine, pet_max, mr_max
