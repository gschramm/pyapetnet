import numpy as np

from scipy.optimize import minimize
from .aff_transform import aff_transform, kul_aff
from .rigid_registration import regis_cost_func, neg_mutual_information

def align_inputs(pet, mr, pet_affine, mr_affine, coreg = True):
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
