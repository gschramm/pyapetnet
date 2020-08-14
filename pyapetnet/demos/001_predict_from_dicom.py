import nibabel as nib
import json
import os
import tensorflow as tf

import numpy as np
import os

import matplotlib.pyplot as py

import pyapetnet
from pyapetnet.preprocessing import preprocess_volumes
from pyapetnet.utils         import flip_ras_lps, create_demo_dcm_data, pet_dcm_keys_to_copy

from pymirc.fileio import DicomVolume, write_3d_static_dicom

from warnings import warn

#------------------------------------------------------------------
# inputs (adapt to your needs)

# create demo dicom data from the included nifti data sets
# just needed in case no real data is available
if not os.path.exists('demo_dcm'): create_demo_dcm_data('demo_dcm')

pet_dcm_pattern = os.path.join('demo_dcm','PT','*.dcm')
mr_dcm_pattern  = os.path.join('demo_dcm','MR','*.dcm')

# the name of the trained CNN
model_name = '200710_mae_osem_psf_bet_10'

# output dicom dir
output_dcm_dir = os.path.join('demo_dcm',f'prediction_{model_name}')


# preprocessing parameters
coreg_inputs = True   # rigidly coregister PET and MR using mutual information
crop_mr      = True   # crop the input to the support of the MR (saves memory + speeds up the computation)

#------------------------------------------------------------------
# load the trained CNN and its internal voxel size used for training
co ={'ssim_3d_loss': None,'mix_ssim_3d_mae_loss': None}

model_abs_path = os.path.join(os.path.dirname(pyapetnet.__file__),'trained_models',model_name)

model = tf.keras.models.load_model(model_abs_path, custom_objects = co)
                   
# load the voxel size used for training
with open(os.path.join(model_abs_path,'config.json')) as f:
  cfg = json.load(f)
  training_voxsize = cfg['internal_voxsize']*np.ones(3)

#------------------------------------------------------------------
# load and preprocess the input PET and MR volumes
pet_dcm = DicomVolume(pet_dcm_pattern)
mr_dcm  = DicomVolume(mr_dcm_pattern)

pet = pet_dcm.get_data()
mr  = mr_dcm.get_data()

pet_affine = pet_dcm.affine
mr_affine  = mr_dcm.affine

# preprocess the input volumes (coregistration, interpolation and intensity normalization)
pet_preproc, mr_preproc, o_aff, pet_max, mr_max = preprocess_volumes(pet, mr, 
  pet_affine, mr_affine, training_voxsize, perc = 99.99, coreg = coreg_inputs, crop_mr = crop_mr)

#------------------------------------------------------------------
# the actual CNN prediction
x = [np.expand_dims(np.expand_dims(pet_preproc,0),-1), np.expand_dims(np.expand_dims(mr_preproc,0),-1)]
pred = model.predict(x).squeeze()

# undo the intensity normalization
pet_preproc *= pet_max
mr_preproc  *= mr_max
pred        *= pet_max


#------------------------------------------------------------------
# save the preprocessed input and output
# dicom volumes are read as LPS, but nifti volumes have to be in RAS
nib.save(nib.Nifti1Image(*flip_ras_lps(pet_preproc, o_aff)), 'pet_preproc.nii')
nib.save(nib.Nifti1Image(*flip_ras_lps(mr_preproc, o_aff)), 'mr_preproc.nii')
nib.save(nib.Nifti1Image(*flip_ras_lps(pred, o_aff)), f'prediction_{model_name}.nii')

#------------------------------------------------------------------
# save prediction also as dicom

# get a list of dicom keys to copy from the original PET dicom header
dcm_kwargs = {}
for key in pet_dcm_keys_to_copy():
  try:
    dcm_kwargs[key] = getattr(pet_dcm.firstdcmheader,key)
  except AttributeError:
    warn('Cannot copy tag ' + key)
    
# write the dicom volume  
if not os.path.exists(output_dcm_dir):
  write_3d_static_dicom(pred, output_dcm_dir, affine = o_aff, ReconstructionMethod = 'CNN MAP Bowsher', 
                        SeriesDescription = f'CNN MAP Bowsher {model_name}', **dcm_kwargs)
else:
  warn('Output dicom directory already exists. Not ovewrting it')

#------------------------------------------------------------------
# show the results
import pymirc.viewer as pv
pmax = np.percentile(pred,99.9)
mmax = np.percentile(mr_preproc,99.9)

ims = [{'vmin':0, 'vmax': mmax, 'cmap': py.cm.Greys_r}, 
       {'vmin':0, 'vmax': pmax}, {'vmin':0, 'vmax': pmax}]
pv.ThreeAxisViewer([mr_preproc, pet_preproc, pred], imshow_kwargs = ims)
