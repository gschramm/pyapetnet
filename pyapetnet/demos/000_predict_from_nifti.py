import nibabel as nib
import json
import os
import tensorflow as tf

import numpy as np
import os

import matplotlib.pyplot as py

import pyapetnet
from pyapetnet.preprocessing import preprocess_volumes
from pyapetnet.utils         import load_nii_in_ras

#------------------------------------------------------------------
# inputs (adapt to your needs)

# the name of the trained CNN
model_name = '200710_mae_osem_psf_bet_10'

# we use a simulated demo data included in pyapetnet (based on the brainweb phantom)
pet_fname  = os.path.join(os.path.dirname(pyapetnet.__file__), 'data', 'brainweb_06_osem.nii')
mr_fname   = os.path.join(os.path.dirname(pyapetnet.__file__), 'data', 'brainweb_06_t1.nii')

# preprocessing parameters

coreg_inputs = False  # rigidly coregister PET and MR using mutual information
crop_mr      = True   # crop the input to the support of the MR (saves memory + speeds up the computation)

# the name of the ouput file
output_fname =  f'prediction_{model_name}.nii'


#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


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
pet, pet_affine = load_nii_in_ras(pet_fname)
mr, mr_affine   = load_nii_in_ras(mr_fname)

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
nib.save(nib.Nifti1Image(pet_preproc, o_aff), 'pet_preproc.nii')
nib.save(nib.Nifti1Image(mr_preproc, o_aff), 'mr_preproc.nii')
nib.save(nib.Nifti1Image(pred, o_aff), output_fname)


# show the results
import pymirc.viewer as pv
pmax = np.percentile(pred,99.9)
mmax = np.percentile(mr_preproc,99.9)

ims = [{'vmin':0, 'vmax': mmax, 'cmap': py.cm.Greys_r}, 
       {'vmin':0, 'vmax': pmax}, {'vmin':0, 'vmax': pmax}]
pv.ThreeAxisViewer([np.flip(mr_preproc,(0,1)),np.flip(pet_preproc,(0,1)),np.flip(pred,(0,1))],
                   imshow_kwargs = ims)
