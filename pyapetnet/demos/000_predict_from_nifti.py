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

#------
# inputs
pet_fname  = 'pet.nii'
mr_fname   = 'mr.nii'
model_name = '200710_mae_osem_psf_bet_10'

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
pet, pet_affine = load_nii_in_ras(pet_fname)
mr, mr_affine   = load_nii_in_ras(mr_fname)

# preprocess the input volumes (coregistration, interpolation and intensity normalization)
pet_preproc, mr_preproc, o_aff, pet_max, mr_max = preprocess_volumes(pet, mr, 
  pet_affine, mr_affine, training_voxsize, perc = 99.99, coreg = True, crop_mr = True)


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
nib.save(nib.Nifti1Image(pred, o_aff), f'prediction_{model_name}.nii')


# show the results
fig, ax = py.subplots(3,3, figsize = (9,9))
ax[0,0].imshow(pet_preproc[:,::-1,pet_preproc.shape[2]//2].T, cmap = py.cm.Greys, vmax = pet_max)
ax[0,1].imshow(pet_preproc[:,pet_preproc.shape[1]//2,::-1].T, cmap = py.cm.Greys, vmax = pet_max)
ax[0,2].imshow(pet_preproc[pet_preproc.shape[0]//2,:,::-1].T, cmap = py.cm.Greys, vmax = pet_max)
ax[1,0].imshow(mr_preproc[:,::-1,pet_preproc.shape[2]//2].T, cmap = py.cm.Greys_r, vmax = mr_max)
ax[1,1].imshow(mr_preproc[:,pet_preproc.shape[1]//2,::-1].T, cmap = py.cm.Greys_r, vmax = mr_max)
ax[1,2].imshow(mr_preproc[pet_preproc.shape[2]//2,:,::-1].T, cmap = py.cm.Greys_r, vmax = mr_max)
ax[2,0].imshow(pred[:,::-1,pet_preproc.shape[2]//2].T, cmap = py.cm.Greys, vmax = pet_max)
ax[2,1].imshow(pred[:,pet_preproc.shape[1]//2,::-1].T, cmap = py.cm.Greys, vmax = pet_max)
ax[2,2].imshow(pred[pet_preproc.shape[0]//2,:,::-1].T, cmap = py.cm.Greys, vmax = pet_max)
for axx in ax.flatten(): axx.set_axis_off()

ax[0,1].set_title('input PET')
ax[1,1].set_title('input MR')
ax[2,1].set_title('predicted MAP Bowsher')

fig.tight_layout()
fig.show()
