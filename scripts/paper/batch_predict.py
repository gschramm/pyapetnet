import sys
import os
if not os.path.join('..','..') in sys.path: sys.path.append(os.path.join('..','..'))

import pickle
import h5py
import nibabel as nib
import numpy   as np
import json

from glob             import glob
from scipy.ndimage import zoom
from pyapetnet.losses import ssim_3d_loss, mix_ssim_3d_mae_loss

import tensorflow
if tensorflow.__version__ >= '2':
  from tensorflow.keras.models import load_model
else:
  from keras.models import load_model

from scipy.ndimage    import find_objects

#==========================================================================================
def load_nii(fname):
  nii = nib.load(fname)
  nii = nib.as_closest_canonical(nii)
  vol = nii.get_fdata()

  return vol, nii.affine

#==========================================================================================
def predict_from_nii(pet_input, 
                     mr_input, 
                     model,
                     training_voxsize, 
                     output_file, 
                     perc = 99.99):

  # read the input data
  mr_vol, mr_affine = load_nii(mr_input)
  mr_voxsize        = np.sqrt((mr_affine**2).sum(axis = 0))[:-1]
  bbox              = find_objects(mr_vol > 0.1*mr_vol.max(), max_label = 1)[0]
  mr_vol_crop       = mr_vol[bbox]
  crop_origin       = np.array([x.start for x in bbox] + [1])

  # update the affine after cropping
  mr_affine_crop         = mr_affine.copy()
  mr_affine_crop[:-1,-1] = (mr_affine_crop @ crop_origin)[:-1]

  # read the pet data
  pet_vol, pet_affine = load_nii(pet_input)
  pet_vol_crop        = pet_vol[bbox]

  # interpolate the volumes to the internal voxelsize of the trained model 
  zoomfacs            = mr_voxsize / training_voxsize
  mr_vol_crop_interp  = zoom(mr_vol_crop, zoomfacs, order = 1, prefilter = False)
  pet_vol_crop_interp = zoom(pet_vol_crop, zoomfacs, order = 1, prefilter = False)

  # normalize the input
  pmax = np.percentile(pet_vol_crop_interp, perc)
  mmax = np.percentile(mr_vol_crop_interp, perc)

  pet_vol_crop_interp /= pmax
  mr_vol_crop_interp  /= mmax

  # make the prediction
  x = [np.expand_dims(np.expand_dims(pet_vol_crop_interp,0),-1), np.expand_dims(np.expand_dims(mr_vol_crop_interp,0),-1)]

  pred = model.predict(x).squeeze() 

  # unnormalize the data
  pred                *= pmax
  pet_vol_crop_interp *= pmax
  mr_vol_crop_interp  *= mmax

  # generat the output affine transform
  output_affine = mr_affine_crop.copy()
  for i in range(3):  output_affine[i,:-1] /= zoomfacs

  # save the prediction
  nib.save(nib.Nifti1Image(pred, output_affine), output_file)

  print('wrote: ', output_file)

  # save the bbox and the zoomfacs
  pickle.dump({'bbox':bbox, 'zoomfacs':zoomfacs}, open(os.path.splitext(output_file)[0] + '_bbox.pkl','wb'))

  return pred

#==========================================================================================
#==========================================================================================
#==========================================================================================

model_name = sys.argv[1]
dataset    = sys.argv[2]
osem_sdir  = sys.argv[3]
osem_file  = sys.argv[4]

model_dir = os.path.join('..','..','data','trained_models')
mr_file   = 'aligned_t1.nii'

if dataset == 'mmr-fdg':
  mdir      = '../../data/test_data/mMR/Tim-Patients'
  pdirs     = glob(os.path.join(mdir,'Tim-Patient-*'))
elif dataset == 'signa-pe2i':
  mdir      = '../../data/test_data/signa/signa-pe2i'
  pdirs     = glob(os.path.join(mdir,'ANON????'))
elif dataset == 'signa-fet':
  mdir      = '../../data/test_data/signa/signa-fet'
  pdirs     = glob(os.path.join(mdir,'ANON????'))
elif dataset == 'signa-fdg':
  mdir      = '../../data/test_data/signa/signa-fdg'
  pdirs     = glob(os.path.join(mdir,'?-?'))
elif dataset == 'signa-amyloid':
  mdir      = '../../data/test_data/signa/signa-amyloid'
  pdirs     = glob(os.path.join(mdir,'ANON???'))

# load the model
# we have to check whether the model is in h5 or protobuf format
# in the h5 case, the internal voxsize is stored in the h5 file
# in the protobuf case we read it from the config file

if os.path.isdir(os.path.join(model_dir,model_name)):
  with open(os.path.join(model_dir,model_name,'config.json')) as f:
    cfg = json.load(f)
    training_voxsize = cfg['internal_voxsize']*np.ones(3)
else:
  with h5py.File(os.path.join(model_dir,model_name)) as model_data:
    training_voxsize = model_data['header/internal_voxsize'][:] 

model = load_model(os.path.join(model_dir,model_name),
                   custom_objects={'ssim_3d_loss': ssim_3d_loss, 
                                     'mix_ssim_3d_mae_loss': mix_ssim_3d_mae_loss})
for pdir in pdirs:
  print(pdir)

  output_dir = os.path.join(pdir,'predictions',osem_sdir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  output_file = os.path.join(output_dir, '___'.join([os.path.splitext(model_name)[0],osem_file]))

  if not os.path.exists(output_file):
    pred = predict_from_nii(os.path.join(pdir,osem_sdir,osem_file),
                            os.path.join(pdir,mr_file),
                            model,
                            training_voxsize,
                            output_file)
  else:
    print(output_file,' already exists.')
