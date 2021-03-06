import os
import argparse
from glob import glob

def list_models():
  parser = argparse.ArgumentParser(description= 'list available trained models for pyapetnet')
  parser.add_argument('--model_path',  help = 'absolute path of directory containing trained models',
                                       default = None)
  args = parser.parse_args()

  #-------------------------------------------------------------------------------------------------
  # parse input parameters
  import pyapetnet
  model_path  = args.model_path

  if model_path is None:
    model_path = os.path.join(os.path.dirname(pyapetnet.__file__),'trained_models')

  #-------------------------------------------------------------------------------------------------
  cfg_files = sorted(glob(os.path.join(model_path, '*', 'config.json')))

  print(f'\nModel path: {model_path}')
  print('\nAvailable models')
  print('----------------')

  for i,cfg_file in enumerate(cfg_files):
    print(f'{os.path.basename(os.path.dirname(cfg_file))}')

  print(f'\nFor details about the models, read \n{os.path.join(model_path,"model_description.md")}\nor the look at the config.json files in the model directories')

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def predict_from_nifti():
  parser = argparse.ArgumentParser(description='pyapetnet prediction of anatomy-guided \
                                                PET reconstruction')
  parser.add_argument('pet_fname',     help = 'absolute path of PET input nifti file')
  parser.add_argument('mr_fname',      help = 'absolute path of MR  input nifti file')
  parser.add_argument('model_name',    help = 'name of trained CNN')
  parser.add_argument('--model_path',  help = 'absolute path of directory containing trained models',
                                       default = None)
  parser.add_argument('--output_dir',  help = 'name of the output directory', default = '.')
  parser.add_argument('--output_name', help = 'basename of prediction file', default = None)
  parser.add_argument('--no_coreg',    help = 'do not coregister input volumes', action = 'store_true')
  parser.add_argument('--no_crop',     help = 'do not crop volumes to MR bounding box', 
                                       action = 'store_true')
  parser.add_argument('--show',        help = 'show the results', action = 'store_true')
  parser.add_argument('--verbose',     help = 'print (extra) verbose output', action = 'store_true')
  parser.add_argument('--no_preproc_save', help = 'do not save preprocessed volumes', 
                                           action = 'store_true')
  
  args = parser.parse_args()

  #-------------------------------------------------------------------------------------------------
  # load modules

  import pyapetnet
  from pyapetnet.preprocessing import preprocess_volumes
  from pyapetnet.utils         import load_nii_in_ras

  import nibabel as nib
  import json
  import tensorflow as tf
  
  import numpy as np
  import matplotlib.pyplot as plt
  import pymirc.viewer as pv

  #-------------------------------------------------------------------------------------------------
  # parse input parameters

  pet_fname   = args.pet_fname
  mr_fname    = args.mr_fname
  model_name  = args.model_name
  output_dir  = args.output_dir
  output_name = args.output_name

  if output_name is None:
    output_name = f'prediction_{model_name}.nii'

  model_path  = args.model_path

  if model_path is None:
    model_path = os.path.join(os.path.dirname(pyapetnet.__file__),'trained_models')

  coreg_inputs = not args.no_coreg
  crop_mr      = not args.no_crop
  show         = args.show
  verbose      = args.verbose
  save_preproc = not args.no_preproc_save


  #-------------------------------------------------------------------------------------------------
  # load the trained model
  
  co ={'ssim_3d_loss': None,'mix_ssim_3d_mae_loss': None}
 
  if verbose:
    print('loading CNN {os.path.join(model_path, model_name)}')

  model = tf.keras.models.load_model(os.path.join(model_path, model_name), custom_objects = co)
                     
  # load the voxel size used for training
  with open(os.path.join(model_path, model_name, 'config.json')) as f:
    cfg = json.load(f)
    training_voxsize = cfg['internal_voxsize']*np.ones(3)


  #------------------------------------------------------------------
  # load and preprocess the input PET and MR volumes
  pet, pet_affine = load_nii_in_ras(pet_fname)
  mr, mr_affine   = load_nii_in_ras(mr_fname)
  
  # preprocess the input volumes (coregistration, interpolation and intensity normalization)
  pet_preproc, mr_preproc, o_aff, pet_scale, mr_scale = preprocess_volumes(pet, mr, 
    pet_affine, mr_affine, training_voxsize, perc = 99.99, coreg = coreg_inputs, crop_mr = crop_mr)
  
  
  #------------------------------------------------------------------
  # the actual CNN prediction
  x = [np.expand_dims(np.expand_dims(pet_preproc,0),-1), np.expand_dims(np.expand_dims(mr_preproc,0),-1)]
  pred = model.predict(x).squeeze()
  
  # undo the intensity normalization
  pet_preproc *= pet_scale
  
  #------------------------------------------------------------------
  # save the preprocessed input and output
  os.makedirs(output_dir, exist_ok = True)

  if save_preproc:
    nib.save(nib.Nifti1Image(pet_preproc, o_aff), os.path.join(output_dir, 'pet_preproc.nii'))
    if verbose: print('wrote pre-processed PET to: {os.path.join(output_dir, "pet_preproc.nii")}')

    nib.save(nib.Nifti1Image(mr_preproc, o_aff), os.path.join(output_dir, 'mr_preproc.nii'))
    if verbose: print('wrote pre-processed MR  to: {os.path.join(output_dir, "mr_preproc.nii")}')

  nib.save(nib.Nifti1Image(pred, o_aff), os.path.join(output_dir, output_name))
  if verbose: print('wrote predictin         to: {os.path.join(output_dir, output_name)}')

  # undo the intensity normalization
  mr_preproc  *= mr_scale
  pred        *= pet_scale
  
  #------------------------------------------------------------------
  # show the results
  if show:
    pmax = np.percentile(pred,99.9)
    mmax = np.percentile(mr_preproc,99.9)
    
    ims = [{'vmin':0, 'vmax': mmax, 'cmap': plt.cm.Greys_r}, 
           {'vmin':0, 'vmax': pmax}, {'vmin':0, 'vmax': pmax}]
    vi = pv.ThreeAxisViewer([np.flip(mr_preproc,(0,1)),np.flip(pet_preproc,(0,1)),np.flip(pred,(0,1))],
                            imshow_kwargs = ims)
    plt.show()
