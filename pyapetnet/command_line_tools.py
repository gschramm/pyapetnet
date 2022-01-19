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
  parser.add_argument('--output_on_mr_grid', help = 'regrid the CNN output to the original MR grid', 
                                             action = 'store_true')
  parser.add_argument('--output_on_pet_grid', help = 'regrid the CNN output to the original PET grid (overrules output_on_mr_grid)', 
                                             action = 'store_true')
  parser.add_argument('--align_frames_individually', help = 'align each of the PET frames individually to the MR rather than the sum of the frames', 
                                             action = 'store_true')
  parser.add_argument('--exclude_frames', help = 'list of frames not to align individually (used with option --align_frames_individually), use the mean alignment')
  
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
  from pymirc.image_operations import aff_transform

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
  output_on_mr_grid = args.output_on_mr_grid
  output_on_pet_grid = args.output_on_pet_grid
  align_frames_individually = args.align_frames_individually
  print('args.exclude_frames '+str(args.exclude_frames))
  if args.exclude_frames:
    exclude_frames = eval(args.exclude_frames)
  else:
    exclude_frames = [];

  print('exclude_frames '+str(exclude_frames))

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
  pet_preproc, mr_preproc, o_aff, pet_scale, mr_scale, pet_mr_interp_aff = preprocess_volumes(pet, mr, 
    pet_affine, mr_affine, training_voxsize, perc = 99.99, coreg = coreg_inputs, crop_mr = crop_mr,
    align_individually = align_frames_individually, exclude_frames = exclude_frames, verbose = verbose)
    
  #------------------------------------------------------------------
  # the actual CNN prediction
  if (pet_preproc.ndim == 4):
    pred = np.zeros(pet_preproc.shape)
    for i in range(pet_preproc.shape[-1]):
      if verbose: print('CNN prediction for frame ' + str(i) + ' of ' + str(pet_preproc.shape[-1]) )
      x = [np.expand_dims(np.expand_dims(pet_preproc[...,i],0),-1), np.expand_dims(np.expand_dims(mr_preproc,0),-1)]
      pred[...,i] = model.predict(x).squeeze()
  else:
    x = [np.expand_dims(np.expand_dims(pet_preproc,0),-1), np.expand_dims(np.expand_dims(mr_preproc,0),-1)]
    pred = model.predict(x).squeeze()
  
  # undo the intensity normalization
  pred        *= pet_scale
  pet_preproc *= pet_scale
  mr_preproc  *= mr_scale
  
  #------------------------------------------------------------------
  # save the preprocessed input and output
  os.makedirs(output_dir, exist_ok = True)

  if save_preproc:
    nib.save(nib.Nifti1Image(pet_preproc, o_aff), os.path.join(output_dir, 'pet_preproc.nii'))
    if verbose: print('wrote pre-processed PET to: {os.path.join(output_dir, "pet_preproc.nii")}')

    nib.save(nib.Nifti1Image(mr_preproc, o_aff), os.path.join(output_dir, 'mr_preproc.nii'))
    if verbose: print('wrote pre-processed MR  to: {os.path.join(output_dir, "mr_preproc.nii")}')

    # save the intensity normalization factors
    np.savetxt(os.path.join(output_dir, 'preproc_scaling_factors.txt'), np.array([pet_scale,mr_scale]))
    if verbose: print('wrote scaling factors   to: {os.path.join(output_dir, "preproc_scaling_factors.txt")}')

  if output_on_pet_grid:
    oss = np.ceil(np.linalg.norm(pet_affine[:-1,:-1], axis = 0)/training_voxsize).astype(int)
    if (pred.ndim == 4):
      pred_regrid = np.zeros(pet.shape[:])
      for i in range(pred.shape[-1]):
        pred_regrid[...,i] = aff_transform(pred[...,i], np.linalg.inv(pet_mr_interp_aff), pet.shape[:-1], cval = pred.min(),
                                os0 = oss[0], os1 = oss[1], os2 = oss[2])
    else:
      pred_regrid = aff_transform(pred, np.linalg.inv(pet_mr_interp_aff), pet.shape, cval = pred.min(),
                                os0 = oss[0], os1 = oss[1], os2 = oss[2]) 
    nib.save(nib.Nifti1Image(pred_regrid, pet_affine), os.path.join(output_dir, output_name))
  elif output_on_mr_grid:
    oss = np.ceil(np.linalg.norm(mr_affine[:-1,:-1], axis = 0)/training_voxsize).astype(int)
    if (pred.ndim == 4):
      pred_regrid = np.zeros(mr.shape[:]+(pred.shape[-1],))
      for i in range(pred.shape[-1]):
        pred_regrid[...,i] = aff_transform(pred[...,i], np.linalg.inv(o_aff) @ mr_affine, mr.shape, cval = pred.min(),
                                os0 = oss[0], os1 = oss[1], os2 = oss[2])
    else:
      pred_regrid = aff_transform(pred, np.linalg.inv(o_aff) @ mr_affine, mr.shape, cval = pred.min(),
                                os0 = oss[0], os1 = oss[1], os2 = oss[2]) 
    nib.save(nib.Nifti1Image(pred_regrid, mr_affine), os.path.join(output_dir, output_name))
  else:
    nib.save(nib.Nifti1Image(pred, o_aff), os.path.join(output_dir, output_name))
  if verbose: print('wrote prediction to       : {os.path.join(output_dir, output_name)}')
  
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

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def predict_from_dicom():
  parser = argparse.ArgumentParser(description='pyapetnet prediction of anatomy-guided \
                                                PET reconstruction')
  parser.add_argument('pet_dcm_dir',     help = 'absolute path of PET input dicom directory')
  parser.add_argument('mr_dcm_dir',      help = 'absolute path of MR  input dicom directory')
  parser.add_argument('model_name',      help = 'name of trained CNN')

  parser.add_argument('--pet_dcm_pattern', help = 'file pattern for PET dicom dir', default = '*')
  parser.add_argument('--mr_dcm_pattern',  help = 'file pattern for MR dicom dir',  default = '*')
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
  parser.add_argument('--output_on_mr_grid', help = 'regrid the CNN output to the original MR grid', 
                                             action = 'store_true')
  parser.add_argument('--output_on_pet_grid', help = 'regrid the CNN output to the original PET grid', 
                                             action = 'store_true')
  
  args = parser.parse_args()

  #-------------------------------------------------------------------------------------------------
  # load modules

  import pyapetnet
  from pyapetnet.preprocessing import preprocess_volumes
  from pyapetnet.utils         import load_nii_in_ras
  from pymirc.fileio           import DicomVolume, write_3d_static_dicom

  import nibabel as nib
  import json
  import tensorflow as tf
  
  import numpy as np
  import matplotlib.pyplot as plt
  import pymirc.viewer as pv
  from pymirc.image_operations import aff_transform

  from pyapetnet.utils import flip_ras_lps, pet_dcm_keys_to_copy
  from warnings import warn
  #-------------------------------------------------------------------------------------------------
  # parse input parameters

  pet_dcm_dir = args.pet_dcm_dir
  mr_dcm_dir  = args.mr_dcm_dir
  pet_dcm_pattern = args.pet_dcm_pattern
  mr_dcm_pattern  = args.mr_dcm_pattern

  model_name  = args.model_name
  output_dir  = args.output_dir
  output_name = args.output_name

  if output_name is None:
    output_name = f'prediction_{model_name}'

  model_path  = args.model_path

  if model_path is None:
    model_path = os.path.join(os.path.dirname(pyapetnet.__file__),'trained_models')

  coreg_inputs = not args.no_coreg
  crop_mr      = not args.no_crop
  show         = args.show
  verbose      = args.verbose
  save_preproc = not args.no_preproc_save
  output_on_mr_grid = args.output_on_mr_grid
  output_on_pet_grid = args.output_on_pet_grid

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
  pet_dcm = DicomVolume(os.path.join(pet_dcm_dir, pet_dcm_pattern))
  mr_dcm  = DicomVolume(os.path.join(mr_dcm_dir,  mr_dcm_pattern))
  
  pet = pet_dcm.get_data()
  mr  = mr_dcm.get_data()
  
  pet_affine = pet_dcm.affine
  mr_affine  = mr_dcm.affine
  
  # preprocess the input volumes (coregistration, interpolation and intensity normalization)
  pet_preproc, mr_preproc, o_aff, pet_scale, mr_scale = preprocess_volumes(pet, mr, 
    pet_affine, mr_affine, training_voxsize, perc = 99.99, coreg = coreg_inputs, crop_mr = crop_mr)
  
  #------------------------------------------------------------------
  # the actual CNN prediction
  x = [np.expand_dims(np.expand_dims(pet_preproc,0),-1), np.expand_dims(np.expand_dims(mr_preproc,0),-1)]
  pred = model.predict(x).squeeze()
  
  # undo the intensity normalization
  pred        *= pet_scale
  pet_preproc *= pet_scale
  mr_preproc  *= mr_scale

  #------------------------------------------------------------------
  # save the preprocessed input and output
  # dicom volumes are read as LPS, but nifti volumes have to be in RAS
  os.makedirs(output_dir, exist_ok = True)

  if save_preproc:
    nib.save(nib.Nifti1Image(*flip_ras_lps(pet_preproc, o_aff)), 
                             os.path.join(output_dir, 'pet_preproc.nii'))
    if verbose: print('wrote pre-processed PET to: {os.path.join(output_dir, "pet_preproc.nii")}')
    nib.save(nib.Nifti1Image(*flip_ras_lps(mr_preproc, o_aff)),  
                             os.path.join(output_dir, 'mr_preproc.nii'))
    if verbose: print('wrote pre-processed MR  to: {os.path.join(output_dir, "mr_preproc.nii")}')

    # save the intensity normalization factors
    np.savetxt(os.path.join(output_dir, 'preproc_scaling_factors.txt'), np.array([pet_scale,mr_scale]))
    if verbose: print('wrote scaling factors   to: {os.path.join(output_dir, "preproc_scaling_factors.txt")}')

  if output_on_pet_grid:
    oss = np.ceil(np.linalg.norm(pet_affine[:-1,:-1], axis = 0)/training_voxsize).astype(int)
    pred_regrid = aff_transform(pred, np.linalg.inv(o_aff) @ pet_affine, pet.shape, cval = pred.min(),
                                os0 = oss[0], os1 = oss[1], os2 = oss[2]) 
    nib.save(nib.Nifti1Image(pred_regrid, pet_affine), os.path.join(output_dir, output_name))
  elif output_on_mr_grid:
    oss = np.ceil(np.linalg.norm(mr_affine[:-1,:-1], axis = 0)/training_voxsize).astype(int)
    pred_regrid = aff_transform(pred, np.linalg.inv(o_aff) @ mr_affine, mr.shape, cval = pred.min(),
                                os0 = oss[0], os1 = oss[1], os2 = oss[2]) 
    nib.save(nib.Nifti1Image(*flip_ras_lps(pred_regrid, mr_affine)), 
                             os.path.join(output_dir, f'{output_name}.nii'))
  else:
    nib.save(nib.Nifti1Image(*flip_ras_lps(pred, o_aff)), 
                             os.path.join(output_dir, f'{output_name}.nii'))

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
  output_dcm_dir = os.path.join(output_dir, output_name)
  if not os.path.exists(output_dcm_dir):
    if output_on_pet_grid:
      write_3d_static_dicom(pred_regrid, output_dcm_dir, affine = pet_affine, 
                            ReconstructionMethod = 'CNN MAP Bowsher', 
                            SeriesDescription = f'CNN MAP Bowsher {model_name}', **dcm_kwargs)
    elif output_on_mr_grid:
      write_3d_static_dicom(pred_regrid, output_dcm_dir, affine = mr_affine, 
                            ReconstructionMethod = 'CNN MAP Bowsher', 
                            SeriesDescription = f'CNN MAP Bowsher {model_name}', **dcm_kwargs)
    else:
      write_3d_static_dicom(pred, output_dcm_dir, affine = o_aff, 
                            ReconstructionMethod = 'CNN MAP Bowsher', 
                            SeriesDescription = f'CNN MAP Bowsher {model_name}', **dcm_kwargs)
  else:
    warn('Output dicom directory already exists. Not ovewrting it')
  
  #------------------------------------------------------------------
  # show the results
  if show:
    pmax = np.percentile(pred,99.9)
    mmax = np.percentile(mr_preproc,99.9)
    
    ims = [{'vmin':0, 'vmax': mmax, 'cmap': plt.cm.Greys_r}, 
           {'vmin':0, 'vmax': pmax}, {'vmin':0, 'vmax': pmax}]
    vi = pv.ThreeAxisViewer([mr_preproc, pet_preproc, pred], imshow_kwargs = ims)
    plt.show()
