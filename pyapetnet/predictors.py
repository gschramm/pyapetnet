import numpy as np
import os
import nibabel as nib
import pydicom
import h5py

from copy                  import deepcopy
from warnings              import warn
from glob                  import glob
from scipy.ndimage         import find_objects, gaussian_filter
from scipy.optimize        import minimize

from keras.models          import load_model

from .threeaxisviewer     import ThreeAxisViewer
from .read_dicom          import DicomVolume
from .rigid_registration  import regis_cost_func, neg_mutual_information
from .aff_transform       import aff_transform, kul_aff
from .write_dicom         import write_3d_static_dicom

import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matplotlib.pyplot as py

# imports for old predictors
from time import time
from .generators import PatchSequence

from .zoom3d             import zoom3d

#--------------------------------------------------------------------------------------

def predict(pet_input,
            mr_input,
            model_name,
            input_format         = 'dicom',
            odir                 = None,
            model_dir            = os.path.join('..','trained_models'),
            perc                 = 99.99,
            verbose              = False,
            clip_neg             = True,
            ReconstructionMethod = 'CNN Bowsher',
            coreg                = True,
            seriesdesc           = None,
            affine               = None,
            crop_mr              = False,
            patchsize            = (128,128,128),
            overlap              = 8,
            output_on_pet_grid   = False,
            mr_ps_fwhm_mm        = None,
            debug_mode           = False):

  if seriesdesc is None:
    SeriesDescription = 'CNN Bowsher beta = 10 ' + model_name.replace('.h5','')
  else:
    SeriesDescription = seriesdesc
  
  if affine is None:
    if input_format == 'dicom':
      if isinstance(pet_input, list):
        regis_affine_file = os.path.dirname(pet_input[0]) + '_coreg_affine.txt'
      else:
        regis_affine_file = os.path.dirname(pet_input) + '_coreg_affine.txt'
    else:
      regis_affine_file = pet_input + '_coreg_affine.txt'

  else:
    regis_affine_file = affine

  # generate the output directory
  if odir is None:
    if input_format == 'dicom':
      if isinstance(pet_input, list):
        odir = os.path.join(os.path.dirname(os.path.dirname(pet_input[0])), 
                            'cnn_bow_' + model_name.replace('.h5',''))
      else:
        odir = os.path.join(os.path.dirname(os.path.dirname(pet_input)), 
                            'cnn_bow_' + model_name.replace('.h5',''))
    else:
      odir = os.path.join(os.path.dirname(pet_input), 'cnn_bow_' + model_name.replace('.h5',''))
  
  # check if output directory already exists, if so add a counter to prevent overwriting
  o_suf = 1
  if os.path.isdir(odir):
    while os.path.isdir(odir + '_' + str(o_suf)):
      o_suf += 1
    odir = odir + '_' + str(o_suf)
  
  # load the model
  model = load_model(os.path.join(model_dir,model_name))
 
  # read the input data
  if input_format == 'dicom':
    # read the MR dicoms
    if verbose: print('\nreading MR dicoms')
    if isinstance(mr_input, list):
      mr_files   = deepcopy(mr_input)
    else:
      mr_files   = glob(mr_input)
    mr_dcm     = DicomVolume(mr_files)
    mr_vol     = mr_dcm.get_data()
    mr_affine  = mr_dcm.affine
    mr_voxsize = np.sqrt((mr_affine**2).sum(axis = 0))[:-1]
 
    # read the PET dicoms
    if verbose: print('\nreading PET dicoms')
    if isinstance(pet_input, list):
      pet_files  = deepcopy(pet_input)
    else:
      pet_files  = glob(pet_input)
    pet_dcm    = DicomVolume(pet_files)
    pet_vol    = pet_dcm.get_data()
    pet_affine = pet_dcm.affine

  elif input_format == 'nifti':
    if verbose: print('\nreading MR nifti')
    mr_nii        = nib.load(mr_input)
    mr_nii        = nib.as_closest_canonical(mr_nii)
    mr_vol_ras    = mr_nii.get_data()
    mr_affine_ras = mr_nii.affine
    # the closest canonical orientation of nifti is RAS
    # we have to convert that into LPS (dicom standard)
    mr_vol    = np.flip(np.flip(mr_vol_ras, 0), 1)
    mr_affine = mr_affine_ras.copy()
    mr_affine[0,-1] = (-1 * mr_nii.affine @ np.array([mr_vol.shape[0]-1,0,0,1]))[0]
    mr_affine[1,-1] = (-1 * mr_nii.affine @ np.array([0,mr_vol.shape[1]-1,0,1]))[1]
    mr_voxsize      = np.sqrt((mr_affine**2).sum(axis = 0))[:-1]

    if verbose: print('\nreading PET nifti')
    pet_nii        = nib.load(pet_input)
    pet_nii        = nib.as_closest_canonical(pet_nii)
    pet_vol_ras    = pet_nii.get_data()

    pet_affine_ras = pet_nii.affine
    # the closest canonical orientation of nifti is RAS
    # we have to convert that into LPS (dicom standard)
    pet_vol    = np.flip(np.flip(pet_vol_ras, 0), 1)
    pet_affine = pet_affine_ras.copy()
    pet_affine[0,-1] = (-1 * pet_nii.affine @ np.array([pet_vol.shape[0]-1,0,0,1]))[0]
    pet_affine[1,-1] = (-1 * pet_nii.affine @ np.array([0,pet_vol.shape[1]-1,0,1]))[1]
    pet_voxsize      = np.sqrt((pet_affine**2).sum(axis = 0))[:-1]
  else:
    raise TypeError('Unsupported input data format')

  # crop the MR if needed
  if crop_mr:
    bbox              = find_objects(mr_vol > 0.1*mr_vol.max(), max_label = 1)[0]
    mr_vol            = mr_vol[bbox]
    crop_origin       = np.array([x.start for x in bbox] + [1])
    mr_affine[:-1,-1] = (mr_affine @ crop_origin)[:-1]

  # post-smooth MR if needed
  if mr_ps_fwhm_mm is not None:
    print('post-smoothing MR with ' + str(mr_ps_fwhm_mm) + ' mm')
    mr_vol = gaussian_filter(mr_vol, mr_ps_fwhm_mm / (2.35*mr_voxsize))

  # interpolate the PET image to the MR voxel grid
  if verbose: print('\ninterpolationg PET to MR grid')

  # this is the affine that maps from the PET onto the MR grid
  pre_affine = np.linalg.inv(pet_affine) @ mr_affine

  ################################################################
  ############ coregister the images #############################
  ################################################################
 
  if coreg: 
    # rigidly align the shifted PET to CT by minimizing neg mutual information
    # downsample the arrays by a factor for a fast pre-registration
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

  # the regis_aff is the affine that maps from the PET onto MR grid including coregistration

  #############################################################
  #############################################################

  # read the internal voxel size that was used during training from the model header
  model_data = h5py.File(os.path.join(model_dir,model_name))

  if 'header/internal_voxsize' in model_data:
    training_voxsize = model_data['header/internal_voxsize'][:] 
  else:
    # in the old models the training (internal) voxel size is not in the header
    # but it was always 1x1x1 mm^3
    training_voxsize = np.ones(3)
    
  # interpolate both volumes to 1mm^3 voxels
  # this is needed because the model was trained on 1mm^3 voxels
  if verbose: print('\ninterpolationg input volumes to 1mm^3 voxels')
  zoomfacs            = mr_voxsize / training_voxsize
  mr_vol_interpolated          = zoom3d(mr_vol, zoomfacs)

  # this is the final affine that maps from the PET grid to interpolated MR grid 
  # using the small voxels used during training 
  pet_mr_interp_aff   = regis_aff @ np.diag(np.concatenate((1./zoomfacs,[1])))

  pet_vol_mr_grid_interpolated = aff_transform(pet_vol, pet_mr_interp_aff, mr_vol_interpolated.shape, cval = pet_vol.min()) 
 
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

  ############################
  # make the actual prediction
  ############################
  
  if verbose: print('\npredicting the bowsher')

  if patchsize is None:
    # case of predicting the whole volume in one big chunk
    # bring the input volumes in the correct shape for the model
    x = [np.expand_dims(np.expand_dims(pet_vol_mr_grid_interpolated,0),-1), np.expand_dims(np.expand_dims(mr_vol_interpolated,0),-1)]
    predicted_bow = model.predict(x).squeeze() 
  else:
    # case of doing the prediction in multiple smaller 3D chunks (patches)
    predicted_bow = np.zeros(pet_vol_mr_grid_interpolated.shape, dtype = np.float32)

    for i in range(pet_vol_mr_grid_interpolated.shape[0]//patchsize[0] + 1):
      for j in range(pet_vol_mr_grid_interpolated.shape[1]//patchsize[1] + 1):
        for k in range(pet_vol_mr_grid_interpolated.shape[2]//patchsize[2] + 1):
          istart = max(i*patchsize[0] - overlap, 0)
          jstart = max(j*patchsize[1] - overlap, 0)
          kstart = max(k*patchsize[2] - overlap, 0)

          ioffset = i*patchsize[0] - istart
          joffset = j*patchsize[1] - jstart
          koffset = k*patchsize[2] - kstart

          iend  = min(((i+1)*patchsize[0] + overlap), pet_vol_mr_grid_interpolated.shape[0])
          jend  = min(((j+1)*patchsize[1] + overlap), pet_vol_mr_grid_interpolated.shape[1])
          kend  = min(((k+1)*patchsize[2] + overlap), pet_vol_mr_grid_interpolated.shape[2])

          pet_patch  = pet_vol_mr_grid_interpolated[istart:iend,jstart:jend,kstart:kend]
          mr_patch   = mr_vol_interpolated[istart:iend,jstart:jend,kstart:kend]
         
          # make the prediction
          x = [np.expand_dims(np.expand_dims(pet_patch,0),-1), np.expand_dims(np.expand_dims(mr_patch,0),-1)]
          tmp = model.predict(x).squeeze() 

          ntmp0  = min((i+1)*patchsize[0], pet_vol_mr_grid_interpolated.shape[0]) - i*patchsize[0]
          ntmp1  = min((j+1)*patchsize[1], pet_vol_mr_grid_interpolated.shape[1]) - j*patchsize[1]
          ntmp2  = min((k+1)*patchsize[2], pet_vol_mr_grid_interpolated.shape[2]) - k*patchsize[2]
          
          predicted_bow[i*patchsize[0]:(i*patchsize[0] + ntmp0),j*patchsize[1]:(j*patchsize[1] + ntmp1), k*patchsize[2]:(k*patchsize[2]+ntmp2)] = tmp[ioffset:(ioffset+ntmp0),joffset:(joffset+ntmp1),koffset:(koffset+ntmp2)]

  if clip_neg: np.clip(predicted_bow, 0, None, predicted_bow)
  
  # unnormalize the data
  if verbose: print('\nunnormalizing the images')
  mr_vol_interpolated          *= mr_max
  pet_vol_mr_grid_interpolated *= pet_max
  predicted_bow       *= pet_max

  # safe the input volumes in case of debug mode 
  if debug_mode: 
    np.savez_compressed(odir + '_debug.npz', 
                        mr_vol = mr_vol, 
                        mr_vol_interpolated = mr_vol_interpolated,
                        pet_vol = pet_vol,
                        pet_vol_mr_grid_interpolated = pet_vol_mr_grid_interpolated,
                        predicted_bow = predicted_bow,
                        mr_affine = mr_affine, pet_affine = pet_affine,
                        pet_mr_interp_aff = pet_mr_interp_aff,
                        training_voxsize = training_voxsize)

  print('\n\n------------------------------------------')
  print('------------------------------------------')
  print('\nCNN prediction finished')


  ##############################################################
  ########## write the output as nifti, png, dcm ###############
  ##############################################################

  # write output pngs
  pmax = np.percentile(pet_vol_mr_grid_interpolated, 99.99)
  mmax = np.percentile(mr_vol_interpolated, 99.99)
  imshow_kwargs = [{'cmap':py.cm.Greys_r, 'vmin':0, 'vmax':mmax},
                   {'cmap':py.cm.Greys,   'vmin':0, 'vmax':pmax},
                   {'cmap':py.cm.Greys,   'vmin':0, 'vmax':pmax}]

  vi = ThreeAxisViewer([mr_vol_interpolated, pet_vol_mr_grid_interpolated, predicted_bow], 
                       imshow_kwargs = imshow_kwargs, ls = '')
  vi.fig.savefig(odir + '.png')
  py.close(vi.fig)
  py.close(vi.fig_cb)
  py.close(vi.fig_sl)

  #---------------------------------------------------------------
  # generate the output affines
  if output_on_pet_grid:
    output_affine = pet_affine.copy()
    predicted_bow = aff_transform(predicted_bow, np.linalg.inv(pet_mr_interp_aff), 
                                  pet_vol.shape, cval = pet_vol.min()) 
  else:
    output_affine = mr_affine.copy()
    for i in range(3):  output_affine[i,:-1] *= training_voxsize[i]/np.sqrt((output_affine[i,:-1]**2).sum())

  # create the output affine in RAS orientation to save niftis
  output_affine_ras       = output_affine.copy()
  output_affine_ras[0,-1] = (-1 * output_affine @ np.array([predicted_bow.shape[0]-1,0,0,1]))[0]
  output_affine_ras[1,-1] = (-1 * output_affine @ np.array([0,predicted_bow.shape[1]-1,0,1]))[1]
   
  #------------------------------------------------------------
  # write a simple nifti as fall back in case the dicoms are not working
  # keep in mind that nifti used RAS internally
  nib.save(nib.Nifti1Image(np.flip(np.flip(predicted_bow,0),1), output_affine_ras), odir + '.nii')
  print('\nWrote nifti:') 
  print(odir + '.nii\n')
  
  #------------------------------------------------------------
  # write the dicoms
  if input_format == 'dicom': 
    # read the reference PET dicom file to copy some header tags
    refdcm = pydicom.read_file(pet_files[0])
    dcm_kwargs   = {}
    # copy the following tags if present in the reference dicom 
    pet_keys_to_copy = ['AcquisitionDate',
                        'AcquisitionTime',
                        'ActualFrameDuration',
                        'AccessionNumber',
                        'DecayCorrection',
                        'DecayCorrectionDateTime',
                        'DecayFactor',
                        'DoseCalibrationFactor',
                        'FrameOfReferenceUID',
                        'FrameReferenceTime',
                        'InstitutionName',
                        'ManufacturerModelName',
                        'OtherPatientIDs',
                        'PatientAge',
                        'PatientBirthDate',
                        'PatientID',
                        'PatientName',
                        'PatientPosition',
                        'PatientSex',
                        'PatientWeight',
                        'ProtocolName',
                        'RadiopharmaceuticalInformationSequence',
                        'RescaleType',
                        'SeriesDate',
                        'SeriesTime',
                        'StudyDate',
                        'StudyDescription',
                        'StudyID',
                        'StudyInstanceUID',
                        'StudyTime',
                        'Units']
    
    for key in pet_keys_to_copy:
      try:
        dcm_kwargs[key] = getattr(refdcm,key)
      except AttributeError:
        warn('Cannot copy tag ' + key)
    
    # write the dicom volume  
    write_3d_static_dicom(predicted_bow, odir, affine = output_affine,
                          ReconstructionMethod = ReconstructionMethod, SeriesDescription = SeriesDescription,
                          **dcm_kwargs)
    print('\nWrote dicom folder:')
    print(odir,'\n')

  print('------------------------------------------')
  print('------------------------------------------')

#--------------------------------------------------------------------------------------

def predict_single_case(model, input_fnames, verbose = False, **kwargs):
  """
  predict a single case using an apetnet model

  inputs
  ------

  model          ... a keras apetnet model 

  inputfile_list ... (list) of input channels passed to PathSequence
                      [input_channel_1, input_channel_2, ...]


  Keyword arguments
  -----------------

  verbose        ... print verbose output 

  **kwargs       ... passed to PatchSequence
  """

  ps = PatchSequence([input_fnames], **kwargs)
 
  x =  [np.expand_dims(z,0) for z in ps.input_vols[0][:len(model.input_shape)]]

  t_start = time()
  prediction = model.predict(x).squeeze()
  t_stop = time()

  if verbose: print('\ntime needed for prediction (s): ', t_stop - t_start, '\n')

  # undo the normalization
  prediction *= ps.slopes[0][0]
  prediction += ps.intercepts[0][0]
  
  ps.unnormalize()

  return (ps, prediction)

#--------------------------------------------------------------------------------------

def predict_eval_single_case(models, input_fnames, target_fname, verbose = False, **kwargs):
  """
  predict a single case using an apetnet model

  inputs
  ------

  models         ... a list of keras models

  inputfile_list ... (list) of input channels passed to PathSequence
                      [input_channel_1, input_channel_2, ...]

  target_fname   ... name of reference (target) file

  Keyword arguments
  -----------------

  verbose        ... print verbose output 

  **kwargs       ... passed to PatchSequence
  """

  ps = PatchSequence([input_fnames], [target_fname], **kwargs)
 
  x =  [np.expand_dims(z,0) for z in ps.input_vols[0][:len(models[0].input_shape)]]

  predictions = []

  for model in models:
    t_start = time()
    prediction = model.predict(x).squeeze()
    t_stop = time()

    if verbose: print('\ntime needed for prediction (s): ', t_stop - t_start, '\n')

    # undo the normalization
    prediction *= ps.slopes[0][0]
    prediction += ps.intercepts[0][0]
  
    predictions.append(prediction)

  ps.unnormalize()

  return (ps, predictions)


