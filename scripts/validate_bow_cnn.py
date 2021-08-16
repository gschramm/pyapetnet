import sys
if not '..' in sys.path: sys.path.append('..')
import os

#------------------------------------------------------------------------------------------------
# parse the command line

from argparse import ArgumentParser

parser = ArgumentParser(description = 'Validate APETNET')
parser.add_argument('--log_dir',     help = 'log dir of training')
parser.add_argument('--patch_size',  help = 'patch size for center crop', default = None, type = int)
parser.add_argument('--interactive', help = 'open 3 interactive axis viewer', action = 'store_true')

args = parser.parse_args()
#------------------------------------------------------------------------------------------------

import pathlib
import numpy as np
import h5py
import json
import shutil
import random

import tensorflow as tf

from glob         import glob
from datetime     import datetime

import tensorflow
from tensorflow.python.client import device_lib

if tensorflow.__version__ >= '2':
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.models     import load_model
  from tensorflow.keras.callbacks  import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
  from tensorflow.keras.utils      import model_to_dot
  from tensorflow.keras.utils      import multi_gpu_model
  from tensorflow.keras.models     import load_model
else: 
  from keras.optimizers      import Adam
  from keras.models          import load_model
  from keras.callbacks       import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
  from keras.utils.vis_utils import model_to_dot
  from keras.utils           import multi_gpu_model
  from keras.models          import load_model
  from keras.utils.generic_utils import get_custom_objects

from pyapetnet.generators import PatchSequence, petmr_brain_data_augmentation
from pyapetnet.losses     import ssim_3d_loss, mix_ssim_3d_mae_loss

from scipy.ndimage import gaussian_filter

import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matplotlib.pyplot as py

np.random.seed(42)

#------------------------------------------------------------------------------------------------

# read and process the config file
with open(os.path.join(args.log_dir,'config.json')) as f:
  cfg = json.load(f)

# input parameters
val_batch_size   = cfg['val_batch_size']   # batch size in training
if args.patch_size is None:
  val_patch_size   = (cfg['val_patch_size'],)*3  # patch size for validation data
else:
  val_patch_size   = (args.patch_size,)*3  # patch size for validation data

internal_voxsize = cfg['internal_voxsize']*np.ones(3) # internal voxsize (mm)
loss             = cfg['loss'] 

val_input_fnames  = []
val_target_fnames = []

# get the training and validation names
for vf in cfg['validation_files']:
  val_input_fnames.append(vf[:-1])
  val_target_fnames.append(vf[-1])

val_ps = PatchSequence(val_input_fnames, target_fnames = val_target_fnames, 
                       batch_size = val_batch_size, patch_size = val_patch_size,
                       internal_voxsize = internal_voxsize)

# for the validation we only use the first patch
validation_data = val_ps.get_input_vols_center_crop(val_patch_size + (1,), (0,0,0,0))

#----------------------------------------------------------------------------------------------
# load the model
n_gpus = len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

# we have to test whether the model was saved in HDF5 or in protobuf
if os.path.exists(os.path.join(args.log_dir,'trained_model.h5')):
  model = load_model(os.path.join(args.log_dir,'trained_model.h5'), 
                     custom_objects={'ssim_3d_loss': ssim_3d_loss, 
                                     'mix_ssim_3d_mae_loss': mix_ssim_3d_mae_loss})
elif os.path.exists(os.path.join(args.log_dir,'trained_model')):
  model = load_model(os.path.join(args.log_dir,'trained_model'), 
                     custom_objects={'ssim_3d_loss': ssim_3d_loss, 
                                     'mix_ssim_3d_mae_loss': mix_ssim_3d_mae_loss})

#----------------------------------------------------------------------------------------------
# predict a simplistic phantom

x0 = (np.arange(149) - 149/2 + 0.5)*internal_voxsize[0]
x1 = (np.arange(149) - 149/2 + 0.5)*internal_voxsize[1]
x2 = (np.arange(149) - 149/2 + 0.5)*internal_voxsize[2]

X0,X1,X2 = np.meshgrid(x0, x1, x2, indexing = 'ij')

pet = np.zeros((149,149,149))
pet[np.sqrt(X0**2 + X1**2 + X2**2) < 70] = 0.5
pet[np.sqrt((X0-35)**2 + X1**2 + X2**2) < 10] = 1.
pet[np.sqrt((X0+35)**2 + X1**2 + X2**2) < 10] = 0
pet[np.sqrt(X0**2 + (X1-35)**2 + X2**2) < 5]  = 1.
pet[np.sqrt(X0**2 + (X1+35)**2 + X2**2) < 5]  = 0

mr  = (pet.max() - pet)**0.5

ph_fwhms = [4.5,5,5.5,6]
xp       = np.zeros((len(ph_fwhms),) + pet.shape)
xm       = np.zeros((len(ph_fwhms),) + pet.shape)

for i, ph_fwhm in enumerate(ph_fwhms):
  xp[i,...] = gaussian_filter(pet, ph_fwhm / (2.35*internal_voxsize))
  xm[i,...] = mr

x = [np.expand_dims(xp,-1), np.expand_dims(xm,-1)]
y = model.predict(x)

# plot results
imshow_kwargs = {'vmin':0, 'vmax':1.2}
sl0, sl1, sl2 = [x//2 for x in y.shape[1:-1]]

for i, ph_fwhm in enumerate(ph_fwhms):
  fig, ax = py.subplots(3,3, figsize = (7,7))
  ax[0,0].imshow(np.flip(x[0][i,:,:,sl2,0].T,1),            cmap = py.cm.Greys, **imshow_kwargs)
  ax[0,1].imshow(np.flip(np.flip(x[0][i,:,sl1,:,0].T,1),0), cmap = py.cm.Greys, **imshow_kwargs)
  ax[0,2].imshow(np.flip(np.flip(x[0][i,sl0,:,:,0].T,1),0), cmap = py.cm.Greys, **imshow_kwargs)
  ax[1,0].imshow(np.flip(y[i,:,:,sl2,0].T,1),               cmap = py.cm.Greys, **imshow_kwargs)
  ax[1,1].imshow(np.flip(np.flip(y[i,:,sl1,:,0].T,1),0),    cmap = py.cm.Greys, **imshow_kwargs)
  ax[1,2].imshow(np.flip(np.flip(y[i,sl0,:,:,0].T,1),0),    cmap = py.cm.Greys, **imshow_kwargs)
  ax[2,0].imshow(np.flip(x[1][i,:,:,sl2,0].T,1),            cmap = py.cm.Greys, **imshow_kwargs)
  ax[2,1].imshow(np.flip(np.flip(x[1][i,:,sl1,:,0].T,1),0), cmap = py.cm.Greys, **imshow_kwargs)
  ax[2,2].imshow(np.flip(np.flip(x[1][i,sl0,:,:,0].T,1),0), cmap = py.cm.Greys, **imshow_kwargs)
  
  fig.tight_layout()
  fig.savefig(os.path.join(args.log_dir,f'val_phantom_{ph_fwhm}mm.png'))
  py.close(fig)


#----------------------------------------------------------------------------------------------
# predict the validation data sets

p = model.predict(validation_data[0])

#-----------------------------------------------------------------------------------------------
# show final prediction of validation data
imshow_kwargs = {'vmin':0, 'vmax':1.2}
sl0, sl1, sl2 = [x//2 for x in p.shape[1:-1]]

for i in range(len(val_input_fnames)):
  fig, ax = py.subplots(3,3, figsize = (7,7))
  ax[0,0].imshow(np.flip(validation_data[0][0][i,:,:,sl2,0].T,1),            cmap = py.cm.Greys,**imshow_kwargs)
  ax[0,1].imshow(np.flip(np.flip(validation_data[0][0][i,:,sl1,:,0].T,1),0), cmap = py.cm.Greys,**imshow_kwargs)
  ax[0,2].imshow(np.flip(np.flip(validation_data[0][0][i,sl0,:,:,0].T,1),0), cmap = py.cm.Greys,**imshow_kwargs)
  ax[1,0].imshow(np.flip(p[i,:,:,sl2,0].T,1),                                cmap = py.cm.Greys,**imshow_kwargs)
  ax[1,1].imshow(np.flip(np.flip(p[i,:,sl1,:,0].T,1),0),                     cmap = py.cm.Greys,**imshow_kwargs)
  ax[1,2].imshow(np.flip(np.flip(p[i,sl0,:,:,0].T,1),0),                     cmap = py.cm.Greys,**imshow_kwargs)
  ax[2,0].imshow(np.flip(validation_data[1][i,:,:,sl2,0].T,1),               cmap = py.cm.Greys,**imshow_kwargs)
  ax[2,1].imshow(np.flip(np.flip(validation_data[1][i,:,sl1,:,0].T,1),0),    cmap = py.cm.Greys,**imshow_kwargs)
  ax[2,2].imshow(np.flip(np.flip(validation_data[1][i,sl0,:,:,0].T,1),0),    cmap = py.cm.Greys,**imshow_kwargs)

  fig.tight_layout()
  fig.savefig(os.path.join(args.log_dir,f'val_{i}.png'))
  py.close(fig)

if args.interactive:
  from pymirc.viewer import ThreeAxisViewer
  vi = ThreeAxisViewer([validation_data[0][0].squeeze(),p.squeeze(),validation_data[1].squeeze()], 
                        imshow_kwargs = imshow_kwargs)
