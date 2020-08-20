import sys
if not '..' in sys.path: sys.path.append('..')
import os

#------------------------------------------------------------------------------------------------
# parse the command line

from argparse import ArgumentParser

parser = ArgumentParser(description = 'Train APETNET')
parser.add_argument('--cfg_file', default = 'train_cfg.json',  help = 'training config file')

args = parser.parse_args()
#------------------------------------------------------------------------------------------------

import pathlib
import numpy as np
import h5py
import json
import shutil
import random
import warnings

import tensorflow as tf

from glob         import glob
from datetime     import datetime

import tensorflow
from tensorflow.python.client import device_lib

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models     import load_model
from tensorflow.keras.callbacks  import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils      import model_to_dot
from tensorflow.keras.utils      import multi_gpu_model
from pyapetnet.generators import PatchSequence, petmr_brain_data_augmentation
from pyapetnet.models     import apetnet, apetnet_vv5_onnx
from pyapetnet.losses     import ssim_3d_loss, mix_ssim_3d_mae_loss

np.random.seed(42)

# check if we have an X display
has_x_disp = os.getenv('DISPLAY') is not None 

#------------------------------------------------------------------------------------------------

# read and process the config file
with open(args.cfg_file) as f:
  cfg = json.load(f)

# input parameters
n_epochs         = cfg['n_epochs']         # number of epochs to train (around 300 is reasonable)
steps_per_epoch  = cfg['steps_per_epoch']  # training steps per epoch
batch_size       = cfg['batch_size']       # batch size in training
patch_size       = (cfg['patch_size'],)*3      # patch size for training batches
val_patch_size   = (cfg['val_patch_size'],)*3  # patch size for validation data
learning_rate    = cfg['learning_rate']        # learning rate
model_kwargs     = cfg['model_kwargs']
data_aug_kwargs  = cfg['data_aug_kwargs']
output_suffix    = cfg['output_suffix']
masterlogdir     = cfg['masterlogdir'] 
internal_voxsize = cfg['internal_voxsize']*np.ones(3) # internal voxsize (mm)
loss             = cfg['loss'] 

input_fnames      = []
target_fnames     = []
val_input_fnames  = []
val_target_fnames = []

# get the training and validation names
for train_files in cfg['training_files']:
  input_fnames.append(train_files[:-1])
  target_fnames.append(train_files[-1])

for vf in cfg['validation_files']:
  val_input_fnames.append(vf[:-1])
  val_target_fnames.append(vf[-1])

#shuffle the input list
rinds         = random.sample(range(len(input_fnames)),len(input_fnames))
input_fnames  = [input_fnames[x]  for x in rinds] 
target_fnames = [target_fnames[x] for x in rinds] 

rvinds            = random.sample(range(len(val_input_fnames)),len(val_input_fnames))
val_input_fnames  = [val_input_fnames[x]  for x in rvinds] 
val_target_fnames = [val_target_fnames[x] for x in rvinds] 

ps = PatchSequence(input_fnames, target_fnames = target_fnames, batch_size = batch_size,
                   patch_size = patch_size, data_aug_func = petmr_brain_data_augmentation, 
                   data_aug_kwargs = data_aug_kwargs, random_flip = True,
                   internal_voxsize = internal_voxsize, preload_data = True)

val_ps = PatchSequence(val_input_fnames, target_fnames = val_target_fnames, 
                       batch_size = batch_size, patch_size = val_patch_size,
                       internal_voxsize = internal_voxsize)


# for the validation we only use the first patch
validation_data = val_ps.get_input_vols_center_crop(val_patch_size + (1,), (0,0,0,0))

#-----------------------------------------------------------------------------------------------
# set up the log dir
pathlib.Path(masterlogdir).mkdir(exist_ok = True)
time_str          = str(datetime.now())[:-7].replace(' ','_').replace(':','_')
tmp_logdir        = os.path.join(masterlogdir, time_str + '_' + output_suffix)
pathlib.Path(tmp_logdir).mkdir(exist_ok = True)
checkpoint_path   = os.path.join(tmp_logdir, 'cnn_bow_check')
output_model_file = os.path.join(tmp_logdir, 'trained_model')

# copy the input config file to the logdir
shutil.copyfile(args.cfg_file, os.path.join(tmp_logdir,'config.json'))
#-----------------------------------------------------------------------------------------------
# set up the model to train

n_gpus = len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

if n_gpus >= 2:
  # define not parallized model on CPU
  with tf.device('/cpu:0'):
      model = apetnet(**model_kwargs)
  
  parallel_model = multi_gpu_model(model, gpus = n_gpus, cpu_merge = False)
else:
    parallel_model = apetnet(**model_kwargs)

if loss == 'ssim':
  loss    = ssim_3d_loss
elif loss == 'mix_ssim_mae':
  loss    = mix_ssim_3d_mae_loss

metrics = [ssim_3d_loss, mix_ssim_3d_mae_loss, 'mse', 'mae']

parallel_model.compile(optimizer = Adam(lr = learning_rate), loss = loss, metrics = metrics)

#----------------------------------------------------------------------------------------------
# define the keras call backs

checkpoint = ModelCheckpoint(checkpoint_path, 
                             monitor           = 'val_loss', 
                             verbose           = 1, 
                             save_best_only    = True, 
                             save_weights_only = False, 
                             mode              ='min')

csvlog    = CSVLogger(os.path.join(tmp_logdir,'log.csv'))

# reduce learning rate by a factor of 2 if validation loss does not improve for 1000 epochs
lr_reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 100, 
                              verbose = 1, min_lr = 1e-4) 

#-----------------------------------------------------------------------------------------------
# train the model
history = parallel_model.fit(x                = ps, 
                             epochs           = n_epochs, 
                             steps_per_epoch  = steps_per_epoch,
                             verbose          = 1,
                             callbacks        = [checkpoint, csvlog, lr_reduce],
                             validation_data  = validation_data,
                             validation_steps = 1)

shutil.copytree(checkpoint_path, output_model_file)
parallel_model.save(output_model_file + '_last')
parallel_model.save(output_model_file + '_last.h5')
