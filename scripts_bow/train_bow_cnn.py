import sys
if not '..' in sys.path: sys.path.append('..')

import pathlib
import numpy as np
import os
import h5py
import json
import shutil
import random

import tensorflow as tf

from glob         import glob
from datetime     import datetime

import tensorflow
if tensorflow.__version__ >= '2':
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.models     import load_model
  from tensorflow.keras.callbacks  import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
  from tensorflow.keras.utils      import model_to_dot
  from tensorflow.keras.utils      import multi_gpu_model
else: 
  from keras.optimizers      import Adam
  from keras.models          import load_model
  from keras.callbacks       import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
  from keras.utils.vis_utils import model_to_dot
  from keras.utils           import multi_gpu_model

from pyapetnet.generators import PatchSequence, petmr_brain_data_augmentation
from pyapetnet.models     import apetnet

np.random.seed(42)

# check if we have an X display
has_x_disp = os.getenv('DISPLAY') is not None 

#------------------------------------------------------------------------------------------------
# parse the command line

from argparse import ArgumentParser

parser = ArgumentParser(description = 'Train APETNET')
parser.add_argument('--cfg_file', default = 'train_cfg.json',  help = 'training config file')

args = parser.parse_args()

#------------------------------------------------------------------------------------------------

# read and process the config file
with open(args.cfg_file) as f:
  cfg = json.load(f)

# input parameters
n_epochs         = cfg['n_epochs']         # number of epochs to train (around 300 is reasonable)
steps_per_epoch  = cfg['steps_per_epoch']  # training steps per epoch
batch_size       = cfg['batch_size']       # batch size in training
val_batch_size   = cfg['val_batch_size']   # batch size in training
patch_size       = (cfg['patch_size'],)*3      # patch size for training batches
val_patch_size   = (cfg['val_patch_size'],)*3  # patch size for validation data
learning_rate    = cfg['learning_rate']        # learning rate
model_kwargs     = cfg['model_kwargs']
data_aug_kwargs  = cfg['data_aug_kwargs']
output_suffix    = cfg['output_suffix']
masterlogdir     = cfg['masterlogdir'] 
internal_voxsize = cfg['internal_voxsize']*np.ones(3) # internal voxsize (mm)

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
                       batch_size = val_batch_size, patch_size = val_patch_size,
                       internal_voxsize = internal_voxsize)


# for the validation we only use the first patch
validation_data = val_ps.__getitem__(0)

#-----------------------------------------------------------------------------------------------
# set up the log dir
pathlib.Path(masterlogdir).mkdir(exist_ok = True)
time_str          = str(datetime.now())[:-7].replace(' ','_').replace(':','_')
tmp_logdir        = os.path.join(masterlogdir, time_str + '_' + output_suffix)
pathlib.Path(tmp_logdir).mkdir(exist_ok = True)
checkpoint_path   = os.path.join(tmp_logdir, 'cnn_bow_check.h5')
output_model_file = os.path.join(tmp_logdir, 'trained_model.h5')

#-----------------------------------------------------------------------------------------------
# set up the model to train

# define not parallized model on CPU
with tf.device('/cpu:0'):
  model = apetnet(**model_kwargs)

parallel_model = multi_gpu_model(model, gpus = 4, cpu_merge = False)

parallel_model.compile(optimizer = Adam(lr = learning_rate), loss = 'mse')

# plot the model as svg - only works if we have an X display
if has_x_disp:
  with open(os.path.join(tmp_logdir, 'model_graph.svg'),'w') as ff:
    print("{}".format(model_to_dot(model, show_shapes = True).create(prog='dot', format='svg').decode("utf-8")), 
          file = ff)

#----------------------------------------------------------------------------------------------
# define the keras call backs

checkpoint = ModelCheckpoint(checkpoint_path, 
                             monitor           = 'val_loss', 
                             verbose           = 1, 
                             save_best_only    = True, 
                             save_weights_only = False, 
                             mode              ='min')
          
tensorboard = TensorBoard(log_dir                = tmp_logdir, 
                          histogram_freq         = 0, 
                          batch_size             = batch_size, 
                          write_graph            = True, 
                          write_grads            = False, 
                          write_images           = False, 
                          embeddings_freq        = 0, 
                          embeddings_layer_names = None, 
                          embeddings_metadata    = None)

csvlog    = CSVLogger(os.path.join(tmp_logdir,'log.csv'))

# reduce learning rate by a factor of 2 if validation loss does not improve for 1000 epochs
lr_reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 100, 
                              verbose = 1, min_lr = 1e-4) 

#-----------------------------------------------------------------------------------------------
# train the model
parallel_model.fit_generator(ps, 
                             steps_per_epoch  = steps_per_epoch, 
                             epochs           = n_epochs, 
                             verbose          = 1, 
                             callbacks        = [checkpoint, tensorboard, csvlog, lr_reduce],
                             validation_data  = validation_data,
                             validation_steps = 1)

shutil.copyfile(checkpoint_path, output_model_file)

#-----------------------------------------------------------------------------------------------
# write a small header containing the training / validation parameters

with h5py.File(output_model_file) as hf:
  grp = hf.create_group('header/training_input_fnames')
  for i in range(len(ps.input_fnames)): 
    for ch in range(len(ps.input_fnames[i])): 
      grp['input_fname_' + str(i).zfill(3) + '_' + str(ch).zfill(3)] = os.path.abspath(ps.input_fnames[i][ch])

  grp = hf.create_group('header/training_target_fnames')
  for i in range(len(ps.target_fnames)): 
    grp['target_fname_' + str(i).zfill(3)] = os.path.abspath(ps.target_fnames[i])

  grp = hf.create_group('header/validation_input_fnames')
  for i in range(len(val_ps.input_fnames)): 
    for ch in range(len(val_ps.input_fnames[i])): 
      grp['input_fname_' + str(i).zfill(3) + '_' + str(ch).zfill(3)] = os.path.abspath(val_ps.input_fnames[i][ch])

  grp = hf.create_group('header/validation_target_fnames')
  for i in range(len(val_ps.target_fnames)): 
    grp['target_fname_' + str(i)] = os.path.abspath(val_ps.target_fnames[i])

  hf['header/patch_size']       = patch_size
  hf['header/batch_size']       = batch_size
  hf['header/n_epochs']         = n_epochs
  hf['header/steps_per_epochs'] = steps_per_epoch
  hf['header/internal_voxsize'] = internal_voxsize

#-----------------------------------------------------------------------------------------------
# show final prediction of validation data
if has_x_disp:
  from pyapetnet.threeaxisviewer import ThreeAxisViewer
  p = model.predict(validation_data[0])

  imshow_kwargs = {'vmin':0, 'vmax':1.2}
  vi = ThreeAxisViewer([validation_data[0][0].squeeze(),p.squeeze(),validation_data[1].squeeze()], 
                        imshow_kwargs = imshow_kwargs)
