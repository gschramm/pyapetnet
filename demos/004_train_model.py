# This is small demo on how to use keras fit_generator to train a CNN with NYU mMR data.
# It shows how to generate a PatchSequence object that takes care of of the data reading,
# data preprocessing and batch sampling.

import numpy as np
import os

from glob     import glob
from datetime import datetime

import tensorflow
if tensorflow.__version__ >= '2':
  from tensorflow.keras.optimizers import Adam
else:
  from keras.optimizers import Adam

from pyapetnet.generators import PatchSequence, petmr_brain_data_augmentation
from pyapetnet.models     import apetnet

np.random.seed(42)

#------------------------------------------------------------------------------------------------
# parse the command line

from argparse import ArgumentParser

parser = ArgumentParser(description = 'Train APETNET')

parser.add_argument('--n_epochs',        default = 5,    help = 'number of epochs', type = int)
parser.add_argument('--steps_per_epoch', default = 10,   help = 'steps per epoch', type = int)
parser.add_argument('--batch_size',      default = 30,   help = 'size of training mini batch', type = int)
parser.add_argument('--n_train',         default = 5,    help = 'number of training data sets', type = int)
parser.add_argument('--patch_size',      default = 19,   help = 'training patch size', type = int)
parser.add_argument('--learning_rate',   default = 1e-3, help = 'learning rate', type = float)

parser.add_argument('--dir_pattern',     
                    default = os.path.join('..','data','training_data','brainweb','subject??'), 
                    help = 'dir pattern of input images')

parser.add_argument('--mr_file', default = 't1.nii',  help = 'name of mr file')

args = parser.parse_args()

#------------------------------------------------------------------------------------------------

# input parameters
n_epochs        = args.n_epochs         # number of epochs to train (around 300 is reasonable)
steps_per_epoch = args.steps_per_epoch  # training steps per epoch
batch_size      = args.batch_size       # batch size in training
train_sl        = slice(0,args.n_train,None) # data sets used in training
patch_size      = (args.patch_size,)*3       # patch size for training batches
dir_pattern     = args.dir_pattern
mr_file         = args.mr_file  

#------------------------------------------------------------------------------------------------
# set up the list of file names and the PatchSequences

pdirs = sorted(glob(dir_pattern)) 

input_fnames  = []
target_fnames = []

for pdir in pdirs[train_sl]:
  pstr = str(int(pdir.split('subject')[-1])*1000)
  input_fnames.append([os.path.join(pdir, 'osem_psf_3_4_' + pstr + '.nii'),os.path.join(pdir, mr_file)])
  target_fnames.append(os.path.join(pdir, 'pet_' + pstr + '.nii'))

# this is the Sequence that we use for training
ps = PatchSequence(input_fnames, target_fnames = target_fnames, batch_size = batch_size,
                   patch_size = patch_size)

#-----------------------------------------------------------------------------------------------
# set up the model to train

model = apetnet(n_ch               = 2,        # number of input channels
                n_ind_layers       = 1,        # number of individual layers
                n_common_layers    = 7,        # number of common layers 
                n_kernels_ind      = 15,       # number of features in ind. layers
                n_kernels_common   = 30,       # number of features in common layers
                kernel_shape       = (3,3,3))  # kernel shapes

model.compile(optimizer = Adam(lr = args.learning_rate), loss = 'mse')

#-----------------------------------------------------------------------------------------------
# train the model
model.fit_generator(ps, 
                    steps_per_epoch  = steps_per_epoch, 
                    epochs           = n_epochs, 
                    verbose          = 1)

oname = 'myfirstmodel.h5'
model.save(oname)
print('saved model to: ', oname)
