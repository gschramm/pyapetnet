import h5py
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from bow_generator import BOWSequence
from apetnet2d import apetnet2d

from datetime import datetime

import matplotlib as mpl
try:
  mpl.use('Qt5Agg')
except:
  mpl.use('Agg')
import matplotlib.pyplot as py

def ssim_loss(y_true, y_pred):

    '''
    Equal to 1 minus the ssim coefficient
    '''
    #calc dyn rage dr

    #dr = y_true.numpy().max()-y_true.numpy().min()
    dr = tf.math.reduce_max(y_true) - tf.math.reduce_min(y_true)

    return 1 - tf.image.ssim(y_true, y_pred, dr)


#---------------------------------------------------------------------------------------------
#--- parse command line for input arguments --------------------------------------------------
#---------------------------------------------------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description='BOW training')
parser.add_argument('--batch_size',       default = 50,     type = int)
parser.add_argument('--epochs',           default = 5,      type = int)
parser.add_argument('--oname',            default = None,   type = str)
parser.add_argument('--loss_fct',         default = 'ssim', type = str, choices = ('mse','ssim'))
parser.add_argument('--n_ind_layers',     default = 1,      type = int)
parser.add_argument('--n_common_layers',  default = 7,      type = int)
parser.add_argument('--n_kernels_ind',    default = 15,     type = int)
parser.add_argument('--n_kernels_common', default = 30,     type = int)
parser.add_argument('--patch_size',       default = None,   type = int)
parser.add_argument('--margin',           default = 0,      type = int)
parser.add_argument('--add_final_relu',   action = 'store_true')

batch_size       = args.batch_size
epochs           = args.epochs
oname            = args.oname
n_ind_layers     = args.n_ind_layers
n_common_layers  = args.n_common_layers
n_kernels_ind    = args.n_kernels_ind
n_kernels_common = args.n_kernels_common
patch_size       = args.patch_size
margin           = args.margin
add_final_relu   = args.add_final_relu

import pdb; pdb.set_trace()

learning_rate = 1e-3
lr_reduce_fac = 0.75
lr_patience   = 10
min_lr        = 1e-4
loss_fct      = args.loss_fct



if oname is None:
  dt_str = datetime.now().strftime("%Y%d%m_%H%M%S")
  oname  = f"bow2d_{dt_str}_ne_{epochs}_bs_{batch_size}_nki_{n_kernels_ind}_nkc_{n_kernels_common}_nli_{n_ind_layers}_nci_{n_common_layers}_loss_{loss_fct}_patchsize_{patch_size}_margin_{margin}.h5"
  oname  = os.path.join('trained_models', oname)
  os.makedirs('trained_models', exist_ok = True)

#---------------------------------------------------------------------------------------------

h5data = h5py.File('bow_2d_sim_beta_5_counts_1e7.h5', 'r')

# get all the groups in the hdf5 file
groups         = list(h5data.keys())

# get all groups containing image data (not the header)
subjects       = list(filter(lambda x: x.startswith('s_'), groups))

# get a two digit subject number (string) to sort the subject according to number
subject_number = [x[2:].zfill(2) for x in subjects]

# sort the subject strings
subjects_sorted = [x for _,x in sorted(zip(subject_number,subjects))]

# lists to temp. store the images
osem_3d = []
t1_3d   = []
bow_3d  = []

# loop over all subjects and read the data
for subject in subjects_sorted:
  print(subject)
 
  # get the simulated OSEM recons
  osem_3d.append(h5data[subject]['osem_3d'][:])
  # get the simulated T1 MR
  t1_3d.append(h5data[subject]['t1_3d'][:])
  # get the simulated PET recon with bowsher prior
  bow_3d.append(h5data[subject]['bow_3d'][:])


# convert the lists to numpy "batch arrays" of 2D images
osem_3d = np.array(osem_3d)
osem_3d = osem_3d.reshape((osem_3d.shape[0]*osem_3d.shape[1]),osem_3d.shape[2],osem_3d.shape[3])

t1_3d = np.array(t1_3d)
t1_3d = t1_3d.reshape((t1_3d.shape[0]*t1_3d.shape[1]),t1_3d.shape[2],t1_3d.shape[3])

bow_3d = np.array(bow_3d)
bow_3d = bow_3d.reshape((bow_3d.shape[0]*bow_3d.shape[1]),bow_3d.shape[2],bow_3d.shape[3])


# add last dimension for features
osem_3d = np.expand_dims(osem_3d, -1) 
t1_3d   = np.expand_dims(t1_3d, -1) 
bow_3d  = np.expand_dims(bow_3d, -1) 


nim = osem_3d.shape[0]

# normalize data

norm_pet = np.zeros(nim)
norm_t1  = np.zeros(nim)


for i in range(nim):
  norm_pet[i] = np.percentile(osem_3d[i,...],99.9)
  norm_t1[i]  = np.percentile(t1_3d[i,...],99.9)

  osem_3d[i,...]  /= norm_pet[i]
  t1_3d[i,...]  /= norm_t1[i]
  bow_3d[i,...]  /= norm_pet[i]



# split data
tr_osem = osem_3d[0:int(0.6*nim),...]
val_osem = osem_3d[int(0.6*nim): int(0.8*nim),...]
test_osem = osem_3d[int(0.8*nim):,...]

tr_t1 = t1_3d[0:int(0.6*nim),...]
val_t1 = t1_3d[int(0.6*nim): int(0.8*nim),...]
test_t1 = t1_3d[int(0.8*nim):,...]

tr_bow = bow_3d[0:int(0.6*nim),...]
val_bow = bow_3d[int(0.6*nim): int(0.8*nim),...]
test_bow = bow_3d[int(0.8*nim):,...]

# shuffle the training data
inds_shuffled = np.random.permutation(np.arange(tr_osem.shape[0]))
tr_osem = tr_osem[inds_shuffled,...]
tr_t1   = tr_t1[inds_shuffled,...]
tr_bow  = tr_bow[inds_shuffled,...]

# setup data generator
bow_gen = BOWSequence(tr_osem, tr_t1, tr_bow, batch_size, patch_size = patch_size, margin = margin)

#---------------------------------------------------------------------------------------------
#--- setup and train the model ---------------------------------------------------------------
#---------------------------------------------------------------------------------------------

if loss_fct == 'mse':
 loss = keras.losses.mse
else:
 loss = ssim_loss

model = apetnet2d(n_ch = 2, n_ind_layers = n_ind_layers, 
                  n_common_layers = n_common_layers, 
                  n_kernels_ind = n_kernels_ind, 
                  n_kernels_common = n_kernels_common,
                  add_final_relu = add_final_relu)




model.compile(optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
              loss = loss, metrics = [ssim_loss])


# define a callback that reduces the learning rate
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor  = 'val_loss',
                                              factor   = lr_reduce_fac,
                                              patience = lr_patience,
                                              min_lr   = min_lr)
#-----------------------------------------------------------------
# train the model

mc = keras.callbacks.ModelCheckpoint(oname, monitor='val_loss', mode='min',
                                     save_best_only=True, verbose = 1)

history = model.fit(bow_gen,
                    epochs              = epochs,
                    validation_data     = ((val_osem, val_t1), val_bow),
                    callbacks           = [reduce_lr, mc])


# save history arrays
np.save(oname.replace('.h5','_history.npy'), history.history)

#------------------------------------
# plot the loss functions

py.rcParams['axes.titlesize'] = 'medium'
fig3, ax3 = py.subplots(figsize = (4,4))
ax3.plot(history.history['loss'], label = 'train_loss')
ax3.plot(history.history['val_loss'], label = 'val_loss')
ax3.plot(history.history['ssim_loss'], label = 'train ssim')
ax3.plot(history.history['val_ssim_loss'], label = 'val ssim')
ax3.legend()
ax3.set_title('loss')
ax3.grid(ls = ':')
fig3.tight_layout()
fig3.savefig(oname.replace('.h5','_loss.png'))
fig3.show()




