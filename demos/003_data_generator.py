# this is a short demo on how to setup the PatchSequence that
# we use for training Keras models

import numpy as np
import os

from glob import glob

from pyapetnet.generators      import PatchSequence, petmr_brain_data_augmentation
from pyapetnet.threeaxisviewer import ThreeAxisViewer

#------------------------------------------------------------------------------------------------

np.random.seed(3)

batch_size = 10
n_subjects = 10

patch_size = (96,96,96)


dir_pattern = os.path.join('..','data','training_data','brainweb','subject??')
pdirs       = sorted(glob(dir_pattern)) 

ch0_files    = [os.path.join(pdir, 'ch-000.nii.gz') for pdir in pdirs]
ch1_files    = [os.path.join(pdir, 'ch-001.nii.gz') for pdir in pdirs]
target_files = [os.path.join(pdir, 'target.nii.gz') for pdir in pdirs]

input_fnames  = [[x,y] for x,y in zip(ch0_files[:n_subjects],ch1_files[:n_subjects])]
target_fnames = target_files[:n_subjects]

ps = PatchSequence(input_fnames, target_fnames = target_fnames, batch_size = batch_size,
                   patch_size = patch_size, data_aug_func = petmr_brain_data_augmentation, random_flip = True)

imshow_kwargs = {'vmin':0, 'vmax':1.1}

for ip in range(3):
  # lets get a batch of random patches from the patch sequence
  x,y = ps.__getitem__(0)
  # show the batch
  vi = ThreeAxisViewer([xx.squeeze() for xx in x] + [y.squeeze()], imshow_kwargs = imshow_kwargs)
  tmp = input('\nUse Left/Right arrow to go through batch items / Press enter to continue')
