# demo script that shows how to setup patch based batch sampler for pytorch using torchio

import torch
import torchio as tio
from torch.utils.data import DataLoader
from time import time
import pathlib
import numpy as np
import nibabel as nib
import os
from scipy.ndimage import find_objects

def read_nifty(path, percentile = 99):
  nii = nib.as_closest_canonical(nib.load(path))
  vol = np.expand_dims(nii.get_fdata(),0)

  return vol, nii.affine

#------------------------------------------------------------------------------------------------

data_dir            = pathlib.Path('../data/training_data/brainweb/brainweb_petmr')
ntrain              = 14
training_batch_size = 20
num_workers         = os.cpu_count() - 1
patch_size          = 29
samples_per_volume  = 64
max_queue_length    = ntrain*samples_per_volume

#-------------------------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

sdirs = list(data_dir.glob('subject??'))

training_subjects   = []
validation_subjects = []

for i, sdir in enumerate(sdirs):

  pet_low, pet_low_aff = read_nifty(sdir / 'sim_0' / 'osem_psf_counts_1.0E+07.nii.gz')
  pet_high, pet_high_aff = read_nifty(sdir / 'sim_0' / 'true_pet.nii.gz')
  mr, mr_aff = read_nifty(sdir / 't1.nii.gz')

  # crop volumes
  bbox = find_objects(mr > mr.mean())[0]
  pet_low  = pet_low[bbox]
  pet_high = pet_high[bbox]
  mr       = mr[bbox]

  # rescale volumes
  mr_scale  = np.percentile(mr, 99)
  pet_scale = np.percentile(pet_low[mr > mr.mean()], 95)

  pet_low /= pet_scale
  pet_high /= pet_scale
  mr /= mr_scale

  subject = tio.Subject(pet_low  = tio.ScalarImage(tensor = pet_low, affine = pet_low_aff),
                        pet_high = tio.ScalarImage(tensor = pet_high, affine = pet_high_aff),
                        mr       = tio.ScalarImage(tensor = mr, affine = mr_aff))

  if i < ntrain:
    training_subjects.append(subject)
    print('training', sdir)
  else:
    validation_subjects.append(subject)
    print('validation', sdir)

#-------------------------------------------------------------------------------------------------

training_transforms = tio.Compose([tio.ToCanonical(),
                                   tio.RandomAffine(degrees = (-20,20)),
                                   tio.RandomGamma(log_gamma = (-1,1), include = ['mr'])])

validation_transforms = tio.Compose([tio.ToCanonical()])

training_set   = tio.SubjectsDataset(training_subjects, transform = training_transforms)
validation_set = tio.SubjectsDataset(validation_subjects, transform = validation_transforms)

#-------------------------------------------------------------------------------------------------

sampler = tio.data.UniformSampler(patch_size)

patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=2*samples_per_volume,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=True,
    shuffle_patches=True,
)

training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)


num_epochs = 3
model = torch.nn.Identity()

for epoch_index in range(num_epochs):
  print(epoch_index)
  ta = time()  
  for patches_batch in training_loader_patches:
    x0 = patches_batch['pet_low'][tio.DATA].to(device)
    x1 = patches_batch['mr'][tio.DATA].to(device)
    y  = patches_batch['pet_high'][tio.DATA].to(device)

  print(time() - ta)
