import torch
import torchio as tio
from torch.utils.data import DataLoader
from time import time
import pathlib
import numpy as np
import os
from scipy.ndimage import find_objects
import pytorch_lightning as pl

from models import  APetNet
from fileio import read_nifty

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

data_dir            = pathlib.Path('../../data/training_data/brainweb/brainweb_petmr')
ntrain              = 14
training_batch_size = 20
num_workers         = os.cpu_count() - 1
patch_size          = 29
samples_per_volume  = 32
max_queue_length    = ntrain*samples_per_volume
petOnly             = False

#-------------------------------------------------------------------------------------------------
sdirs = list(data_dir.glob('subject??'))

training_subjects   = []
validation_subjects = []

for i, sdir in enumerate(sdirs):
  for sim in range(3):
    pet_low, pet_low_aff = read_nifty(sdir / f'sim_{sim}' / 'osem_psf_counts_1.0E+07.nii.gz')
    pet_high, pet_high_aff = read_nifty(sdir / f'sim_{sim}' / 'true_pet.nii.gz')
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

    # flip mr contrast
    if (i < ntrain) and (i % 2 == 1):
      mr = mr.max() - mr

    subject = tio.Subject(pet_low  = tio.ScalarImage(tensor = pet_low, affine = pet_low_aff),
                          pet_high = tio.ScalarImage(tensor = pet_high, affine = pet_high_aff),
                          mr       = tio.ScalarImage(tensor = mr, affine = mr_aff))

    if i < ntrain:
      training_subjects.append(subject)
      print('training', sdir, sim)
    else:
      validation_subjects.append(subject)
      print('validation', sdir, sim)

#-------------------------------------------------------------------------------------------------

training_transforms = tio.Compose([tio.transforms.RandomFlip(axes = (0,1,2)),
                                   tio.RandomAffine(degrees = (-20,20)),
                                   tio.RandomGamma(log_gamma = (-1,1), include = ['mr'])])

validation_transforms = tio.Compose([tio.CropOrPad((96,96,96))])

training_set   = tio.SubjectsDataset(training_subjects, transform = training_transforms)
validation_set = tio.SubjectsDataset(validation_subjects, transform = validation_transforms)

#-------------------------------------------------------------------------------------------------

sampler = tio.data.UniformSampler(patch_size)

patches_training_set = tio.Queue(
    subjects_dataset   = training_set,
    max_length         = 2*samples_per_volume,
    samples_per_volume = samples_per_volume,
    sampler            = sampler,
    num_workers        = num_workers,
    shuffle_subjects   = True,
    shuffle_patches    = True,
)

training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size = training_batch_size)

validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size  = len(validation_set),
    num_workers = num_workers,
)

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor = "val_loss")

model = APetNet(petOnly = petOnly)
trainer = pl.Trainer(gpus = 1, min_epochs = 0, max_epochs = 250, callbacks = [checkpoint_callback])
trainer.fit(model, training_loader_patches, validation_loader)

print(checkpoint_callback.best_model_path)

#--------------------------------------------------------------------------------------
import pymirc.viewer as pv

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model.to(device)

vis = []

for i, vsub in enumerate(validation_subjects):
  if petOnly:
    x  = vsub['pet_low'][tio.DATA].unsqueeze(0).to(device)
  else:
    x0 = vsub['pet_low'][tio.DATA].unsqueeze(0)
    x1 = vsub['mr'][tio.DATA].unsqueeze(0)
    x  = torch.cat((x0,x1),1).to(device)
  
  with torch.no_grad():
    p = model.forward(x).detach().cpu().numpy().squeeze()
  
  y = vsub['pet_high'][tio.DATA].numpy().squeeze()
  
  vis.append(pv.ThreeAxisViewer([x[0,0,...].cpu().numpy().squeeze(),p,y], imshow_kwargs={'vmin':0, 'vmax':1}))
