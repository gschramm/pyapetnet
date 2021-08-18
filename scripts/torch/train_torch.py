import torch
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pathlib
import os
import json

import numpy as np
from scipy.ndimage import find_objects

from models import  APetNet
from fileio import read_nifty

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
from argparse import ArgumentParser

parser = ArgumentParser(description = 'Train APETNET')
parser.add_argument('--cfg_file', default = 'train_cfg.json',  help = 'training config file')

args = parser.parse_args()

#-------------------------------------------------------------------------------------------------

training_batch_size = 50
num_workers         = os.cpu_count() - 1
patch_size          = 29
samples_per_volume  = 10
petOnly             = False
internal_voxsize    = np.ones(3)
nsl                 = 150

# read and process the config file
with open(args.cfg_file) as f:
  cfg = json.load(f)


#-------------------------------------------------------------------------------------------------
training_subjects   = []

# load the training data
for i, ts in enumerate(cfg['training_files']):
  print(i, ts)

  pet_low, pet_low_aff = read_nifty(pathlib.Path(cfg['data_dir']) / ts[0], internal_voxsize = internal_voxsize)
  mr, mr_aff = read_nifty(pathlib.Path(cfg['data_dir']) / ts[1], internal_voxsize = internal_voxsize)
  pet_high, pet_high_aff = read_nifty(pathlib.Path(cfg['data_dir']) / ts[2], internal_voxsize = internal_voxsize)

  # crop volumes
  bbox = find_objects(mr > mr.mean())[0]
  pet_low  = pet_low[bbox][...,-nsl:]
  pet_high = pet_high[bbox][...,-nsl:]
  mr       = mr[bbox][...,-nsl:]

  # rescale volumes
  mr_scale  = np.percentile(mr, 99)
  pet_scale = np.percentile(pet_low[mr > mr.mean()], 95)

  pet_low /= pet_scale
  pet_high /= pet_scale
  mr /= mr_scale

  mask = (mr > mr.mean())

  # flip mr contrast
  if (i % 2 == 1):
    mr = np.clip(1.5 - mr, 0, None)

  training_subjects.append(tio.Subject(pet_low  = tio.ScalarImage(tensor = pet_low, affine = pet_low_aff),
                           pet_high = tio.ScalarImage(tensor = pet_high, affine = pet_high_aff),
                           mr       = tio.ScalarImage(tensor = mr, affine = mr_aff),
                           mask     = tio.ScalarImage(tensor = mask, affine = mr_aff)))

#-------------------------------------------------------------------------------------------------
validation_subjects = []

# load the training data
for i, ts in enumerate(cfg['validation_files']):
  print(i, ts)

  pet_low, pet_low_aff = read_nifty(pathlib.Path(cfg['data_dir']) / ts[0], internal_voxsize = internal_voxsize)
  mr, mr_aff = read_nifty(pathlib.Path(cfg['data_dir']) / ts[1], internal_voxsize = internal_voxsize)
  pet_high, pet_high_aff = read_nifty(pathlib.Path(cfg['data_dir']) / ts[2], internal_voxsize = internal_voxsize)

  # crop volumes
  bbox = find_objects(mr > mr.mean())[0]
  pet_low  = pet_low[bbox][...,-nsl:]
  pet_high = pet_high[bbox][...,-nsl:]
  mr       = mr[bbox][...,-nsl:]

  # rescale volumes
  mr_scale  = np.percentile(mr, 99)
  pet_scale = np.percentile(pet_low[mr > mr.mean()], 95)

  pet_low /= pet_scale
  pet_high /= pet_scale
  mr /= mr_scale

  validation_subjects.append(tio.Subject(pet_low  = tio.ScalarImage(tensor = pet_low, affine = pet_low_aff),
                             pet_high = tio.ScalarImage(tensor = pet_high, affine = pet_high_aff),
                             mr       = tio.ScalarImage(tensor = mr, affine = mr_aff)))


#-------------------------------------------------------------------------------------------------

training_transforms = tio.Compose([tio.transforms.RandomFlip(axes = (0,1,2)),
                                   tio.RandomGamma(log_gamma = (-1,1), include = ['mr'])])

                                   #tio.RandomAffine(degrees = (-10,10)),

validation_transforms = tio.Compose([tio.CropOrPad((96,96,96))])

training_set   = tio.SubjectsDataset(training_subjects, transform = training_transforms)
validation_set = tio.SubjectsDataset(validation_subjects, transform = validation_transforms)

#-------------------------------------------------------------------------------------------------

#sampler = tio.data.UniformSampler(patch_size)
sampler = tio.data.WeightedSampler(patch_size, 'mask')

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
trainer = pl.Trainer(gpus = 1, min_epochs = 0, max_epochs = 2000, callbacks = [checkpoint_callback])
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
  
  vis.append(pv.ThreeAxisViewer([x[0,0,...].cpu().numpy().squeeze(),p,y], imshow_kwargs={'vmin':0, 'vmax':1.5}))
