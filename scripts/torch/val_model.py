import torch
import torchio as tio
from torch.utils.data import DataLoader
from time import time
import pathlib
import numpy as np
import os
from scipy.ndimage import find_objects
import pytorch_lightning as pl

from models import APetNet
from fileio import read_nifty

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

data_dir = pathlib.Path('../../data/training_data/brainweb/brainweb_petmr')
ntrain = 14

#-------------------------------------------------------------------------------------------------
sdirs = list(data_dir.glob('subject??'))

validation_subjects = []

for i, sdir in enumerate(sdirs):
    if i >= ntrain:
        pet_low, pet_low_aff = read_nifty(sdir / 'sim_0' /
                                          'osem_psf_counts_1.0E+07.nii.gz')
        pet_high, pet_high_aff = read_nifty(sdir / 'sim_0' / 'true_pet.nii.gz')
        mr, mr_aff = read_nifty(sdir / 't1.nii.gz')

        # crop volumes
        bbox = find_objects(mr > mr.mean())[0]
        pet_low = pet_low[bbox]
        pet_high = pet_high[bbox]
        mr = mr[bbox]

        # rescale volumes
        mr_scale = np.percentile(mr, 99)
        pet_scale = np.percentile(pet_low[mr > mr.mean()], 95)

        pet_low /= pet_scale
        pet_high /= pet_scale
        mr /= mr_scale

        subject = tio.Subject(pet_low=tio.ScalarImage(tensor=pet_low,
                                                      affine=pet_low_aff),
                              pet_high=tio.ScalarImage(tensor=pet_high,
                                                       affine=pet_high_aff),
                              mr=tio.ScalarImage(tensor=mr, affine=mr_aff))

        validation_subjects.append(subject)
        print('validation', sdir)

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

model = APetNet().load_from_checkpoint(
    'lightning_logs/version_4/checkpoints/epoch=1014-step=69019.ckpt')

#--------------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model.to(device)

vis = []

for i, vsub in enumerate(validation_subjects):
    x0 = vsub['pet_low'][tio.DATA].unsqueeze(0)
    x1 = vsub['mr'][tio.DATA].unsqueeze(0)
    x = torch.cat((x0, x1), 1).to(device)

    with torch.no_grad():
        p = model.forward(x).detach().cpu().numpy().squeeze()

    y = vsub['pet_high'][tio.DATA].numpy().squeeze()

    import pymirc.viewer as pv
    vis.append(
        pv.ThreeAxisViewer(
            [np.flip(z, (0, 1)) for z in [x0.numpy().squeeze(), p, y]],
            imshow_kwargs={
                'vmin': 0,
                'vmax': 1
            }))
