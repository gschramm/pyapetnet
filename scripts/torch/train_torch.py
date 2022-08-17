""" train pyapetnet using torch and torch_io"""
import os
import pathlib
import json
from argparse import ArgumentParser

import torch
import torchio as tio

import numpy as np
from scipy.ndimage import find_objects

from fileio import read_nifty
from models import SequentialStructureConvNet, SimpleBlockGenerator

if __name__ == '__main__':
    parser = ArgumentParser(description='Train APETNET')
    parser.add_argument('data_dir')
    parser.add_argument('--cfg_file',
                        default='train_cfg.json',
                        help='training config file')

    args = parser.parse_args()

    data_dir = args.data_dir
    num_workers = os.cpu_count() - 1
    petOnly = False
    # number of slices (discard lower slices in neck region)
    nsl = 150

    # read and process the config file
    with open(args.cfg_file) as f:
        cfg = json.load(f)

    n_epochs = cfg['n_epochs']
    patch_size = cfg['patch_size']
    internal_voxsize = np.full(3, cfg['internal_voxsize'])
    lr = cfg['learning_rate']
    training_batch_size = cfg['batch_size']
    training_patch_size = cfg['patch_size']

    validation_batch_size = cfg['val_batch_size']
    validation_patch_size = cfg['val_patch_size']

    samples_per_volume = cfg['samples_per_volume']

    if petOnly:
        num_input_ch = 1
    else:
        num_input_ch = 2

    #-------------------------------------------------------------------------------------------------
    training_subjects = []

    # load the training data
    for i, ts in enumerate(cfg['training_files']):
        print(i, ts)

        pet_low, pet_low_aff = read_nifty(pathlib.Path(data_dir) / ts[0],
                                          internal_voxsize=internal_voxsize)
        mr, mr_aff = read_nifty(pathlib.Path(data_dir) / ts[1],
                                internal_voxsize=internal_voxsize)
        pet_high, pet_high_aff = read_nifty(pathlib.Path(data_dir) / ts[2],
                                            internal_voxsize=internal_voxsize)

        # crop volumes
        bbox = find_objects(mr > mr.mean())[0]
        pet_low = pet_low[bbox][..., -nsl:]
        pet_high = pet_high[bbox][..., -nsl:]
        mr = mr[bbox][..., -nsl:]

        # rescale volumes
        mr_scale = np.percentile(mr, 99)
        pet_scale = np.percentile(pet_low[mr > mr.mean()], 95)

        pet_low /= pet_scale
        pet_high /= pet_scale
        mr /= mr_scale

        mask = (mr > mr.mean())

        # flip mr contrast
        if (i % 2 == 1):
            mr = np.clip(1.5 - mr, 0, None)

        training_subjects.append(
            tio.Subject(pet_low=tio.ScalarImage(tensor=pet_low,
                                                affine=pet_low_aff),
                        pet_high=tio.ScalarImage(tensor=pet_high,
                                                 affine=pet_high_aff),
                        mr=tio.ScalarImage(tensor=mr, affine=mr_aff),
                        mask=tio.ScalarImage(tensor=mask, affine=mr_aff)))

    #-------------------------------------------------------------------------------------------------
    validation_subjects = []

    # load the validation data
    for i, ts in enumerate(cfg['validation_files']):
        print(i, ts)

        pet_low, pet_low_aff = read_nifty(pathlib.Path(data_dir) / ts[0],
                                          internal_voxsize=internal_voxsize)
        mr, mr_aff = read_nifty(pathlib.Path(data_dir) / ts[1],
                                internal_voxsize=internal_voxsize)
        pet_high, pet_high_aff = read_nifty(pathlib.Path(data_dir) / ts[2],
                                            internal_voxsize=internal_voxsize)

        # crop volumes
        bbox = find_objects(mr > mr.mean())[0]
        pet_low = pet_low[bbox][..., -nsl:]
        pet_high = pet_high[bbox][..., -nsl:]
        mr = mr[bbox][..., -nsl:]

        # rescale volumes
        mr_scale = np.percentile(mr, 99)
        pet_scale = np.percentile(pet_low[mr > mr.mean()], 95)

        pet_low /= pet_scale
        pet_high /= pet_scale
        mr /= mr_scale

        validation_subjects.append(
            tio.Subject(pet_low=tio.ScalarImage(tensor=pet_low,
                                                affine=pet_low_aff),
                        pet_high=tio.ScalarImage(tensor=pet_high,
                                                 affine=pet_high_aff),
                        mr=tio.ScalarImage(tensor=mr, affine=mr_aff)))

    #-------------------------------------------------------------------------------------------------

    training_transforms = tio.Compose([
        tio.transforms.RandomFlip(axes=(0, 1, 2)),
        tio.RandomGamma(log_gamma=(-1, 1), include=['mr'])
    ])

    #tio.RandomAffine(degrees = (-10,10)),

    validation_transforms = tio.Compose([tio.CropOrPad((96, 96, 96))])

    training_set = tio.SubjectsDataset(training_subjects,
                                       transform=training_transforms)
    validation_set = tio.SubjectsDataset(validation_subjects,
                                         transform=validation_transforms)

    #-------------------------------------------------------------------------------------------------

    sampler = tio.data.WeightedSampler(patch_size, 'mask')

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=2 * samples_per_volume,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, 
        batch_size=training_batch_size,
        pin_memory = True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=len(validation_set),
        num_workers=num_workers,
        pin_memory = True
    )

    #--------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    bgen = SimpleBlockGenerator()
    model = SequentialStructureConvNet(num_input_ch, bgen, nfeat=4,
                                       nblocks=3).to(device)

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #--------------------------------------------------------------------------------------

    size = len(training_loader_patches.dataset)

    for i_epoch in range(n_epochs):
        print(f'epoch {i_epoch}')

        # training loop
        for i_batch, batch in enumerate(training_loader_patches):
            print(f'training batch {i_batch}')
            if petOnly:
                x = batch['pet_low'][tio.DATA].to(device)
            else:
                x0 = batch['pet_low'][tio.DATA]
                x1 = batch['mr'][tio.DATA]
                x = torch.cat((x0, x1), 1).to(device)

            y = batch['pet_high'][tio.DATA].to(device)

            # Compute prediction and loss
            pred = model(x)
            loss = loss_fn(pred, y)
            print(f'loss {loss.item()}')

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #-----------------------------------------------------------
        val_loss = 0.
        num_val_batches = len(validation_loader)

        with torch.no_grad():
            for i_batch, batch in enumerate(validation_loader):
                print(f'validation batch {i_batch}')
                if petOnly:
                    x = batch['pet_low'][tio.DATA].to(device)
                else:
                    x0 = batch['pet_low'][tio.DATA]
                    x1 = batch['mr'][tio.DATA]
                    x = torch.cat((x0, x1), 1).to(device)

                # Compute prediction and loss
                y = batch['pet_high'][tio.DATA].to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        val_loss /= num_val_batches
        print(f'validation loss {val_loss}')

#    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
#
#    model = APetNet(petOnly=petOnly)
#    trainer = pl.Trainer(gpus=int(torch.cuda.is_available()),
#                         min_epochs=0,
#                         max_epochs=n_epochs,
#                         callbacks=[checkpoint_callback])
#    trainer.fit(model, training_loader_patches, validation_loader)
#
#    print(checkpoint_callback.best_model_path)
#
#    #--------------------------------------------------------------------------------------
#    import pymirc.viewer as pv
#
#    model = APetNet.load_from_checkpoint(checkpoint_callback.best_model_path)
#
#    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
#    model.to(device)
#
#    vis = []
#
#    for i, vsub in enumerate(validation_subjects):
#        if petOnly:
#            x = vsub['pet_low'][tio.DATA].unsqueeze(0).to(device)
#        else:
#            x0 = vsub['pet_low'][tio.DATA].unsqueeze(0)
#            x1 = vsub['mr'][tio.DATA].unsqueeze(0)
#            x = torch.cat((x0, x1), 1).to(device)
#
#        with torch.no_grad():
#            p = model.forward(x).detach().cpu().numpy().squeeze()
#
#        y = vsub['pet_high'][tio.DATA].numpy().squeeze()
#
#        vis.append(
#            pv.ThreeAxisViewer([x[0, 0, ...].cpu().numpy().squeeze(), p, y],
#                               imshow_kwargs={
#                                   'vmin': 0,
#                                   'vmax': 1.5
#                               }))
#
