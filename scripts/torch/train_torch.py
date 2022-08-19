""" train pyapetnet using torch and torch_io"""
import os
import pathlib
import logging

import torch
import torchio as tio
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import numpy as np
from scipy.ndimage import find_objects

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from fileio import read_nifty
from models import SequentialStructureConvNet, SimpleBlockGenerator

from config import Config

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:

    if cfg.num_workers <= 0:
        num_workers = os.cpu_count() - 1
    else:
        num_workers = cfg.num_workers

    log.info(OmegaConf.to_yaml(cfg))

    # load the training data
    training_subjects = []

    for i, ts in enumerate(cfg.files.training):
        log.info(f'{i}, {ts}')

        pet_low, pet_low_aff = read_nifty(
            pathlib.Path(cfg.data_dir) / ts[0],
            internal_voxsize=cfg.internal_voxsize)
        mr, mr_aff = read_nifty(pathlib.Path(cfg.data_dir) / ts[1],
                                internal_voxsize=cfg.internal_voxsize)
        pet_high, pet_high_aff = read_nifty(
            pathlib.Path(cfg.data_dir) / ts[2],
            internal_voxsize=cfg.internal_voxsize)

        # crop volumes
        bbox = find_objects(mr > mr.mean())[0]
        pet_low = pet_low[bbox][..., -cfg.num_slices:]
        pet_high = pet_high[bbox][..., -cfg.num_slices:]
        mr = mr[bbox][..., -cfg.num_slices:]

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
    for i, ts in enumerate(cfg.files.validation):
        log.info(f'{i}, {ts}')

        pet_low, pet_low_aff = read_nifty(
            pathlib.Path(cfg.data_dir) / ts[0],
            internal_voxsize=cfg.internal_voxsize)
        mr, mr_aff = read_nifty(pathlib.Path(cfg.data_dir) / ts[1],
                                internal_voxsize=cfg.internal_voxsize)
        pet_high, pet_high_aff = read_nifty(
            pathlib.Path(cfg.data_dir) / ts[2],
            internal_voxsize=cfg.internal_voxsize)

        # crop volumes
        bbox = find_objects(mr > mr.mean())[0]
        pet_low = pet_low[bbox][..., -cfg.num_slices:]
        pet_high = pet_high[bbox][..., -cfg.num_slices:]
        mr = mr[bbox][..., -cfg.num_slices:]

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

    sampler = tio.data.WeightedSampler(cfg.patch_size, 'mask')

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=2 * cfg.samples_per_volume,
        samples_per_volume=cfg.samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, batch_size=cfg.batch_size, pin_memory=True)

    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=cfg.val_batch_size,
        num_workers=num_workers,
        pin_memory=True)

    #--------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f'Using {device} device')

    bgen = SimpleBlockGenerator()
    model = SequentialStructureConvNet(cfg.num_input_ch,
                                       bgen,
                                       nfeat=cfg.num_feat,
                                       nblocks=cfg.num_blocks).to(device)

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    #--------------------------------------------------------------------------------------

    # create a tensorboard writer
    writer = SummaryWriter(log_dir=os.getcwd())

    #--------------------------------------------------------------------------------------
    ssim = StructuralSimilarityIndexMeasure()
    psnr = PeakSignalNoiseRatio()

    for i_epoch in range(cfg.num_epochs):
        log.info(f'epoch {i_epoch}')

        # training loop
        for i_batch, batch in enumerate(training_loader_patches):
            log.info(f'training batch {i_batch}')
            if cfg.num_input_ch == 1:
                x = batch['pet_low'][tio.DATA].to(device)
            else:
                x0 = batch['pet_low'][tio.DATA]
                x1 = batch['mr'][tio.DATA]
                x = torch.cat((x0, x1), 1).to(device)

            y = batch['pet_high'][tio.DATA].to(device)

            # Compute prediction and loss
            pred = model(x)
            loss = loss_fn(pred, y)
            train_loss = loss.item()
            log.info(f'loss {train_loss}')

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #-----------------------------------------------------------
        val_loss = 0.
        val_ssim = 0.
        val_psnr = 0.
        num_val_batches = len(validation_loader)

        with torch.no_grad():
            for i_batch, batch in enumerate(validation_loader):
                log.info(f'validation batch {i_batch}')
                if cfg.num_input_ch == 1:
                    x = batch['pet_low'][tio.DATA].to(device)
                else:
                    x0 = batch['pet_low'][tio.DATA]
                    x1 = batch['mr'][tio.DATA]
                    x = torch.cat((x0, x1), 1).to(device)

                # Compute prediction and loss
                y = batch['pet_high'][tio.DATA].to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

                if ((i_epoch + 1) % cfg.val_metric_period) == 0:
                    val_psnr = psnr(pred, y)
                    val_ssim = ssim(pred, y)

        val_loss /= num_val_batches
        val_ssim /= num_val_batches
        val_psnr /= num_val_batches

        log.info(f'validation loss {val_loss}')

        writer.add_scalars('losses', {
            'train': train_loss,
            'validation': val_loss
        }, i_epoch)

        if ((i_epoch + 1) % cfg.val_metric_period) == 0:
            log.info(f'validation ssim {val_ssim}')
            log.info(f'validation psnr {val_psnr}')
            writer.add_scalars('val_metrics', {
                'psnr': val_psnr,
                'ssim': val_ssim
            }, i_epoch)
    writer.close()


if __name__ == '__main__':
    main()