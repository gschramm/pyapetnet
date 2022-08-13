import numpy as np
import os
import numpy as np
import sys
import nibabel as nb
import warnings

from scipy.ndimage import find_objects
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift, rotate
from copy import deepcopy

import tensorflow
from tensorflow.keras.utils import Sequence

from tempfile import NamedTemporaryFile

from pymirc.image_operations import aff_transform, zoom3d
from .utils import affine_center_rotation


def vol_brain_crop(input_vols,
                   target_vol,
                   bbox_vol_ch=1,
                   brain_is_ch=0,
                   brain_is_th=0.35):
    """
    function to crop FOV of list of PET/MR input volumes to brain

    Inputs
    ------

    input_vols  ... list of input volumes

    target_vols ... the target volume

    Keyword arguments
    -----------------

    bbox_vol_ch ... (int) input channel from which to calculate the bounding box
                    default 1
  
    brain_is_ch ... (int) input channel from which to compute the fead head extension of the brain
                    default 0
 
    brain_is_th ... (float) threshold used to calculate fead head extension of brain

  
    Returns
    -------

    a tuple containing the cropped input volumes and the target volume
    """

    n_channels = len(input_vols)

    # by default we use the complete volume
    bbox = [
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None)
    ]

    if not bbox_vol_ch is None:
        bbox = find_objects(
            input_vols[bbox_vol_ch] > 0.1 * input_vols[bbox_vol_ch].max(),
            max_label=1)[0]

    for ch in range(n_channels):
        input_vols[ch] = input_vols[ch][bbox]
    if target_vol is not None: target_vol = target_vol[bbox]

    # clip the FOV in IS direction
    if not brain_is_ch is None:
        prof = input_vols[brain_is_ch].sum(axis=(0, 1)).squeeze()
        tmp = np.argwhere(prof > brain_is_th * prof.max()).flatten()

        start = max(0, tmp[0] - 20)
        stop = max(len(tmp), tmp[-1] + 20)

        for ch in range(n_channels):
            input_vols[ch] = input_vols[ch][..., start:stop, :]
        if target_vol is not None: target_vol = target_vol[..., start:stop, :]

    return input_vols, target_vol


#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------


def petmr_brain_data_augmentation(orig_vols,
                                  rand_contrast_ch=1,
                                  ps_ch=0,
                                  ps_fwhms=[0, 3., 4.],
                                  rand_misalign_ch=None,
                                  shift_amp=2,
                                  rot_amp=5):
    """
    function for data augmentation of input volumes


    Inputs
    ------

    orig_vols ... list of input volumes to be augmented / changed

    Keyword arguments
    -----------------

    rand_contrast_ch ... (int or None) channel where contrast is randomly flipped / quadratically changed
                         default: 1

    rand_ps_ch       ... (int or None) channel which is randomly post smooted 
                         default: 0

    ps_fwhms         ... (float) list of post smoothing fwhms (voxel units)
                         
    rand_misalign_ch ... (int or None) channel which is randomly mislaligned
                         default: None

    shift_amp        ... (float) maximal shift for misalignment in pixels - default: 2

    rot_amp          ... (float) maximal rotation angle for misalignment in degrees - default: 5


    Returns
    -------
 
    a list of augmented input volumes
    """

    n_ch = len(orig_vols)

    augmented_vols = []

    for ps_fwhm in ps_fwhms:
        vols = deepcopy(orig_vols)

        if ps_fwhm > 0:
            vols[ps_ch] = gaussian_filter(vols[ps_ch], ps_fwhm / 2.35)

        # randomly misalign one of the input channels
        if rand_misalign_ch is not None:
            costheta = 2 * np.random.rand(2) - 1
            sintheta = np.sqrt(1 - costheta**2)
            phi = 2 * np.pi * np.random.rand(2)
            rshift = shift_amp * np.random.rand()

            # random translation
            offset = np.array([
                rshift * np.cos(phi[0]) * sintheta[0],
                rshift * np.sin(phi[0]) * sintheta[0], rshift * costheta[0]
            ])

            # random rotation axis
            uv = np.array([
                np.cos(phi[1]) * sintheta[1],
                np.sin(phi[1]) * sintheta[1], costheta[1]
            ])
            rot_angle = rot_amp * np.pi * np.random.rand() / 180.

            bp_center = np.array(vols[rand_misalign_ch].shape[:-1]) / 2 - 0.5
            aff = affine_center_rotation(uv,
                                         rot_angle,
                                         uv_origin=bp_center,
                                         offset=offset)

            # transform the image
            vols[rand_misalign_ch][..., 0] = aff_transform(
                vols[rand_misalign_ch][..., 0],
                aff,
                cval=vols[rand_misalign_ch].min())

        # randomize the contrast of the second input channel
        if rand_contrast_ch is not None:
            r1 = 0.4 * np.random.random() + 0.8
            vols[rand_contrast_ch] = vols[rand_contrast_ch]**r1

            # randomly invert contrast
            if np.random.random() >= 0.5:
                vols[rand_contrast_ch] = vols[rand_contrast_ch].max(
                ) - vols[rand_contrast_ch]

        augmented_vols.append(vols)

    return augmented_vols


#-----------------------------------------------------------------------------------------------------------------


class PatchSequence(Sequence):
    """
    class to generate mini batches of patches of multiple input volumes and an optional
    target volume.
    the class is derived from keras.Sequence
    """
    def __init__(self,
                 input_fnames,
                 target_fnames=None,
                 preload_data=True,
                 batch_size=5,
                 patch_size=(33, 33, 33),
                 input_read_func=lambda x: nb.load(x),
                 get_data_func=lambda x: x.get_data(),
                 get_affine_func=lambda x: x.affine,
                 preproc_func=vol_brain_crop,
                 preproc_kwargs={},
                 input_voxsize=None,
                 internal_voxsize=np.array([1., 1., 1.]),
                 normalize=True,
                 norm_channel=None,
                 target_norm_channel=0,
                 intercept_func=lambda x: x.min(),
                 slope_func=lambda x: (np.percentile(x, 99.99) - x.min()),
                 order=None,
                 target_order=1,
                 random_flip=False,
                 concat_mode=False,
                 data_aug_func=None,
                 data_aug_kwargs={}):
        """
        Inputs
        ------
    
        input_fnames    ... (list of lists) containing the model input file names using the following structure:
                             [
                              [input_channel_1_subject1, input_channel_2_subject1, ...,, input_channel_n_subject1]
                              [input_channel_1_subject2, input_channel_2_subject2, ...,, input_channel_n_subject2]
                              ...
                             ]

  
        Keyword arguments
        -----------------
  
        target_fnames       ... (list of lists) containing the model target file names using the following structure:
                                 [target_subject1, target_subject2, ..., target_subjectn] - default: None

        batch_size          ... (int) size of mini batch - default: 5
 
        preload_data        ... (bool) whether to keep all input volumes in memory  - default true
                                       if false, the preprocessed input volumes are written as .npy to
                                       a tmp directory

        patch_size          ... (int,int,int) size of random patch 

        input_read_func     ... (function) used to open the input data file - default nb.load()

        get_data_func       ... (function) to get image volume from data object - default: lambda x: x.get_data()

        get_affine_func     ... (function) used to get the affine transformation from data object
                                this is used to get the input voxel size if not give - default: lambda: x.affine

        preproc_func        ... (function) to preprocess the input volumes - default: vol_brain_crop

        preproc_kwargs      ... (dictionary) passed as kwargs to preproc_func - default: {}

        input_voxsize       ... (np.array) voxel size of input volumes - if None it is calculated from
                                the affine of the input data - default: None mean that it is retrieved from
                                the read affine matrix
  
        internal_voxsize    ... (np.array) specifying the internal voxel size to which the input
                                images are interpolated to - default: np.array([1.,1.,1.])

        normalize           ... (bool) whether to normalize the intesity of the input and target volumes
                                default: True

        norm_channel        ... (list or None) specifying which slope and intercept should be used for the
                                normalization of the input channels - default: None means that slope and intercept
                                are taken from each input channel itself

        target_norm_channel ... (int) specifying the input channel from which the slope and intercept for the
                                normalization of the target are taken - default: 0

        intercept_func,     ... (function) used to calculate the intercepts and slopes for the normalization
        slope_func              default: lambda x: 0.5*(np.percentile(x,99.9) + x.min()) and
                                         lambda x: 0.5*(np.percentile(x,99.9) - x.min())
                                which maps range approx to [-1,1]

        order               ... (list of ints) order of interpolation used when volumes are interpolated
                                to internal voxel size - default None means 1 for all volumes

        target_order        ... (int) order of interpolation used when target volume is interpolated
                                default 1

        concat_mode         ... (bool) if True than the output input batch is concatenated to a 
                                "single channel" axis along axis 1. This is need in case only
                                single channel input can be handled. default is False.

        data_aug_func       ... (function) to that is called at the end of each Keras epoch 
                                          this can be e.g. to augment the data
                                          default: petmr_brain_data_augmentation

        data_aug_kwargs     ... (dictionary) passed as kwargs to preproc_func - default: {}

        random_flip         ... (bool) randomly flip (reverse) axis when drawing patching - default True
 
        verbose             ... print verbose output
        """

        self.input_fnames = input_fnames
        self.target_fnames = target_fnames
        self.n_data_sets = len(self.input_fnames)
        self.preload_data = preload_data
        self.batch_size = batch_size
        self.n_input_channels = len(input_fnames[0])
        self.patch_size = patch_size
        self.concat_mode = concat_mode

        self.input_read_func = input_read_func
        self.get_data_func = get_data_func
        self.get_affine_func = get_affine_func

        self.input_voxsize = input_voxsize
        self.internal_voxsize = internal_voxsize

        if norm_channel is None:
            self.norm_channel = np.arange(self.n_input_channels)
        else:
            self.norm_channel = norm_channel

        self.target_norm_channel = target_norm_channel

        self.intercept_func = intercept_func
        self.slope_func = slope_func

        self.patch_size = patch_size

        if order is None:
            self.order = np.ones(self.n_input_channels, dtype=np.int)
        else:
            self.order = order

        self.target_order = target_order

        self.data_aug_func = data_aug_func
        self.data_aug_kwargs = data_aug_kwargs

        self.random_flip = random_flip

        self.input_vols = self.n_data_sets * [None]
        self.input_vols_augmented = self.n_data_sets * [None]
        self.target_vols = self.n_data_sets * [None]
        self.isub = 0

        self.slopes = []
        self.intercepts = []

        #--- load and preprocess data sets
        for i in range(self.n_data_sets):
            # (1) read one data set into memory
            tmp = [
                self.input_read_func(fname) for fname in self.input_fnames[i]
            ]
            input_vols = [
                np.expand_dims(self.get_data_func(d), -1) for d in tmp
            ]

            if self.target_fnames is not None:
                tmp = self.input_read_func(self.target_fnames[i])
                target_vol = np.expand_dims(self.get_data_func(tmp), -1)

            # (2) interpolate the volume to target voxel size
            if self.input_voxsize is None:
                in_data = self.input_read_func(self.input_fnames[i][0])
                affine = self.get_affine_func(in_data)
                input_voxsize = np.sqrt((affine**2).sum(axis=0))[:-1]
                zoomfacs = input_voxsize / self.internal_voxsize
            else:
                zoomfacs = self.input_voxsize / self.internal_voxsize

            if not np.all(zoomfacs == 1):
                for ch in range(self.n_input_channels):
                    input_vols[ch] = np.expand_dims(
                        zoom3d(input_vols[ch][..., 0], zoomfacs), -1)
                if self.target_fnames is not None:
                    target_vol = np.expand_dims(
                        zoom3d(target_vol[..., 0], zoomfacs), -1)

            # (3) apply the preprocessing function
            if preproc_func is not None:
                input_vols, target_vol = preproc_func(input_vols, target_vol,
                                                      **preproc_kwargs)

            # (4) normalize data
            intercepts = [self.intercept_func(vol) for vol in input_vols]
            slopes = [self.slope_func(vol) for vol in input_vols]

            self.intercepts.append(intercepts)
            self.slopes.append(slopes)

            for ch in range(self.n_input_channels):
                if self.norm_channel[ch] is not None:
                    input_vols[ch] -= intercepts[self.norm_channel[ch]]
                    input_vols[ch] /= slopes[self.norm_channel[ch]]
            if self.target_fnames is not None:
                if self.target_norm_channel is not None:
                    target_vol -= intercepts[self.target_norm_channel]
                    target_vol /= slopes[self.target_norm_channel]

            # (5) augment data
            if self.data_aug_func is not None:
                input_vols_augmented = self.data_aug_func(
                    input_vols, **self.data_aug_kwargs)
            else:
                input_vols_augmented = None

            #--------------------------------------------------------------
            #--------------------------------------------------------------
            #--------------------------------------------------------------
            # (6) append data or write the preprocessed data to a tmp dir
            if self.preload_data:
                self.input_vols[i] = input_vols
                self.input_vols_augmented[i] = input_vols_augmented
                self.target_vols[i] = target_vol
            else:
                if 'VSC_SCRATCH' in os.environ:
                    tmp_dir = os.environ['VSC_SCRATCH']
                else:
                    tmp_dir = None

                # write the input vols to disk
                tmp_names = []
                for iv in input_vols:
                    # on the VSC we should not write files in /tmp but in $VSC_SCRATCH
                    tmp = NamedTemporaryFile(dir=tmp_dir,
                                             suffix='.npy',
                                             delete=False)
                    np.save(tmp.name, iv)
                    tmp_names.append(tmp.name)
                self.input_vols[i] = tmp_names

                # write the augmented vols to disk
                if self.data_aug_func is not None:
                    aug_names = []
                    for iv in input_vols_augmented:
                        tmp_names = []
                        for a_ch in range(len(iv)):
                            # on the VSC we should not write files in /tmp but in $VSC_SCRATCH
                            tmp = NamedTemporaryFile(dir=tmp_dir,
                                                     suffix='.npy',
                                                     delete=False)
                            np.save(tmp.name, iv[a_ch])
                            tmp_names.append(tmp.name)
                        aug_names.append(tmp_names)
                    self.input_vols_augmented[i] = aug_names

                # write the target vols to disk
                tmp = NamedTemporaryFile(dir=tmp_dir,
                                         suffix='.npy',
                                         delete=False)
                np.save(tmp.name, target_vol)
                self.target_vols[i] = tmp.name

    #------------------------------------------------------------------
    def __del__(self):
        # clean up temporary files
        if not self.preload_data:
            # remove input vols
            for fl in self.input_vols:
                for f in fl:
                    os.remove(f)
            # remove augmented input vols
            if self.data_aug_func is not None:
                for fl in self.input_vols_augmented:
                    for ff in fl:
                        for f in ff:
                            os.remove(f)

            # remove target vols
            for f in self.target_vols:
                os.remove(f)

    #------------------------------------------------------------------
    def __len__(self):
        # not sure why this is needed for a Keras Sequence
        # for random patch sampling it does not make sense
        return 20 * self.batch_size

    #------------------------------------------------------------------
    def __getitem__(self, idx, verbose=False):

        if verbose:
            print('generating batch: ', idx)

        input_batch = [
            np.zeros((self.batch_size, ) + self.patch_size + (1, ))
            for i in range(self.n_input_channels)
        ]

        if self.target_fnames is not None:
            target_batch = np.zeros((self.batch_size, ) + self.patch_size +
                                    (1, ))
        else:
            target_batch = None

        for i in range(self.batch_size):
            # choose a random input data set
            self.isub = np.random.randint(len(self.input_vols))

            if self.preload_data:
                volshape = self.input_vols[self.isub][0].shape
            else:
                volshape = np.load(self.input_vols[self.isub][0]).shape

            ii0 = np.random.randint(0, volshape[0] - self.patch_size[0])
            ii1 = np.random.randint(0, volshape[1] - self.patch_size[1])
            ii2 = np.random.randint(0, volshape[2] - self.patch_size[2])

            patch_slice = (slice(ii0, ii0 + self.patch_size[0], None),
                           slice(ii1, ii1 + self.patch_size[1], None),
                           slice(ii2, ii2 + self.patch_size[2],
                                 None), slice(None, None, None))

            # draw random number of random flips
            if self.random_flip:
                flip_ax = np.random.randint(0, 4)
            else:
                flip_ax = 3

            # draw random variable whether to use augmented data
            if not None in self.input_vols_augmented:
                use_aug = True
                aug_ch = np.random.randint(len(self.input_vols_augmented[0]))
                if verbose:
                    print(self.isub, patch_slice, aug_ch)
            else:
                use_aug = False
                if verbose:
                    print(self.isub, patch_slice)

            for ch in range(self.n_input_channels):
                if use_aug:
                    if self.preload_data:
                        patch = self.input_vols_augmented[
                            self.isub][aug_ch][ch][patch_slice]
                    else:
                        patch = np.load(self.input_vols_augmented[self.isub]
                                        [aug_ch][ch])[patch_slice]
                else:
                    if self.preload_data:
                        patch = self.input_vols[self.isub][ch][patch_slice]
                    else:
                        patch = np.load(
                            self.input_vols[self.isub][ch])[patch_slice]

                if flip_ax < 3:
                    input_batch[ch][i, ...] = np.flip(patch, flip_ax)
                else:
                    input_batch[ch][i, ...] = patch

            if self.target_fnames is not None:
                if self.preload_data:
                    tpatch = self.target_vols[self.isub][patch_slice]
                else:
                    tpatch = np.load(self.target_vols[self.isub])[patch_slice]
                if flip_ax < 3:
                    target_batch[i, ...] = np.flip(tpatch, flip_ax)
                else:
                    target_batch[i, ...] = tpatch

        if self.concat_mode:
            input_batch = np.concatenate(input_batch, axis=1)

        return (input_batch, target_batch)

    #------------------------------------------------------------------
    def get_input_vols_center_crop(self, crop_shape, offset):
        """ get a center crop with shape crop_shape and offset from the input volumes """

        input_batch = [
            np.zeros((self.n_data_sets, ) + crop_shape)
            for x in range(self.n_input_channels)
        ]
        target_batch = np.zeros((self.n_data_sets, ) + crop_shape)

        for i in range(self.n_data_sets):
            start = (np.array(self.input_vols[i][0].shape) //
                     2) - (np.array(crop_shape) // 2) + np.array(offset)
            end = start + np.array(crop_shape)

            sl = [slice(start[x], end[x]) for x in range(start.shape[0])]

            for j in range(self.n_input_channels):
                input_batch[j][i, ...] = self.input_vols[i][j][tuple(sl)]

            target_batch[i, ...] = self.target_vols[i][tuple(sl)]

        if self.concat_mode:
            input_batch = np.concatenate(input_batch, axis=1)

        return (input_batch, target_batch)

    #------------------------------------------------------------------
    #def on_epoch_end(self):
    #  print('epoch end')
