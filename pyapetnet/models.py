import os

import tensorflow
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv3D, ReLU, PReLU, BatchNormalization, Add, Concatenate, Cropping3D
from tensorflow.keras.utils import plot_model

import numpy as np

import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matplotlib.pyplot as py
import matplotlib.image as mpimg

from tempfile import NamedTemporaryFile

#----------------------------------------------------------------------------------------


def apetnet(n_ch=2,
            n_ind_layers=1,
            n_common_layers=7,
            n_kernels_ind=15,
            n_kernels_common=30,
            kernel_shape=(3, 3, 3),
            res_channels=[0],
            add_final_relu=False,
            add_batchnorm=True,
            disp=False):
    """
    Create CNN model for multiple inputs and one voxel-wise prediction channel
 
    |----input_0     input_1    ...     input_n
    |      |            |       ...       |
    |  conv+prelu   conv+prelu  ...   conv+prelu
    |      |            |       ...       |
    |  conv+prelu   conv+prelu  ...   conv+prelu
    |      |            |       ...       |
    |      ---------concatenate ... -------
    |                   |                  
    |               conv+prelu                  
    |                   |                  
    |               conv+prelu                  
    |                   |                  
    |               conv+prelu                  
    |                   |                  
    |                   V                  
    ------------------>add
                        |
                      (relu)
                        |
                      output
 
    keyword arguments
    -----------------

    n_ch             ... (int) number of input channels

    n_ind_layers     ... (int) number of individual layers

    n_common_layers  ... (int) number of common layers

    n_kernels_ind    ... (int) number of kernels for individual layers

    n_kernels_common ... (int) number of kernels for common layers

    kernel_shape     ... (tuple) shape of kernels

    res_channels     ... (list) of channels to add to output of common layers
  
    add_batchnorm    ... (bool) add batch normalization layers between the conv and the 
                                PRELU layers

    add_final_relu   ... (bool) add a final ReLU layer before output to make sure that
                                output is non-negative

    disp             ... (bool) show the model
    """

    inputs = [
        Input(shape=(None, None, None, 1), name='input_' + str(x))
        for x in range(n_ch)
    ]

    # individual paths
    if n_ind_layers > 0:
        #init_val_ind = RandomNormal(mean = 0.0, stddev = np.sqrt(2/(np.prod(kernel_shape)*n_kernels_ind)))

        x1_list = [i for i in inputs]

        for i in range(n_ind_layers):
            for j in range(n_ch):
                x1_list[j] = Conv3D(n_kernels_ind,
                                    kernel_shape,
                                    padding='same',
                                    kernel_initializer='glorot_uniform',
                                    name='conv3d_ind_' + str(i) + '_' +
                                    str(j))(x1_list[j])
                if add_batchnorm:
                    x1_list[j] = BatchNormalization(name='batchnorm_ind_' +
                                                    str(i) + '_' + str(j))(
                                                        x1_list[j])
                x1_list[j] = PReLU(shared_axes=[1, 2, 3],
                                   name='prelu_ind_' + str(i) + '_' + str(j))(
                                       x1_list[j])
        # concatenate inputs
        x1 = Concatenate(name='concat_0')(x1_list)

    else:
        # concatenate inputs
        x1 = Concatenate(name='concat_0')(inputs)

    # common path
    #init_val = RandomNormal(mean = 0.0, stddev = np.sqrt(2/(np.prod(kernel_shape)*n_kernels_common)))

    for i in range(n_common_layers):
        x1 = Conv3D(n_kernels_common,
                    kernel_shape,
                    padding='same',
                    kernel_initializer='glorot_uniform',
                    name='conv3d_' + str(i))(x1)
        if add_batchnorm:
            x1 = BatchNormalization(name='batchnorm_' + str(i))(x1)
        x1 = PReLU(shared_axes=[1, 2, 3], name='prelu_' + str(i))(x1)

    # layers that adds all features
    x1 = Conv3D(1, (1, 1, 1),
                padding='same',
                name='conv_111',
                kernel_initializer=RandomNormal(mean=0.0,
                                                stddev=np.sqrt(2)))(x1)

    if res_channels is not None:
        x1 = Add(name='add_0')([x1] + [inputs[i] for i in res_channels])

    if add_final_relu:
        x1 = ReLU(name='final_relu')(x1)

    model = Model(inputs=inputs, outputs=x1)

    if disp:
        tmp_file = NamedTemporaryFile(prefix='model', suffix='.png')
        plot_model(model, to_file=tmp_file.name)
        img = mpimg.imread(tmp_file)
        fig, ax = py.subplots()
        ax = py.imshow(img)
        py.draw()

    return model


#------------------------------------------------------------------------------------------


def apetnet_vv5_onnx(input_tensor=None,
                     n_ind_layers=1,
                     n_common_layers=7,
                     n_kernels_ind=15,
                     n_kernels_common=30,
                     kernel_shape=(3, 3, 3),
                     add_final_relu=False,
                     debug=False):
    """ Stacked single channel version of apetnet
        
        For description of input parameters see apetnet

        The input_tensor argument is only used determine the input shape.
        If None the input shape us set to (32,16,16,1).
    """
    # define input (stacked PET and MRI image)
    if input_tensor is not None:
        ipt = Input(input_tensor.shape[1:5], name='input')
    else:
        ipt = Input(shape=(32, 16, 16, 1), name='input')

    # extract pet and mri image
    # - first image in order is pet
    ipt_dim_crop = int(ipt.shape[1] // 2)
    mri_image = Cropping3D(cropping=((ipt_dim_crop, 0), (0, 0), (0, 0)),
                           name='extract_mri')(ipt)
    pet_image = Cropping3D(cropping=((0, ipt_dim_crop), (0, 0), (0, 0)),
                           name='extract_pet')(ipt)

    # create the full model
    if not debug:
        # individual paths
        if n_ind_layers > 0:
            init_val_ind = RandomNormal(
                mean=0.0,
                stddev=np.sqrt(2 / (np.prod(kernel_shape) * n_kernels_ind)))

            pet_image_ind = pet_image
            mri_image_ind = mri_image

            for i in range(n_ind_layers):
                pet_image_ind = Conv3D(
                    n_kernels_ind,
                    kernel_shape,
                    padding='same',
                    name='conv3d_pet_ind_' + str(i),
                    kernel_initializer=init_val_ind)(pet_image_ind)
                pet_image_ind = PReLU(shared_axes=[1, 2, 3],
                                      name='prelu_pet_ind_' +
                                      str(i))(pet_image_ind)
                mri_image_ind = Conv3D(
                    n_kernels_ind,
                    kernel_shape,
                    padding='same',
                    name='conv3d_mri_ind_' + str(i),
                    kernel_initializer=init_val_ind)(mri_image_ind)
                mri_image_ind = PReLU(shared_axes=[1, 2, 3],
                                      name='prelu_mri_ind_' +
                                      str(i))(mri_image_ind)

            # concatenate inputs
            net = Concatenate(name='concat_0')([pet_image_ind, mri_image_ind])

        else:
            # concatenate inputs
            net = Concatenate(name='concat_0')([pet_image, mri_image])

        # common path
        init_val_common = RandomNormal(
            mean=0.0,
            stddev=np.sqrt(2 / (np.prod(kernel_shape) * n_kernels_common)))

        for i in range(n_common_layers):
            net = Conv3D(n_kernels_common,
                         kernel_shape,
                         padding='same',
                         name='conv3d_' + str(i),
                         kernel_initializer=init_val_common)(net)
            net = PReLU(shared_axes=[1, 2, 3], name='prelu_' + str(i))(net)

        # layers that adds all features
        net = Conv3D(1, (1, 1, 1),
                     padding='valid',
                     name='conv_final',
                     kernel_initializer=RandomNormal(mean=0.0,
                                                     stddev=np.sqrt(2)))(net)

        # add pet_image to prediction
        net = Add(name='add_0')([net, pet_image])

        # ensure that output is non-negative
        if add_final_relu:
            net = ReLU(name='final_relu')(net)

    # in debug mode only add up pet and mri image
    else:
        net = Concatenate(name='add_0')([pet_image, mri_image])

    # create model
    model = Model(inputs=ipt, outputs=net)

    # return the model
    return model
