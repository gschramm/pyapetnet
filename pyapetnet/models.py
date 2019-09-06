import os

import keras
from keras.models     import Sequential, Model
from keras.layers     import Input, Conv3D, PReLU, BatchNormalization, Add, Concatenate
from keras.utils      import plot_model

import numpy as np

import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matplotlib.pyplot as py
import matplotlib.image  as mpimg

from tempfile import NamedTemporaryFile

#----------------------------------------------------------------------------------------

def apetnet(n_ch               = 2, 
            n_ind_layers       = 1, 
            n_common_layers    = 7, 
            n_kernels_ind      = 15, 
            n_kernels_common   = 30, 
            kernel_shape       = (3,3,3), 
            res_channels       = [0],
            disp               = False):
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
     
  disp             ... (bool) show the model
  """
  
  inputs = [Input(shape = (None, None, None,1), name = 'input_' + str(x)) for x in range(n_ch)]

  # individual paths
  if n_ind_layers > 0: 
    init_val_ind = keras.initializers.RandomNormal(mean = 0.0, 
                                                   stddev = np.sqrt(2/(np.prod(kernel_shape)*n_kernels_ind)))

    x1_list = [i for i in inputs] 

    for i in range(n_ind_layers): 
      for j in range(n_ch):
        x1_list[j] = Conv3D(n_kernels_ind, kernel_shape, padding = 'same', kernel_initializer = init_val_ind,
                            name = 'conv3d_ind_' + str(i) + '_' + str(j))(x1_list[j])
        x1_list[j] = PReLU(shared_axes=[1,2,3], name = 'prelu_ind_' + str(i) + '_' + str(j))(x1_list[j])
    # concatenate inputs
    x1 = Concatenate(name = 'concat_0')(x1_list)

  else:
    # concatenate inputs
    x1 = Concatenate(name = 'concat_0')(inputs)

  # common path
  init_val = keras.initializers.RandomNormal(mean = 0.0, 
                                             stddev = np.sqrt(2/(np.prod(kernel_shape)*n_kernels_common)))

  for i in range(n_common_layers): 
    x1 = Conv3D(n_kernels_common, kernel_shape, padding = 'same', kernel_initializer = init_val,
                name = 'conv3d_' + str(i))(x1)
    x1 = PReLU(shared_axes=[1,2,3], name = 'prelu_' + str(i))(x1)
  
  # layers that adds all features
  x1 = Conv3D(1, (1,1,1), padding='same', name = 'conv_111', 
              kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = np.sqrt(2)))(x1)
  
  if res_channels is not None:
    x1 = Add(name = 'add_0')([x1] + [inputs[i] for i in res_channels])
  
  model  = Model(inputs = inputs, outputs = x1)

  if disp:
    tmp_file = NamedTemporaryFile(prefix = 'model', suffix = '.png')
    plot_model(model, to_file= tmp_file.name)
    img = mpimg.imread(tmp_file)
    fig, ax = py.subplots()
    ax = py.imshow(img)
    py.draw()

  return model

#------------------------------------------------------------------------------------------
