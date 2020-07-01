import numpy as np
import matplotlib.pyplot as py
import os

import tensorflow
if tensorflow.__version__ >= '2':
  from tensorflow.keras.models import load_model
else:
  from keras.models import load_model

from scipy.ndimage import gaussian_filter

n = 128
img = np.zeros((n,n,n), dtype = np.float32)

x = np.arange(n) - n/2 + 0.5
x0, x1, x2 = np.meshgrid(x,x,x, indexing = 'ij')

# set background
r = np.sqrt((x0 - 0)**2 + (x1-0)**2 + (x2-0)**2)
img[r < 40] = 0.5

# set hot sphere
r1 = np.sqrt((x0 - 30)**2 + (x1-0)**2 + (x2-0)**2)
img[r1 < 3] = 1.

r2 = np.sqrt((x0 - 20)**2 + (x1-0)**2 + (x2-0)**2)
img[r2 < 2] = 1.

r3 = np.sqrt((x0 - 5)**2 + (x1-0)**2 + (x2-0)**2)
img[r3 < 5] = 1.

# set cold sphere
r4 = np.sqrt((x0 + 20)**2 + (x1-0)**2 + (x2-0)**2)
img[r4 < 15] = 0

# simulated PET and MR
pet = gaussian_filter(img + 0*np.random.randn(n,n,n), 4.5 / 2.35)
mr  = img.max() - img

model_names = ['190528_paper_bet_10_psf_mlem.h5', 
               '190904_osem_nopsf_bet_10_bs_52_ps_29.h5',
               '190904_osem_psf_bet_10_bs_52_ps_29_psf_4_5.h5']


pred_bows = []

for model_name in model_names:
  # load the model
  model = load_model(os.path.join('../../data/trained_models', model_name))
  
  # make the prediction
  x = [np.expand_dims(np.expand_dims(pet,0),-1), np.expand_dims(np.expand_dims(mr,0),-1)]
  pred_bow = model.predict(x).squeeze()
  pred_bows.append(pred_bow)  

fig, ax = py.subplots(1,len(model_names), figsize = (len(model_names)*4,4), sharey = True)
for i, pred_bow in enumerate(pred_bows):

  ax[i].plot(img[:,n//2,n//2], label = 'gt')
  ax[i].plot(pet[:,n//2,n//2], label = 'osem')
  ax[i].plot(pred_bow[:,n//2,n//2], label = 'pred')
  ax[i].set_title(model_names[i], fontsize = 'small')
  ax[i].grid(ls = ':')

ax[0].legend()
fig.tight_layout()
fig.show()

import pymirc.viewer as pv
ims = {'vmin': 0, 'vmax':1.2}
pv.ThreeAxisViewer([img,pet,mr,], 
                   imshow_kwargs = ims, ls = '', sl_x = 68, sl_y = 63, sl_z = 64)
pv.ThreeAxisViewer(pred_bows, imshow_kwargs = ims, ls = '', sl_x = 68, sl_y = 63, sl_z = 64)
