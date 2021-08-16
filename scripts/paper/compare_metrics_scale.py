# script to test influence of image degration on different metrics

import numpy as np
import nibabel as nib
import matplotlib.pyplot as py

import pymirc.viewer as pv

from scipy.ndimage import gaussian_filter
from skimage.measure import compare_ssim as ssim

img = nib.load('../../data/training_data/brainweb/subject04/pet_4000.nii').get_data()

# crop image
n = 20
img = img[(159-n):(159+n),(112-n):(112+n),(116-n):(116+n)]

#pv.ThreeAxisViewer(gt)

fwhms   = np.linspace(0.8,1.2,7)

sm_imgs = np.zeros((fwhms.shape[0],2*n,2*n,2*n)) 

sm_ssim = np.zeros(fwhms.shape[0])
sm_mse  = np.zeros(fwhms.shape[0])
sm_mae  = np.zeros(fwhms.shape[0])

g_sm_ssim = np.zeros((fwhms.shape[0],2*n,2*n,2*n)) 
g_sm_mse  = np.zeros((fwhms.shape[0],2*n,2*n,2*n)) 
g_sm_mae  = np.zeros((fwhms.shape[0],2*n,2*n,2*n)) 

for i, fwhm in enumerate(fwhms):
  deg_img        = fwhm*img
  sm_imgs[i,...] = deg_img

  sm_ssim[i], g_sm_ssim[i,...] = ssim(img, deg_img, 
                                      data_range = 4, gaussian_weights = True, 
                                      use_sample_covariance = False, gradient = True) 
  sm_mse[i] = ((img - deg_img)**2).mean()
  g_sm_mse[i,...]  = 2*(img - deg_img)
  
  sm_mae[i]  = np.abs(img - deg_img).mean()
  g_sm_mae[i,...]  = 2*(img > deg_img) - 1

fig, ax = py.subplots(5, fwhms.shape[0], figsize = (12,8))
for i, fwhm in enumerate(fwhms):
  ax[0,i].imshow(sm_imgs[i,:,:,n//2].T, vmin = 0, cmap = py.cm.Greys,
                 vmax =  np.abs(img[:,:,n//2]).max())
  ax[0,i].set_title(f'scale {round(fwhm,2)}')
  ax[1,i].imshow(sm_imgs[i,:,:,n//2].T, vmin = 0, cmap = py.cm.jet,
                 vmax =  np.abs(img[:,:,n//2]).max())
  ax[2,i].imshow(g_sm_ssim[i,:,:,n//2].T, cmap = py.cm.bwr,
                 vmin = -np.abs(g_sm_ssim[i,:,:,n//2]).max(), 
                 vmax =  np.abs(g_sm_ssim[i,:,:,n//2]).max())
  ax[3,i].imshow(g_sm_mse[i,:,:,n//2].T, cmap = py.cm.bwr,
                 vmin = -np.abs(g_sm_mse[i,:,:,n//2]).max(), 
                 vmax =  np.abs(g_sm_mse[i,:,:,n//2]).max())
  ax[4,i].imshow(g_sm_mae[i,:,:,n//2].T, cmap = py.cm.bwr,
                 vmin = -np.abs(g_sm_mae[i,:,:,n//2]).max(), 
                 vmax =  np.abs(g_sm_mae[i,:,:,n//2]).max())

ax[0,0].set_ylabel('deg. image')
ax[1,0].set_ylabel('deg. image (jet)')
ax[2,0].set_ylabel('norm. grad ssim')
ax[3,0].set_ylabel('norm. grad mse')
ax[4,0].set_ylabel('norm. grad mae')

for axx in ax.flatten():
  axx.set_xticks([])
  axx.set_yticks([])

fig.tight_layout()
fig.show()

#fig, ax = py.subplots()
#ax. plot(fwhms, 1 - sm_ssim, '.:', label = '1 - ssim')
#ax. plot(fwhms, sm_mse/sm_mse.max(), '.:', label = 'norm. mse')
#ax. plot(fwhms, sm_mae/sm_mae.max(), '.:', label = 'norm. mae')
#ax.set_xlabel('smoothing fwhm (mm)')
#ax.set_ylabel('metric')
#ax.legend()
#fig.tight_layout()
#fig.show()
