import nibabel as nib
import numpy   as np
import os

from glob import glob

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage         import affine_transform

def brainweb(brainweb_raw_dir = os.path.join('..','data','training_data','brainweb','raw'),
             subject          = 'subject54',
             gm_contrast      = 4,
             wm_contrast      = 1,
             csf_contrast     = 0.05,
             skin_contrast    = 0.5,
             fat_contrast     = 0.25,
             bone_contrast    = 0.1):

  dmodel_path = os.path.join(brainweb_raw_dir, subject + '_crisp_v.mnc.gz')
  gm_path     = os.path.join(brainweb_raw_dir, subject + '_gm_v.mnc.gz')
  wm_path     = os.path.join(brainweb_raw_dir, subject + '_wm_v.mnc.gz')
  t1_path     = os.path.join(brainweb_raw_dir, subject + '_t1w_p4.mnc.gz')
  
  # the simulated t1 has different voxel size and FOV)
  dmodel_affine = nib.load(dmodel_path).affine.copy()
  t1_affine     = nib.load(t1_path).affine.copy()
  
  dmodel_voxsize = np.sqrt((dmodel_affine**2).sum(0))[:-1]
  t1_voxsize     = np.sqrt((t1_affine**2).sum(0))[:-1]
  
  dmodel = nib.load(dmodel_path).get_data()
  gm     = nib.load(gm_path).get_data()
  wm     = nib.load(wm_path).get_data()
  
  t1     = nib.load(t1_path).get_data()
  
  pet_gt = gm_contrast*gm + wm_contrast*wm + skin_contrast*(dmodel == 5) + skin_contrast*(dmodel == 6) + fat_contrast*(dmodel == 4) + bone_contrast*(dmodel == 7) + bone_contrast*(dmodel == 11) + csf_contrast*(dmodel == 1)

  pet_gt_regrid = affine_transform(pet_gt, np.linalg.inv(dmodel_affine) @ t1_affine, 
                                    order = 1, prefilter = False, output_shape = t1.shape)

  return np.array([4*np.flip(t1,1)/t1.max(),np.flip(pet_gt_regrid,1)])

