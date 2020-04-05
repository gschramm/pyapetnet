# this is a demo script that produces simulated MLEM and T1 data
# sets from the brain web data base
#
# before running it you have to download the discrete model, grey matter,
# white matter and simulated T1 files from:
# http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html
# into
# '../data/training_data/brainweb/raw'

import nibabel as nib
import numpy   as np
import os

from glob import glob

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage         import affine_transform

from pyapetnet.threeaxisviewer import ThreeAxisViewer

def process_brainweb_subject(brainweb_raw_dir = os.path.join('..','data','training_data','brainweb','raw'),
                             subject          = 'subject54',
                             gm_contrast      = 4,
                             mlem_fwhm_mm     = 4.5):

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
  
  pet_gt = gm_contrast*gm + wm + 0.5*(dmodel == 5) + 0.5*(dmodel == 6) + 0.25*(dmodel == 4) + 0.1*(dmodel == 7) + 0.2*(dmodel == 11)
  
  mlem = gaussian_filter(pet_gt, mlem_fwhm_mm / (2.35*dmodel_voxsize))
  
  mlem_regrid = affine_transform(mlem, np.linalg.inv(dmodel_affine) @ t1_affine, 
                                 order = 1, prefilter = False, output_shape = t1.shape)
  
  pet_gt_regrid = affine_transform(pet_gt, np.linalg.inv(dmodel_affine) @ t1_affine, 
                                    order = 1, prefilter = False, output_shape = t1.shape)

  # return flipped data a new affine

  swap02_aff = np.array([[0., 0., 1., 0.],
                         [0., 1., 0., 0.],
                         [1., 0., 0., 0.],
                         [0., 0., 0., 1.]])
  
  flip1_aff      = np.eye(4, dtype = np.int)
  flip1_aff[1,1] = -1
  
  new_t1_aff = flip1_aff @ swap02_aff @ t1_affine
  
  return (np.flip(np.swapaxes(t1,0,2),1), np.flip(np.swapaxes(mlem_regrid,0,2),1), 
          np.flip(np.swapaxes(pet_gt_regrid,0,2),1), new_t1_aff)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  raw_dir = os.path.join('..','data','training_data','brainweb','raw')

  subjects = [os.path.basename(x).split('_')[0] for x in sorted(glob(os.path.join(raw_dir,'subject*_crisp_v.mnc.gz')))]

  for subject in subjects:
    sdir = os.path.join(os.path.dirname(raw_dir), subject)
    print(sdir)

    if not os.path.exists(sdir): os.mkdir(sdir)

    t1, mlem, pet_gt, aff = process_brainweb_subject(brainweb_raw_dir = raw_dir, subject = subject)

    nifti_img = nib.nifti2.Nifti2Image(mlem, aff)
    nifti_img.to_filename(os.path.join(sdir,'ch-000.nii.gz'))

    nifti_img = nib.nifti2.Nifti2Image(t1, aff)
    nifti_img.to_filename(os.path.join(sdir,'ch-001.nii.gz'))

    nifti_img = nib.nifti2.Nifti2Image(pet_gt, aff)
    nifti_img.to_filename(os.path.join(sdir,'target.nii.gz'))



