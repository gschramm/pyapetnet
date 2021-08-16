import nibabel as nib
import numpy as np

def read_nifty(path):
  nii = nib.as_closest_canonical(nib.load(path))
  vol = np.expand_dims(nii.get_fdata(),0).astype(np.float32)

  return vol, nii.affine
