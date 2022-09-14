import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, shift

nii = nib.load('brainweb_06_osem.nii')
vol = nii.get_fdata().astype(np.float32)

vol_misaligned = shift(rotate(vol, 8.4, reshape = False), (5.5,-4.5,7.4))

nib.save(nib.Nifti1Image(vol_misaligned, nii.affine), 'misaligned_brainweb_06_osem.nii')
