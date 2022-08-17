import nibabel as nib
import numpy as np
from pymirc.image_operations import zoom3d


def read_nifty(path, internal_voxsize=None):
    nii = nib.as_closest_canonical(nib.load(path))
    vol = nii.get_fdata()

    if np.issubdtype(vol.dtype, np.floating):
        vol = vol.astype(np.float32)

    aff = nii.affine

    if internal_voxsize is not None:
        zoomfacs = nii.header['pixdim'][1:4] / internal_voxsize
        vol = zoom3d(vol, zoomfacs).astype(np.float32)
        aff[:3, :3] = np.diag(1 / zoomfacs) @ aff[:3, :3]

    vol = np.expand_dims(vol, 0)

    return vol, nii.affine
