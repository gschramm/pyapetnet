import numpy as np
from scipy.ndimage import find_objects, gaussian_filter
from scipy.optimize import minimize

from pymirc.image_operations import aff_transform, zoom3d, rigid_registration


def preprocess_volumes(pet_vol,
                       mr_vol,
                       pet_affine,
                       mr_affine,
                       training_voxsize,
                       perc=99.99,
                       coreg=True,
                       crop_mr=True,
                       mr_ps_fwhm_mm=None,
                       verbose=False):

    # make a copy of the mr_affine since we will modify it and return
    m_aff = mr_affine.copy()

    # get voxel sizes from affine matrices
    pet_voxsize = np.linalg.norm(pet_affine[:-1, :-1], axis=0)
    mr_voxsize = np.linalg.norm(m_aff[:-1, :-1], axis=0)

    # crop the MR if needed
    if crop_mr:
        bbox = find_objects(mr_vol > 0.1 * mr_vol.max(), max_label=1)[0]
        mr_vol = mr_vol[bbox]
        crop_origin = np.array([x.start for x in bbox] + [1])
        m_aff[:-1, -1] = (m_aff @ crop_origin)[:-1]

    # post-smooth MR if needed
    if mr_ps_fwhm_mm is not None:
        print(f'post-smoothing MR with {mr_ps_fwhm_mm} mm')
        mr_vol = gaussian_filter(mr_vol, mr_ps_fwhm_mm / (2.35 * mr_voxsize))

    # regis_aff is the affine transformation that maps from the PET to the MR grid
    # if coreg is False, it is simply deduced from the affine transformation
    # otherwise, rigid registration with mutual information is used
    if coreg:
        _, regis_aff, _ = rigid_registration(pet_vol, mr_vol, pet_affine,
                                             m_aff)
    else:
        regis_aff = np.linalg.inv(pet_affine) @ m_aff

    # interpolate both volumes to the voxel size used during training
    zoomfacs = mr_voxsize / training_voxsize
    if not np.all(np.isclose(zoomfacs, np.ones(3))):
        if verbose:
            print('interpolationg input volumes to training voxel size')
        mr_vol_interpolated = zoom3d(mr_vol, zoomfacs)
        m_aff = m_aff @ np.diag(np.concatenate((1. / zoomfacs, [1])))
    else:
        mr_vol_interpolated = mr_vol.copy()

    # this is the final affine that maps from the PET grid to interpolated MR grid
    # using the small voxels used during training
    pet_mr_interp_aff = regis_aff @ np.diag(
        np.concatenate((1. / zoomfacs, [1])))

    if not np.all(np.isclose(pet_mr_interp_aff, np.eye(4))):
        pet_vol_mr_grid_interpolated = aff_transform(pet_vol,
                                                     pet_mr_interp_aff,
                                                     mr_vol_interpolated.shape,
                                                     cval=pet_vol.min())
    else:
        pet_vol_mr_grid_interpolated = pet_vol.copy()

    # convert the input volumes to float32
    if not mr_vol_interpolated.dtype == np.float32:
        mr_vol_interpolated = mr_vol_interpolated.astype(np.float32)
    if not pet_vol_mr_grid_interpolated.dtype == np.float32:
        pet_vol_mr_grid_interpolated = pet_vol_mr_grid_interpolated.astype(
            np.float32)

    # normalize the data: we divide the images by the specified percentile (more stable than the max)
    if verbose: print('\nnormalizing the input images')
    mr_max = np.percentile(mr_vol_interpolated, perc)
    mr_vol_interpolated /= mr_max

    pet_max = np.percentile(pet_vol_mr_grid_interpolated, perc)
    pet_vol_mr_grid_interpolated /= pet_max

    return pet_vol_mr_grid_interpolated, mr_vol_interpolated, m_aff, pet_max, mr_max
