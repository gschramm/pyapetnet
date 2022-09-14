import numpy as np
from scipy.ndimage import find_objects, gaussian_filter

import SimpleITK as sitk

from .registration import align_and_resample


def preprocess_volumes(
    pet_vol: np.ndarray,
    mr_vol: np.ndarray,
    pet_affine: np.ndarray,
    mr_affine: np.ndarray,
    training_voxsize: tuple[float, float, float],
    perc: float = 99.99,
    coreg: bool = True,
    crop_mr: bool = True,
    mr_ps_fwhm_mm: float | None = None,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, sitk.Transform]:
    """Preporcess (resample, align and intensity normalize) volumes for pyapetnet

    Parameters
    ----------
    pet_vol : np.ndarray
        the PET volume
    mr_vol : np.ndarray
        the MR volume
    pet_affine : np.ndarray
        the PET affine matrix
    mr_affine : np.ndarray
        the MR affine matrix
    training_voxsize : tuple[float, float, float]
        the internal voxel size used during training
    perc : float, optional
        precentile used for intensity normalization, by default 99.99
    coreg : bool, optional
        whether to coregister or simply resample to volumes, by default True
    crop_mr : bool, optional
        whether to crop the MR to the bounding box of the head, by default True
    mr_ps_fwhm_mm : float | None, optional
        smoothing FWHM (mm) for MR, by default None
    verbose : bool, optional
        print verbose output, by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float, float, sitk.Transform]:
    - the preprocessed PET volume
    - the preprocessed MR volume
    - the affine matrix of the proprocessed MR (cropped and resampled)
    - values used for PET intensity normalization
    - values used for MR intensity normalization
    - the final transform that maps the moving PET image to the resampled fixed MR image 
    """
    mr_voxsize = np.linalg.norm(mr_affine[:-1, :-1], axis=0)

    # crop the MR if needed
    if crop_mr:
        bbox = find_objects(mr_vol > 0.1 * mr_vol.max(), max_label=1)[0]
        mr_vol = mr_vol[bbox]
        crop_origin = np.array([x.start for x in bbox] + [1])
        mr_affine = mr_affine.copy()
        mr_affine[:-1, -1] = (mr_affine @ crop_origin)[:-1]

    # post-smooth MR if needed
    if mr_ps_fwhm_mm is not None:
        print(f'post-smoothing MR with {mr_ps_fwhm_mm} mm')
        mr_vol = gaussian_filter(mr_vol, mr_ps_fwhm_mm / (2.35 * mr_voxsize))

    mr_vol_interpolated, pet_vol_mr_grid_interpolated, transform, m_aff = align_and_resample(
        mr_vol,
        pet_vol,
        mr_affine,
        pet_affine,
        new_spacing=training_voxsize,
        resample_only=(not coreg),
        verbose=verbose)

    # convert the input volumes to float32
    if not mr_vol_interpolated.dtype == np.float32:
        mr_vol_interpolated = mr_vol_interpolated.astype(np.float32)
    if not pet_vol_mr_grid_interpolated.dtype == np.float32:
        pet_vol_mr_grid_interpolated = pet_vol_mr_grid_interpolated.astype(
            np.float32)

    # normalize the data: we divide the images by the specified percentile (more stable than the max)
    if verbose: print('\nnormalizing the input images')
    mr_max = float(np.percentile(mr_vol_interpolated, perc))
    mr_vol_interpolated /= mr_max

    pet_max = float(np.percentile(pet_vol_mr_grid_interpolated, perc))
    pet_vol_mr_grid_interpolated /= pet_max

    return pet_vol_mr_grid_interpolated, mr_vol_interpolated, m_aff, pet_max, mr_max, transform
