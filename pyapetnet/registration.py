import warnings
import numpy as np
import SimpleITK as sitk


def affine_to_direction(
        aff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """convert affine matrix to origin, spacing, direction

    Parameters
    ----------
    aff : np.ndarray
        affine matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        origin, spacing, direction
    """
    origin = aff[:-1, -1].copy()
    direction = aff[:-1, :-1].copy()
    spacing = np.linalg.norm(direction, axis=0)
    direction /= spacing

    return origin, spacing, direction.ravel()


def affine_from_sitk_image(image: sitk.Image) -> np.ndarray:
    """get the affine matrix from an sitk Image (header)

    Parameters
    ----------
    image : sitk.Image

    Returns
    -------
    np.ndarray
        affine matrix
    """
    direction = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    affine = np.eye(4)
    affine[:-1, :-1] = (direction * spacing)
    affine[:-1, -1] = origin

    return affine


def array_to_sitk_image(arr: np.ndarray, aff: np.ndarray) -> sitk.Image:
    """convert a 3D numpy array to an sitk Image

    Parameters
    ----------
    arr : np.ndarray
        3D array with axis order [x,y,z] or [LR,AP,FH]
    aff : np.ndarray
        affine matrix

    Returns
    -------
    sitk.Image

    Note
    ----
    By default sitk's GetImageFromArray assumes that the axis oder
    is [z,y,x] which is why we swap axes before calling it.
    """
    # sitk assumes that in numpy we have stores [z,y,x]
    # so we have to swap axes
    img = sitk.GetImageFromArray(np.swapaxes(arr, 0, 2))
    # remember to make a copy of affine, otherwise sITK operations will overwrite arrays
    origin, spacing, direction = affine_to_direction(aff.copy())
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction.T.flatten())

    return img


def sitk_image_to_array(img: sitk.Image) -> np.ndarray:
    """get the array from an sitk Image

    Parameters
    ----------
    img : sitk.Image

    Returns
    -------
    np.ndarray

    Note
    ----
    By default sitk's GetImageFromArray assumes that the axis oder
    is [z,y,x] which is why we swap axes before calling it.
    """
    # sitk assumes that in numpy we have stores [z,y,x]
    # so we have to swap axes
    return np.swapaxes(sitk.GetArrayFromImage(img), 0, 2)


def resample_sitk_image(volume: sitk.Image,
                        new_spacing: tuple[float, float, float],
                        interpolator: int = sitk.sitkLinear) -> sitk.Image:
    """Resample an sitk image to a new spacing (voxel size)

    Parameters
    ----------
    volume : sitk.Image
    new_spacing : tuple[float, float, float]
        the new spacing (voxel size)    
    interpolator : int, optional
        interpolator to use, by default sitk.sitkLinear

    Returns
    -------
    sitk.Image
        the resampled image
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [
        int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(
            original_size, original_spacing, new_spacing)
    ]

    # calculate the new origin
    old_aff = affine_from_sitk_image(volume)
    new_origin = tuple((old_aff @ np.array([
        0.5 * (new_spacing[0] / original_spacing[0] - 1), 0.5 *
        (new_spacing[1] / original_spacing[1] - 1), 0.5 *
        (new_spacing[2] / original_spacing[2] - 1), 1
    ]))[:-1])

    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         new_origin, new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())


def align_and_resample(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    fixed_affine: np.ndarray,
    moving_affine: np.ndarray,
    new_spacing: tuple[float, float, float] = (1., 1., 1.),
    sampling_rate: float = 0.01,
    init_mode: sitk.Transform | str = 'MOMENTS',
    registration_method: sitk.ImageRegistrationMethod | None = None,
    final_transform: sitk.Transform | None = None,
    resample_only: bool = False,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, sitk.Transform, np.ndarray]:
    """wrapper to align and resample two 3D images using simple ITK

    Parameters
    ----------
    fixed_volume : np.ndarray
        the fixed 3D image
    moving_volume : np.ndarray
        the moving 3D image
    fixed_affine : np.ndarray
        affine maxtrix for fixed image
    moving_affine : np.ndarray
        affine maxtrix for moving image
    new_spacing : tuple[float, float, float], optional
        the new voxel size after resampling, by default (1., 1., 1.)
    sampling_rate : float, optional
        random sampling rate used in registration, by default 0.01
    init_mode : sitk.Transform | str, optional
        initialization mode for transformation, by default 'MOMENTS'
        'MOMENTS' -> align moments of images
        'GEOMETRY' -> align FOV centers
        sitk.Transform -> use this transform
    registration_method : sitk.ImageRegistrationMethod | None, optional
        sitk registration method to use, by default None
        None -> use mattes mutual information and conjugate gradient optimizer
    final_transform : sitk.Transform | None, optional
        final transformation (e.g. loaded from file), by default None
        None -> calculate final transformation that maps moving image to
                resampled fixed image
    resample_only : bool, optional
        resample images to common grid only based on affines, default False
    verbose : bool, optional
        print verbose output, by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray, sitk.Transform, np.ndarray]
        - the resampled fixed image 
        - the aligned moving image
        - the final transform that maps the moving image to the resampled fixed image 
        - the affine matrix of the resampled fixed image
    """

    fixed_image = array_to_sitk_image(fixed_volume, fixed_affine)
    moving_image = array_to_sitk_image(moving_volume, moving_affine)

    # get the initial transform based on the affine information from the header
    resample_transform = sitk.Transform(moving_image.GetDimension(),
                                        sitk.sitkIdentity)

    if resample_only:
        # transform that only resamples bewteen the different grids
        final_transform = resample_transform
    else:
        # choose the initial transform
        if init_mode == 'MOMENTS':
            # based on moments of gray levels
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.MOMENTS)
        elif init_mode == 'GEOMETRY':
            # based on center of image volumes
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, moving_image, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
        elif isinstance(init_mode, sitk.Transform):
            initial_transform = init_mode
        else:
            raise ValueError(
                'initial tranform must be either None, GEOMETRY, MOMENTS or sitk.transform'
            )

        # Registration
        if registration_method is None:
            registration_method = sitk.ImageRegistrationMethod()

            # Similarity metric settings.
            registration_method.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(
                registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(sampling_rate)

            registration_method.SetInterpolator(sitk.sitkLinear)

            # Optimizer settings.
            registration_method.SetOptimizerAsConjugateGradientLineSearch(
                learningRate=1.,
                numberOfIterations=100,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10)

            registration_method.SetOptimizerScalesFromPhysicalShift()

            # Setup for the multi-resolution framework.
            registration_method.SetShrinkFactorsPerLevel(
                shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(
                smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

            registration_method.SetInitialTransform(initial_transform,
                                                    inPlace=False)

        if final_transform is None:
            try:
                final_transform = registration_method.Execute(
                    fixed_image, moving_image)
            except:
                warnings.warn(
                    'SITK registation failed. Using initial transform')
                final_transform = resample_transform

        # Post registration analysis
        if verbose:
            print(
                f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
            )
            print(
                f"Final metric value: {registration_method.GetMetricValue()}")
            print(f"Final parameters: {final_transform.GetParameters()}")

    # calculate the new spacing and origin and apply the transformation
    original_spacing = fixed_image.GetSpacing()
    original_size = fixed_image.GetSize()
    new_size = tuple([
        int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(
            original_size, original_spacing, new_spacing)
    ])

    # calculate the new origin
    old_aff = affine_from_sitk_image(fixed_image)
    new_origin = tuple((old_aff @ np.array([
        0.5 * (new_spacing[0] / original_spacing[0] - 1), 0.5 *
        (new_spacing[1] / original_spacing[1] - 1), 0.5 *
        (new_spacing[2] / original_spacing[2] - 1), 1
    ]))[:-1])

    interpolator = sitk.sitkLinear

    moving_image_aligned = sitk.Resample(moving_image, new_size,
                                         final_transform, interpolator,
                                         new_origin, new_spacing,
                                         fixed_image.GetDirection(),
                                         moving_volume.min(),
                                         moving_image.GetPixelID())

    fixed_image_resampled = sitk.Resample(fixed_image, new_size,
                                          sitk.Transform(), interpolator,
                                          new_origin, new_spacing,
                                          fixed_image.GetDirection(),
                                          fixed_volume.min(),
                                          fixed_image.GetPixelID())

    return sitk_image_to_array(fixed_image_resampled), sitk_image_to_array(
        moving_image_aligned), final_transform, affine_from_sitk_image(
            fixed_image_resampled)
