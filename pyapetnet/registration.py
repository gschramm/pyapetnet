# TODO return / save final transformation, init reg by translation
import numpy as np
import SimpleITK as sitk

def affine_to_direction(aff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = aff[:-1,-1]
    direction = aff[:-1,:-1]
    spacing = np.linalg.norm(direction, axis = 0)
    direction /= spacing

    return origin, spacing, direction

def array_to_sitk_image(arr: np.ndarray, aff: np.ndarray) -> sitk.Image:
    # sitk assumes that in numpy we have stores [z,y,x]
    # so we have to swap axes
    img = sitk.GetImageFromArray(np.swapaxes(arr,0,2))
    # remember to make a copy of affine, otherwise sITK operations will overwrite arrays
    origin, spacing, direction = affine_to_direction(aff.copy())
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction.T.flatten())

    return img

def sitk_image_to_array(img: sitk.Image) -> np.ndarray:
    # sitk assumes that in numpy we have stores [z,y,x]
    # so we have to swap axes
    return np.swapaxes(sitk.GetArrayFromImage(img),0,2)


def resample_sitk_image(volume: sitk.Image,
                    new_spacing: tuple[float, float, float],
                    interpolator: int = sitk.sitkLinear) -> sitk.Image:
    """ resample volume to differnt voxel size"""
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [
        int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(
            original_size, original_spacing, new_spacing)
    ]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing,
                         volume.GetDirection(), 0, volume.GetPixelID())

def align_and_resample(fixed_image: sitk.Image | np.ndarray,
                       moving_image: sitk.Image | np.ndarray,
                       fixed_affine: np.ndarray | None = None,
                       moving_affine: np.ndarray | None = None,
                       new_spacing: tuple[float, float, float] = (1., 1., 1.),
                       sampling_rate: float = 0.01,
                       init_mode: sitk.Transform | str | None = 'MOMENTS',
                       registration_method: sitk.ImageRegistrationMethod | None = None,
                       seed: int = 0,
                       verbose: bool = False) -> tuple[sitk.Image, sitk.Image]:
    """ align two SITK images and interpolate to a nomimal voxel spacing if needed """

    if isinstance(fixed_image, np.ndarray):
        if fixed_affine is None:
            raise TypeError('affine must be given if input is np.ndarray')
        fixed_image = array_to_sitk_image(fixed_image, fixed_affine)

    if isinstance(moving_image, np.ndarray):
        if moving_affine is None:
            raise TypeError('affine must be given if input is np.ndarray')
        moving_image = array_to_sitk_image(moving_image, moving_affine)

    if not tuple(new_spacing) == fixed_image.GetSpacing():
        fixed_image = resample_sitk_image(fixed_image, new_spacing)

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
    elif init_mode is None:
        # get the initial transform based on the affine information from the header 
        # not working yet
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        _ = resampler.Execute(moving_image)
        initial_transform = resampler.GetTransform()
    else:
        raise ValueError('initial tranform must be either None, GEOMETRY, MOMENTS or sitk.transform')


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
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10)

        if (init_mode == 'GEOMETRY') or (init_mode == 'MOMENTS'):
            registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Post registration analysis
    if verbose:
        print(
            f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
        )
        print(f"Final metric value: {registration_method.GetMetricValue()}")
        print(f"Final parameters: {final_transform.GetParameters()}")

    moving_image_aligned = sitk.Resample(moving_image, fixed_image,
                                         final_transform, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelID())

    return fixed_image, moving_image_aligned

if __name__ == '__main__':
    reader = sitk.ImageSeriesReader()
    
    pet_image = sitk.ReadImage('pet.nii', outputPixelType=sitk.sitkFloat32)
    mr_image = sitk.ReadImage('flair.nii', outputPixelType=sitk.sitkFloat32)

    from utils import load_nii_in_ras
    pet_vol, pet_aff = load_nii_in_ras('pet.nii')
    mr_vol, mr_aff = load_nii_in_ras('flair.nii')

    pet_vol = np.roll(pet_vol, 25, 0)

    a, b = align_and_resample(mr_vol, pet_vol, fixed_affine=mr_aff, moving_affine=pet_aff, 
                             new_spacing=(0.9,0.9,0.9), verbose = True)

    c, d = align_and_resample(mr_vol, pet_vol, fixed_affine=mr_aff, moving_affine=pet_aff, 
                             new_spacing=(0.9,0.9,0.9), verbose = True)

    import pymirc.viewer as pv
    vi = pv.ThreeAxisViewer([sitk.GetArrayFromImage(c), sitk.GetArrayFromImage(d)])