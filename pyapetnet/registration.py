import SimpleITK as sitk

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

def align_sitk_images(fixed_image: sitk.Image,
                      moving_image: sitk.Image,
                      new_spacing: tuple[float, float, float] = (1., 1., 1.),
                      sampling_rate: float = 0.01,
                      initial_transform: sitk.Transform | None = None,
                      registration_method: sitk.ImageRegistrationMethod | None = None,
                      verbose: bool = False) -> tuple[sitk.Image, sitk.Image]:
    """ align two SITK images and interpolate to a nomimal voxel spacing if needed """

    if not tuple(new_spacing) == fixed_image.GetSpacing():
        fixed_image = resample_sitk_image(fixed_image, new_spacing)

    # Initial transform -> align image centers based on affine
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, moving_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

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

        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        registration_method.SetInitialTransform(initial_transform)

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
