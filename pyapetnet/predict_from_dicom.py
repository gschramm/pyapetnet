import argparse
import os


def main():
    """ pyapetnet prediction from 3D dicom images"""
    parser = argparse.ArgumentParser(
        description='pyapetnet prediction of anatomy-guided PET reconstruction'
    )
    parser.add_argument('pet_dcm_dir',
                        help='absolute path of PET input dicom directory')
    parser.add_argument('mr_dcm_dir',
                        help='absolute path of MR  input dicom directory')
    parser.add_argument('model_name', help='name of trained CNN')

    parser.add_argument('--pet_dcm_pattern',
                        help='file pattern for PET dicom dir',
                        default='*')
    parser.add_argument('--mr_dcm_pattern',
                        help='file pattern for MR dicom dir',
                        default='*')
    parser.add_argument(
        '--model_path',
        help='absolute path of directory containing trained models',
        default=None)
    parser.add_argument('--output_dir',
                        help='name of the output directory',
                        default='.')
    parser.add_argument('--output_name',
                        help='basename of prediction file',
                        default=None)
    parser.add_argument('--no_coreg',
                        help='do not coregister input volumes',
                        action='store_true')
    parser.add_argument('--no_crop',
                        help='do not crop volumes to MR bounding box',
                        action='store_true')
    parser.add_argument('--show', help='show the results', action='store_true')
    parser.add_argument('--verbose',
                        help='print (extra) verbose output',
                        action='store_true')
    parser.add_argument('--no_preproc_save',
                        help='do not save preprocessed volumes',
                        action='store_true')
    parser.add_argument('--output_on_mr_grid',
                        help='regrid the CNN output to the original MR grid',
                        action='store_true')
    parser.add_argument(
        '--series_description_prefix',
        help=
        'dicom series description prefix for prediction - default: empty string',
        default='')
    parser.add_argument(
        '--series_description',
        help=
        'dicom series description for prediction - default: CNN MAP Bowsher',
        default='CNN AGR Bowsher')
    parser.add_argument(
        '--series_description_suffix',
        help=
        'dicom series description suffix for prediction - default model name: and package version',
        default=None)

    args = parser.parse_args()

    #-------------------------------------------------------------------------------------------------
    # load modules

    import pyapetnet
    from pyapetnet.preprocessing import preprocess_volumes
    from pyapetnet.utils import load_nii_in_ras
    from pymirc.fileio import DicomVolume, write_3d_static_dicom

    import nibabel as nib
    import json
    import tensorflow as tf

    import numpy as np
    import matplotlib.pyplot as plt
    import pymirc.viewer as pv
    from pymirc.image_operations import aff_transform

    from pyapetnet.utils import flip_ras_lps, pet_dcm_keys_to_copy
    from warnings import warn
    #-------------------------------------------------------------------------------------------------
    # parse input parameters

    pet_dcm_dir = args.pet_dcm_dir
    mr_dcm_dir = args.mr_dcm_dir
    pet_dcm_pattern = args.pet_dcm_pattern
    mr_dcm_pattern = args.mr_dcm_pattern

    model_name = args.model_name
    output_dir = args.output_dir
    output_name = args.output_name

    if output_name is None:
        output_name = f'prediction_{model_name}'

    model_path = args.model_path

    if model_path is None:
        model_path = os.path.join(os.path.dirname(pyapetnet.__file__),
                                  'trained_models')

    coreg_inputs = not args.no_coreg
    crop_mr = not args.no_crop
    show = args.show
    verbose = args.verbose
    save_preproc = not args.no_preproc_save
    output_on_mr_grid = args.output_on_mr_grid

    series_description_prefix = args.series_description_prefix
    series_description = args.series_description
    series_description_suffix = args.series_description_suffix

    series_description_prefix = args.series_description_prefix
    if series_description_suffix is None:
        series_description_suffix = f'__m{model_name}_v{pyapetnet.__version__}'

    series_description = f'{series_description_prefix}{series_description}{series_description_suffix}'

    #-------------------------------------------------------------------------------------------------
    # load the trained model

    co = {'ssim_3d_loss': None, 'mix_ssim_3d_mae_loss': None}

    if verbose:
        print('loading CNN {os.path.join(model_path, model_name)}')

    model = tf.keras.models.load_model(os.path.join(model_path, model_name),
                                       custom_objects=co)

    # load the voxel size used for training
    with open(os.path.join(model_path, model_name, 'config.json')) as f:
        cfg = json.load(f)
        training_voxsize = cfg['internal_voxsize'] * np.ones(3)

    #------------------------------------------------------------------
    # load and preprocess the input PET and MR volumes
    pet_dcm = DicomVolume(os.path.join(pet_dcm_dir, pet_dcm_pattern))
    mr_dcm = DicomVolume(os.path.join(mr_dcm_dir, mr_dcm_pattern))

    pet = pet_dcm.get_data()
    mr = mr_dcm.get_data()

    pet_affine = pet_dcm.affine
    mr_affine = mr_dcm.affine

    # preprocess the input volumes (coregistration, interpolation and intensity normalization)
    pet_preproc, mr_preproc, o_aff, pet_scale, mr_scale, _ = preprocess_volumes(
        pet,
        mr,
        pet_affine,
        mr_affine,
        training_voxsize,
        perc=99.99,
        coreg=coreg_inputs,
        crop_mr=crop_mr,
        verbose=verbose)

    #------------------------------------------------------------------
    # the actual CNN prediction
    x = [
        np.expand_dims(np.expand_dims(pet_preproc, 0), -1),
        np.expand_dims(np.expand_dims(mr_preproc, 0), -1)
    ]
    pred = model.predict(x).squeeze()

    # undo the intensity normalization
    pred *= pet_scale
    pet_preproc *= pet_scale
    mr_preproc *= mr_scale

    #------------------------------------------------------------------
    # save the preprocessed input and output
    # dicom volumes are read as LPS, but nifti volumes have to be in RAS
    os.makedirs(output_dir, exist_ok=True)

    if save_preproc:
        nib.save(nib.Nifti1Image(*flip_ras_lps(pet_preproc, o_aff)),
                 os.path.join(output_dir, 'pet_preproc.nii'))
        if verbose:
            print(
                f'wrote pre-processed PET to: {os.path.join(output_dir, "pet_preproc.nii")}'
            )
        nib.save(nib.Nifti1Image(*flip_ras_lps(mr_preproc, o_aff)),
                 os.path.join(output_dir, 'mr_preproc.nii'))
        if verbose:
            print(
                f'wrote pre-processed MR  to: {os.path.join(output_dir, "mr_preproc.nii")}'
            )

        # save the intensity normalization factors
        np.savetxt(os.path.join(output_dir, 'preproc_scaling_factors.txt'),
                   np.array([pet_scale, mr_scale]))
        if verbose:
            print(
                f'wrote scaling factors   to: {os.path.join(output_dir, "preproc_scaling_factors.txt")}'
            )

    if output_on_mr_grid:
        oss = np.ceil(
            np.linalg.norm(mr_affine[:-1, :-1], axis=0) /
            training_voxsize).astype(int)
        pred_regrid = aff_transform(pred,
                                    np.linalg.inv(o_aff) @ mr_affine,
                                    mr.shape,
                                    cval=pred.min(),
                                    os0=oss[0],
                                    os1=oss[1],
                                    os2=oss[2])
        nib.save(nib.Nifti1Image(*flip_ras_lps(pred_regrid, mr_affine)),
                 os.path.join(output_dir, f'{output_name}.nii'))
    else:
        nib.save(nib.Nifti1Image(*flip_ras_lps(pred, o_aff)),
                 os.path.join(output_dir, f'{output_name}.nii'))

    #------------------------------------------------------------------
    # save prediction also as dicom

    # get a list of dicom keys to copy from the original PET dicom header
    dcm_kwargs = {}
    for key in pet_dcm_keys_to_copy():
        try:
            dcm_kwargs[key] = getattr(pet_dcm.firstdcmheader, key)
        except AttributeError:
            warn('Cannot copy tag ' + key)

    # generate a dicom series number based on the input series numbers
    pet_series_num = 2
    if 'SeriesNumber' in pet_dcm.firstdcmheader:
        pet_series_num = int(pet_dcm.firstdcmheader.SeriesNumber)

    mr_series_num = 1
    if 'SeriesNumber' in mr_dcm.firstdcmheader:
        mr_series_num = int(mr_dcm.firstdcmheader.SeriesNumber)

    series_num = 60000 + 100*pet_series_num + mr_series_num

    # write the dicom volume
    output_dcm_dir = os.path.join(output_dir, output_name)
    if not os.path.exists(output_dcm_dir):
        if output_on_mr_grid:
            write_3d_static_dicom(
                pred_regrid,
                output_dcm_dir,
                affine=mr_affine,
                ReconstructionMethod=
                f'CNN MAP Bowsher {model_name} {pyapetnet.__version__}',
                SeriesDescription=series_description,
                SeriesNumber=series_num,
                **dcm_kwargs)
        else:
            write_3d_static_dicom(
                pred,
                output_dcm_dir,
                affine=o_aff,
                ReconstructionMethod=
                f'CNN MAP Bowsher {model_name} {pyapetnet.__version__}',
                SeriesDescription=series_description,
                SeriesNumber=series_num,
                **dcm_kwargs)

        print(f'wrote prediction to: {output_dcm_dir}')
    else:
        warn('Output dicom directory already exists. Not overwriting it')

    #------------------------------------------------------------------
    # show the results
    if show:
        pmax = np.percentile(pred, 99.9)
        mmax = np.percentile(mr_preproc, 99.9)

        ims = [{
            'vmin': 0,
            'vmax': mmax,
            'cmap': plt.cm.Greys_r
        }, {
            'vmin': 0,
            'vmax': pmax
        }, {
            'vmin': 0,
            'vmax': pmax
        }]
        vi = pv.ThreeAxisViewer([mr_preproc, pet_preproc, pred],
                                imshow_kwargs=ims)
        plt.show()


if __name__ == '__main__':
    main()
