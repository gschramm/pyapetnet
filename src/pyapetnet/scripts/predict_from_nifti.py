from __future__ import annotations

import click
import os

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("pet_fname")
@click.argument("mr_fname")
@click.argument("model_name")
@click.option(
    "--model_path",
    help="absolute path of directory containing trained models",
    default=None,
)
@click.option("--output_dir", help="name of the output directory", default=".")
@click.option("--output_name", help="basename of prediction file", default=None)
@click.option(
    "--coreg_inputs/--no-coreg_inputs", default=True, help="coregister input volumes"
)
@click.option(
    "--crop_mr/--no-crop_mr", default=True, help="crop volumes to MR bounding box"
)
@click.option("--show/--no-show", default=True, help="show the results")
@click.option(
    "--verbose/--no-verbose", default=False, help="print (extra) verbose output"
)
@click.option(
    "--save_preproc/--no-save_preproc", default=True, help="save preprocessed volumes"
)
@click.option(
    "--output_on_mr_grid/--no-output_on_mr_grid",
    default=False,
    help="regrid the CNN output to the original MR grid",
)
def predict_from_nifti(
    pet_fname: str,
    mr_fname: str,
    model_name: str,
    model_path: str,
    output_dir: str,
    output_name: str,
    coreg_inputs: bool,
    crop_mr: bool,
    show: bool,
    verbose: bool,
    save_preproc: bool,
    output_on_mr_grid: bool,
) -> None:

    import pyapetnet
    from pyapetnet.preprocessing import preprocess_volumes
    from pyapetnet.utils import load_nii_in_ras

    import nibabel as nib
    import json
    import tensorflow as tf

    import numpy as np
    import matplotlib.pyplot as plt
    import pymirc.viewer as pv
    from pymirc.image_operations import aff_transform

    if output_name is None:
        output_name = f"prediction_{model_name}.nii"
    if model_path is None:
        model_path = os.path.join(os.path.dirname(pyapetnet.__file__), "trained_models")

    # -------------------------------------------------------------------------------------------------
    # load the trained model

    co = {"ssim_3d_loss": None, "mix_ssim_3d_mae_loss": None}

    if verbose:
        print("loading CNN {os.path.join(model_path, model_name)}")

    model = tf.keras.models.load_model(
        os.path.join(model_path, model_name), custom_objects=co
    )

    # load the voxel size used for training
    with open(os.path.join(model_path, model_name, "config.json")) as f:
        cfg = json.load(f)
        training_voxsize = cfg["internal_voxsize"] * np.ones(3)

    # ------------------------------------------------------------------
    # load and preprocess the input PET and MR volumes
    pet, pet_affine = load_nii_in_ras(pet_fname)
    mr, mr_affine = load_nii_in_ras(mr_fname)

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
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # the actual CNN prediction
    x = [
        np.expand_dims(np.expand_dims(pet_preproc, 0), -1),
        np.expand_dims(np.expand_dims(mr_preproc, 0), -1),
    ]
    pred = model.predict(x).squeeze()

    # undo the intensity normalization
    pred *= pet_scale
    pet_preproc *= pet_scale
    mr_preproc *= mr_scale

    # ------------------------------------------------------------------
    # save the preprocessed input and output
    os.makedirs(output_dir, exist_ok=True)

    if save_preproc:
        nib.save(
            nib.Nifti1Image(pet_preproc, o_aff),
            os.path.join(output_dir, "pet_preproc.nii"),
        )
        if verbose:
            print(
                f'wrote pre-processed PET to: {os.path.join(output_dir, "pet_preproc.nii")}'
            )

        nib.save(
            nib.Nifti1Image(mr_preproc, o_aff),
            os.path.join(output_dir, "mr_preproc.nii"),
        )
        if verbose:
            print(
                f'wrote pre-processed MR  to: {os.path.join(output_dir, "mr_preproc.nii")}'
            )

        # save the intensity normalization factors
        np.savetxt(
            os.path.join(output_dir, "preproc_scaling_factors.txt"),
            np.array([pet_scale, mr_scale]),
        )
        if verbose:
            print(
                f'wrote scaling factors   to: {os.path.join(output_dir, "preproc_scaling_factors.txt")}'
            )

    if output_on_mr_grid:
        oss = np.ceil(
            np.linalg.norm(mr_affine[:-1, :-1], axis=0) / training_voxsize
        ).astype(int)
        pred_regrid = aff_transform(
            pred,
            np.linalg.inv(o_aff) @ mr_affine,
            mr.shape,
            cval=pred.min(),
            os0=oss[0],
            os1=oss[1],
            os2=oss[2],
        )
        nib.save(
            nib.Nifti1Image(pred_regrid, mr_affine),
            os.path.join(output_dir, output_name),
        )
    else:
        nib.save(nib.Nifti1Image(pred, o_aff), os.path.join(output_dir, output_name))
    if verbose:
        print(f"wrote prediction to       : {os.path.join(output_dir, output_name)}")

    # ------------------------------------------------------------------
    # show the results
    if show:
        pmax = np.percentile(pred, 99.9)
        mmax = np.percentile(mr_preproc, 99.9)

        ims = [
            {"vmin": 0, "vmax": mmax, "cmap": plt.cm.Greys_r},
            {"vmin": 0, "vmax": pmax},
            {"vmin": 0, "vmax": pmax},
        ]
        vi = pv.ThreeAxisViewer(
            [
                np.flip(mr_preproc, (0, 1)),
                np.flip(pet_preproc, (0, 1)),
                np.flip(pred, (0, 1)),
            ],
            imshow_kwargs=ims,
        )
        plt.show()


if __name__ == "__main__":
    predict_from_nifti()
