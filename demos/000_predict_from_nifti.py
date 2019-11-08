import sys, os
if not '..' in sys.path: sys.path.append('..')

from pyapetnet.predictors import predict

"""
Tutorial on how to do a Bowsher CNN preduction from 2 nifti input files
-----------------------------------------------------------------------

Steps done by the predict() function from pyapetnet.predictors:

(1) read 2 input images (nifti or dicom)

(2) use the affine matrix stored in the headers to regrid the images on a common
    1x1x1 mm^3 grid

(3) (optional) regidly coregister the input images

(3) normalize the input images

(4) perform a CNN prediction

(5) unnormalize the preduction

(6) save the output (as nifti for nifti input, as nifti and dicom for dicom input)
"""

pdir = '/foo/bar'

predict(mr_input      = os.path.join(pdir,'sim_t1.nii'),             # the input MR nifti file
        pet_input     = os.path.join(pdir,'sim_osem.nii'),           # the input PET nifti file
        input_format  = 'nifti',                                     # the input image format
        odir          = os.path.join(pdir,'test_prediction'),        # the name of the output nifti (.nii is added)
        model_name    = 'test_model.h5',                             # the file basename of the trained CNN
        model_dir     = pdir,                                        # the directory where the CNN file sits
        perc          = 99.99,                                       # precentile used for data normalization
        patchsize     = (64,64,64),                                  # patchsize used in prediction
        verbose       = True,                                        # print verbose output
        clip_neg      = True,                                        # clip negative values in output
        coreg         = False,                                       # rigidly align the inputs using mutual inf.
        affine        = None,                                        # a file containing registration parameters
        crop_mr       = False,                                       # crop input volumes to the MR head contour 
        debug_mode    = False)                                       # debug mode saves some intermediate files
