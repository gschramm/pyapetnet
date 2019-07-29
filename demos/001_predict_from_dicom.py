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

pdir = os.path.join('..','data','demo_data','ANON1002')

predict(mr_input      = os.path.join(pdir,'BRAVO_MR','*.img'),       # the high res MR input nifti file
        pet_input     = os.path.join(pdir,'PT','*.img'),             # the low res PET input nifti file
        input_format  = 'dicom',                                     # the input image format
        odir          = os.path.join('..','data',
                                     'demo_data','test_prediction'), # the directory for the output dicoms
        model_name    = '190528_paper_bet_10_psf_mlem.h5',           # the file basename containing the 
                                                                     # trained CNN
        model_dir     = os.path.join('..','data','trained_models'),  # the directory where the CNN file sits
        perc          = 99.99,                                       # precentile used for data normalization
        patchsize     = (128,128,128),                               # patchsize used in prediction
        verbose       = True,                                        # print verbose output
        clip_neg      = True,                                        # clip negative values in output
        coreg         = True,                                        # rigidly align the inputs using mutual inf.
        affine        = None,                                        # a file containing registration parameters
        crop_mr       = False,                                       # crop input volumes to the MR head contour 
        debug_mode    = False)                                       # debug mode saves some intermediate files
