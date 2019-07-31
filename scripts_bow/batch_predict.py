import sys
if not '..' in sys.path: sys.path.append('..')

import os
from pyapetnet.predictors import predict


pdir      = '../data/test_data/mMR/Tim-Patients/test'
mlem_sdir = '20_min'
mlem_file = 'mlem_3_21.nii'
mr_file   = 'aligned_t1.nii'

model_dir  = '../data/trained_models'
model_name = '190528_paper_bet_10_psf_mlem.h5'

if not os.path.exists(os.path.join(pdir,'predictions')):
  os.makedirs(os.path.join(pdir,'predictions'))

predict(mr_input      = os.path.join(pdir,mr_file),
        pet_input     = os.path.join(pdir,mlem_sdir,mlem_file),
        input_format  = 'nifti',
        odir          = os.path.join(pdir,'predictions','p1'),
        model_name    = model_name,
        model_dir     = model_dir,
        coreg         = False,
        crop_mr       = False,
        patchsize     = (128,128,128),
        overlap       = 8,
        debug_mode    = False)


