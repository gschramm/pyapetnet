import sys
if not '..' in sys.path: sys.path.append('..')

import os
from pyapetnet.predictors import predict

from glob import glob

mdir  = '../data/test_data/mMR/Tim-Patients'
pdirs = glob(os.path.join(mdir,'Tim-Patient-*'))

model_dir  = '../data/trained_models'
model_name = '190528_paper_bet_10_psf_mlem.h5'

osem_sdir = '20_min'
osem_file = 'osem_psf_3_4.nii'
mr_file   = 'aligned_t1.nii'

for pdir in pdirs:
  print(pdir)

  output_dir = os.path.join(pdir,'predictions',osem_sdir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  output_file = os.path.join(output_dir, '___'.join([os.path.splitext(model_name)[0],osem_file]))

  if not os.path.exists(output_file):
    predict(mr_input           = os.path.join(pdir,mr_file),
            pet_input          = os.path.join(pdir,osem_sdir,osem_file),
            input_format       = 'nifti',
            odir               = os.path.splitext(output_file)[0],
            model_name         = model_name,
            model_dir          = model_dir,
            coreg              = False,
            crop_mr            = True,
            patchsize          = (128,128,128),
            overlap            = 8,
            output_on_pet_grid = True,
            debug_mode         = False)
  else:
    print(output_file,' already exists.')
   


