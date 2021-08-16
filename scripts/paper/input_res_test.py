# script for comparing prediction for FLAIR vs T1
import sys
import os
import h5py
import pickle
import nibabel as nib
import numpy   as np
import pylab   as py
import pymirc.viewer as pymv


from scipy.ndimage         import find_objects, zoom, gaussian_filter
from keras.models          import load_model

#if not apetcnn_path in sys.path: sys.path.append(apetcnn_path)
#from pyapetnet.predictors import predict
#---------------------------------------

pdir = '../../data/validation_data/mMR/Tim-Patients/Tim-Patient-67'

model_name = '190904_osem_nopsf_bet_10_bs_52_ps_29.h5'
model_dir  = '../../data/trained_models'

model = load_model(os.path.join(model_dir,model_name))

# read the internal voxel size that was used during training from the model header
model_data = h5py.File(os.path.join(model_dir,model_name))

training_voxsize = model_data['header/internal_voxsize'][:]

# load the data
mr_data   = nib.load(os.path.join(pdir,'aligned_t1.nii'))
mr_data   = nib.as_closest_canonical(mr_data)
mr        = np.flip(mr_data.get_data(),(0,1))
zoomfacs  = np.sqrt((mr_data.affine**2).sum(axis = 0))[:-1] / training_voxsize
mr        = zoom(mr, zoomfacs, order = 1, prefilter = False)

mlem_data   = nib.load(os.path.join(pdir,'20_min','osem_nopsf.nii'))
mlem_data   = nib.as_closest_canonical(mlem_data)
mlem        = np.flip(mlem_data.get_data(),(0,1))
zoomfacs    = np.sqrt((mlem_data.affine**2).sum(axis = 0))[:-1] / training_voxsize
mlem        = zoom(mlem, zoomfacs, order = 1, prefilter = False)

bow_data   = nib.load(os.path.join(pdir,'20_min','bow_bet_1.0E+01_psf_4_5.nii'))
bow_data   = nib.as_closest_canonical(bow_data)
bow        = np.flip(bow_data.get_data(),(0,1))
zoomfacs   = np.sqrt((bow_data.affine**2).sum(axis = 0))[:-1] / training_voxsize
bow        = zoom(bow, zoomfacs, order = 1, prefilter = False)

# crop the data
bbox      = find_objects(mr > 0.1*mr.max(), max_label = 1)[0]
mr        = mr[bbox]
mlem      = mlem[bbox]
bow       = bow[bbox]

# scale the images
pfac = np.percentile(mlem,99.99)
mfac = np.percentile(mr,99.99)

mlem /= pfac
bow  /= pfac
mr   /= mfac

# predictions with original data
p1 = model.predict([np.expand_dims(np.expand_dims(mlem,0),-1), np.expand_dims(np.expand_dims(mr,0),-1)]).squeeze()

# prediction of 6mm res osem data
mlem_5mm = gaussian_filter(mlem, np.sqrt(5**2 - 4.5**2)/(2.35*training_voxsize))
mlem_6mm = gaussian_filter(mlem, np.sqrt(6**2 - 4.5**2)/(2.35*training_voxsize))

p5 = model.predict([np.expand_dims(np.expand_dims(mlem_5mm,0),-1), np.expand_dims(np.expand_dims(mr,0),-1)]).squeeze()
p6 = model.predict([np.expand_dims(np.expand_dims(mlem_6mm,0),-1), np.expand_dims(np.expand_dims(mr,0),-1)]).squeeze()

imshow_kwargs = 4*[{'vmin':0, 'vmax':1.2}]
#pymv.ThreeAxisViewer([p1,p5,p6,bow], imshow_kwargs = imshow_kwargs)
pymv.ThreeAxisViewer([mlem, mlem_5mm, mlem_6mm], imshow_kwargs = imshow_kwargs)
