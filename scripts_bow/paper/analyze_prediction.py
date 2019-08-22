import os
import nibabel as nib
import numpy   as np
import pandas  as pd
import re
import seaborn as sns
import pylab   as py

from glob            import glob
from skimage.measure import compare_ssim as ssim

#------------------------------------------------------------------------------------------------------------
def read_nii(fname):
  nii = nib.load(fname)
  nii = nib.as_closest_canonical(nii)
  vol = np.flip(np.flip(nii.get_data(),0),1)

  return vol, nii.header['pixdim'][1:4]

#------------------------------------------------------------------------------------------------------------
def regional_statistics(vol, ref_vol, labelimg):
 
  _,ss_img = ssim(vol, ref_vol, full = True, data_range = ref_vol.max() - ref_vol.min())

  df = pd.DataFrame()

  for roinum in np.unique(labelimg):
    roiinds = np.where(labelimg == roinum)
   
    x = vol[roiinds]
    y = ref_vol[roiinds]

    data = {'roinum':  roinum, 
            'mean':    x.mean(), 
            'rc_mean': x.mean()/y.mean(), 
            'ssim':    ss_img[roiinds].mean(), 
            'rmse':    np.sqrt(((x - y)**2).mean())/y.mean()}

    df = df.append([data], ignore_index = True)
 
  return df
#------------------------------------------------------------------------------------------------------------
def roi_to_region(roi):

  if '-Cerebral-White-Matter'      in roi: region = 'wm'
  elif '-Ventricle'                in roi: region = 'ventricle'
  elif '-Cerebellum-White-Matter'  in roi: region = 'cereb_wm'
  elif '-Cerebellum-Cortex'        in roi: region = 'cereb_cortex'
  elif '-Thalamus'                 in roi: region = 'thalamus'
  elif '-Caudate'                  in roi: region = 'basal_ganglia'
  elif '-Putamen'                  in roi: region = 'basal_ganglia'
  elif '-Pallidum'                 in roi: region = 'basal_ganglia'
  elif '3rd-Ventricle'             in roi: region = 'ventricle'
  elif '4th-Ventricle'             in roi: region = 'ventricle'
  elif '-Hippocampus'              in roi: region = 'hippocampus'
  elif '-Amygdala'                 in roi: region = 'temporal'
  elif '-Insula'                   in roi: region = 'other'
  elif '-Accumbens-area'           in roi: region = 'basal_ganglia'
  elif roi == 'Unknown':                   region = 'background'
  elif bool(re.match(r'ctx-.*-corpuscallosum',roi)):   region = 'corpuscallosum'
  elif bool(re.match(r'ctx-.*-cuneus',roi)):           region = 'occipital'
  elif bool(re.match(r'ctx-.*-entorhinal',roi)):       region = 'temporal'
  elif bool(re.match(r'ctx-.*-fusiform',roi)):         region = 'temporal'
  elif bool(re.match(r'ctx-.*-paracentral',roi)):      region = 'frontal'
  elif bool(re.match(r'ctx-.*-parsopercularis',roi)):  region = 'frontal'
  elif bool(re.match(r'ctx-.*-parsorbitalis',roi)):    region = 'frontal'
  elif bool(re.match(r'ctx-.*-parstriangularis',roi)): region = 'frontal'
  elif bool(re.match(r'ctx-.*-pericalcarine',roi)):    region = 'occipital'
  elif bool(re.match(r'ctx-.*-postcentral',roi)):      region = 'parietal'
  elif bool(re.match(r'ctx-.*-precentral',roi)):       region = 'frontal'
  elif bool(re.match(r'ctx-.*-precuneus',roi)):        region = 'parietal'
  elif bool(re.match(r'ctx-.*-supramarginal',roi)):    region = 'parietal'
  elif bool(re.match(r'ctx-.*-frontalpole',roi)):      region = 'frontal'
  elif bool(re.match(r'ctx-.*-temporalpole',roi)):     region = 'temporal'
  elif bool(re.match(r'ctx-.*-insula',roi)):           region = 'insula'
  elif bool(re.match(r'ctx-.*frontal',roi)):           region = 'frontal'
  elif bool(re.match(r'ctx-.*parietal',roi)):          region = 'parietal'
  elif bool(re.match(r'ctx-.*temporal',roi)):          region = 'temporal'
  elif bool(re.match(r'ctx-.*cingulate',roi)):         region = 'cingulate'
  elif bool(re.match(r'ctx-.*occipital',roi)):         region = 'occipital'
  elif bool(re.match(r'ctx-.*lingual',roi)):           region = 'occipital'
  elif bool(re.match(r'ctx-.*hippocampal',roi)):       region = 'temporal'
  else:                                                region = 'other' 

  return region

#------------------------------------------------------------------------------------------------------------

mdir      = '../../data/test_data/mMR/Tim-Patients'
pdirs     = glob(os.path.join(mdir,'Tim-Patient-*'))
recompute = False

model_dir  = '../data/trained_models'
model_name = '190528_paper_bet_10_psf_mlem.h5'

osem_sdir  = '20_min'
osem_file  = 'osem_psf_3_4.nii'
bow_file   = 'bow_bet_1.0E+01_psf_3_4.nii'
mr_file    = 'aligned_t1.nii'
aparc_file = 'aparc+aseg_native.nii'

roilut = pd.read_table('FreeSurferColorLUT.txt', comment = '#', sep = '\s+', 
                       names = ['num','roi','r','g','b','a'])

reg_results = pd.DataFrame()

for pdir in pdirs:
  print(pdir)

  output_dir      = os.path.join(pdir,'predictions',osem_sdir)
  prediction_file = os.path.join(output_dir, '___'.join([os.path.splitext(model_name)[0],osem_file]))

  bow, voxsize = read_nii(os.path.join(pdir,osem_sdir,bow_file))
  cnn_bow, _   = read_nii(prediction_file)
  aparc, _     = read_nii(os.path.join(pdir,aparc_file))

  df_file = os.path.splitext(prediction_file)[0] + '_regional_stats.csv'

  if (not os.path.exists(df_file)) or recompute:
    df             = regional_statistics(cnn_bow, bow, aparc)
    df["subject"]  = os.path.basename(pdir)
    df['roiname']  = df['roinum'].apply(lambda x: roilut[roilut.num == x].roi.to_string(index = False))
    df['region']   = df["roiname"].apply(roi_to_region)
    df['bow_file'] = bow_file
    df             = df.reindex(columns=['subject','roinum','roiname','region','bow_file',
                                         'mean','rc_mean','rmse','ssim'])
 
    df.to_csv(df_file) 
    print('wrote: ', df_file)
  else:
    print('reading : ', df_file)
    df = pd.read_csv(df_file)
  
  reg_results = reg_results.append([df], ignore_index = True)

#---------------------------------------------------------------------------------
# filter background ROIs
reg_results = reg_results.loc[(reg_results['region'] != 'other') & 
                              (reg_results['region'] != 'background')]

#---------------------------------------------------------------------------------
# make plots

fp = dict(marker = 'o', markerfacecolor = '0.3', markeredgewidth = 0, markersize = 2.5) 

fig, ax = py.subplots(3,1, figsize = (12,8), sharex = True)
sns.boxplot(x='region',  y ='rc_mean', data = reg_results, ax = ax[0], color = 'r', flierprops = fp)
sns.boxplot(x='region',  y ='ssim',    data = reg_results, ax = ax[1], color = 'r', flierprops = fp)
sns.boxplot(x='region',  y ='rmse',    data = reg_results, ax = ax[2], color = 'r', flierprops = fp)
for axx in ax: 
  axx.set_xticklabels(axx.get_xticklabels(),rotation=15)
  axx.grid(ls = ':')
fig.tight_layout()
fig.show()

fig2, ax2 = py.subplots(3,1, figsize = (12,8), sharex = True)
sns.boxplot(x='subject',  y ='rc_mean', data = reg_results, ax = ax2[0], color = 'r', flierprops = fp)
sns.boxplot(x='subject',  y ='ssim',    data = reg_results, ax = ax2[1], color = 'r', flierprops = fp)
sns.boxplot(x='subject',  y ='rmse',    data = reg_results, ax = ax2[2], color = 'r', flierprops = fp)
for axx in ax2: 
  axx.set_xticklabels(axx.get_xticklabels(),rotation=15)
  axx.grid(ls = ':')
fig2.tight_layout()
fig2.show()

