import os
import sys
if not os.path.join('..','..') in sys.path: sys.path.append(os.path.join('..','..'))

import nibabel as nib
import numpy   as np
import pandas  as pd
import re

import matplotlib as mpl
if os.getenv('DISPLAY') is None: mpl.use('Agg')
import matplotlib.pyplot as py

import seaborn as sns
import pickle

from glob            import glob
from skimage.metrics import structural_similarity as ssim

from pymirc.viewer import ThreeAxisViewer

from scipy.ndimage    import find_objects, zoom

py.rcParams.update({'mathtext.default':  'regular' })

#------------------------------------------------------------------------------------------------------------
def read_nii(fname):
  nii = nib.load(fname)
  nii = nib.as_closest_canonical(nii)
  vol = nii.get_data()

  return vol, nii.header['pixdim'][1:4]

#------------------------------------------------------------------------------------------------------------
def regional_statistics(vol, ref_vol, labelimg):

  #_,ss_img = ssim(vol.astype(np.float32), ref_vol.astype(np.float32), full = True)
  _,ss_img = ssim(vol.astype(np.float32), ref_vol.astype(np.float32), full = True, data_range = 2*ref_vol.max(),
                   gaussian_weights = True)

  df = pd.DataFrame()

  for roinum in np.unique(labelimg):
    roiinds = np.where(labelimg == roinum)
    x = vol[roiinds]
    y = ref_vol[roiinds]

    data = {'roinum':  roinum, 
            'mean':    x.mean(), 
            'rc_mean': x.mean()/y.mean(), 
            'ssim':    ss_img[roiinds].mean(), 
            'rmse':    np.sqrt(((x - y)**2).mean())/y.mean(),
            'nvox':    len(roiinds[0])}

    df = df.append([data], ignore_index = True)

  return df
#------------------------------------------------------------------------------------------------------------
def roi_to_region(roi):

  if '-Cerebral-White-Matter'      in roi: region = 'white matter'
  elif '-Ventricle'                in roi: region = 'ventricle'
  elif '-Cerebellum-White-Matter'  in roi: region = 'cerebellum'
  elif '-Cerebellum-Cortex'        in roi: region = 'cerebellum'
  elif '-Thalamus'                 in roi: region = 'thalamus'
  elif '-Caudate'                  in roi: region = 'basal ganglia'
  elif '-Putamen'                  in roi: region = 'basal ganglia'
  elif '-Pallidum'                 in roi: region = 'basal ganglia'
  elif '3rd-Ventricle'             in roi: region = 'ventricle'
  elif '4th-Ventricle'             in roi: region = 'ventricle'
  elif '-Hippocampus'              in roi: region = 'hippocampus'
  elif '-Amygdala'                 in roi: region = 'temporal cortex'
  elif '-Insula'                   in roi: region = 'temporal cortex'
  elif '-Accumbens-area'           in roi: region = 'basal ganglia'
  elif roi == 'Unknown':                   region = 'background'
  elif bool(re.match(r'ctx-.*-corpuscallosum',roi)):   region = 'corpuscallosum'
  elif bool(re.match(r'ctx-.*-cuneus',roi)):           region = 'occipital cortex'
  elif bool(re.match(r'ctx-.*-entorhinal',roi)):       region = 'temporal cortex'
  elif bool(re.match(r'ctx-.*-fusiform',roi)):         region = 'temporal cortex'
  elif bool(re.match(r'ctx-.*-paracentral',roi)):      region = 'frontal cortex'
  elif bool(re.match(r'ctx-.*-parsopercularis',roi)):  region = 'frontal cortex'
  elif bool(re.match(r'ctx-.*-parsorbitalis',roi)):    region = 'frontal cortex'
  elif bool(re.match(r'ctx-.*-parstriangularis',roi)): region = 'frontal cortex'
  elif bool(re.match(r'ctx-.*-pericalcarine',roi)):    region = 'occipital cortex'
  elif bool(re.match(r'ctx-.*-postcentral',roi)):      region = 'parietal cortex'
  elif bool(re.match(r'ctx-.*-precentral',roi)):       region = 'frontal cortex'
  elif bool(re.match(r'ctx-.*-precuneus',roi)):        region = 'parietal cortex'
  elif bool(re.match(r'ctx-.*-supramarginal',roi)):    region = 'parietal cortex'
  elif bool(re.match(r'ctx-.*-frontalpole',roi)):      region = 'frontal cortex'
  elif bool(re.match(r'ctx-.*-temporalpole',roi)):     region = 'temporal cortex'
  elif bool(re.match(r'ctx-.*-insula',roi)):           region = 'temporal cortex'
  elif bool(re.match(r'ctx-.*frontal',roi)):           region = 'frontal cortex'
  elif bool(re.match(r'ctx-.*parietal',roi)):          region = 'parietal cortex'
  elif bool(re.match(r'ctx-.*temporal',roi)):          region = 'temporal cortex'
  elif bool(re.match(r'ctx-.*cingulate',roi)):         region = 'cingulate cortex'
  elif bool(re.match(r'ctx-.*occipital',roi)):         region = 'occipital cortex'
  elif bool(re.match(r'ctx-.*lingual',roi)):           region = 'occipital cortex'
  elif bool(re.match(r'ctx-.*hippocampal',roi)):       region = 'temporal cortex'
  else:                                                region = 'other' 

  return region

#------------------------------------------------------------------------------------------------------------
from argparse import ArgumentParser
parser = ArgumentParser(description = 'boxplots of CNN Bowsher models')

parser.add_argument('model_name', help = 'model to analyze')

parser.add_argument('--tracer', default = 'FDG', choices = ['FDG','PE2I','FET'], help = 'data set to analyze')
parser.add_argument('--osem_sdir', default = '20_min', help = 'osem count level')
parser.add_argument('--osem_file', default = 'osem_psf_4_5.nii', help = 'osem file to use')
parser.add_argument('--bow_file',  default = 'bow_bet_1.0E+01_psf_4_5.nii', help = 'bowsher file to use')

args = parser.parse_args()

model_name = args.model_name
tracer     = args.tracer
osem_sdir  = args.osem_sdir
osem_file  = args.osem_file
bow_file   = args.bow_file

model_dir  = '../../data/trained_models'
recompute  = False

#------------------------------------------------------------------------------------------------------------
mr_file    = 'aligned_t1.nii'
aparc_file = 'aparc+aseg_native.nii'

roilut = pd.read_table('FreeSurferColorLUT.txt', comment = '#', sep = '\s+', 
                         names = ['num','roi','r','g','b','a'])

reg_results = pd.DataFrame()

lps_flip = lambda x: np.flip(np.flip(x,0),1)

if tracer == 'FDG':
  mdir      = '../../data/test_data/mMR/Tim-Patients'
  pdirs     = glob(os.path.join(mdir,'Tim-Patient-*'))
elif tracer == 'PE2I':
  mdir      = '../../data/test_data/signa/signa-pe2i'
  pdirs     = glob(os.path.join(mdir,'ANON????'))
elif tracer == 'FET':
  mdir      = '../../data/test_data/signa/signa-fet'
  pdirs     = glob(os.path.join(mdir,'ANON????'))
else:
  raise ValueError('Invalid tracer: ', tracer)

for pdir in pdirs:
  print(pdir)

  output_dir      = os.path.join(pdir,'predictions',osem_sdir)
  prediction_file = os.path.join(output_dir, '___'.join([os.path.splitext(model_name)[0],osem_file]))

  # read the prediction
  cnn_bow, voxsize   = read_nii(prediction_file)
  bbox_data          = pickle.load(open(os.path.splitext(prediction_file)[0] + '_bbox.pkl','rb'))

  bow, _    = read_nii(os.path.join(pdir,'20_min',bow_file))
  osem, _   = read_nii(os.path.join(pdir,osem_sdir,osem_file))
  aparc, _  = read_nii(os.path.join(pdir,aparc_file))
  mr, _     = read_nii(os.path.join(pdir,mr_file))

  # crop and interpolate
  bow = bow[bbox_data['bbox']]
  bow = zoom(bow, bbox_data['zoomfacs'], order = 1, prefilter = False)

  osem = osem[bbox_data['bbox']]
  osem = zoom(osem, bbox_data['zoomfacs'], order = 1, prefilter = False)

  mr = mr[bbox_data['bbox']]
  mr = zoom(mr, bbox_data['zoomfacs'], order = 1, prefilter = False)

  aparc = aparc[bbox_data['bbox']]
  aparc = zoom(aparc, bbox_data['zoomfacs'], order = 0, prefilter = False)

  df_file = os.path.splitext(prediction_file)[0] + '_regional_stats.csv'

  if (not os.path.exists(df_file)) or recompute:
    df             = regional_statistics(cnn_bow, bow, aparc)
    df["subject"]  = os.path.basename(pdir)
    df['roiname']  = df['roinum'].apply(lambda x: roilut[roilut.num == x].roi.to_string(index = False).strip())
    df['region']   = df["roiname"].apply(roi_to_region)
    df['bow_file'] = bow_file
    df             = df.reindex(columns=['subject','roinum','roiname','region','bow_file','nvox',
                                         'mean','rc_mean','rmse','ssim'])
 
    df.to_csv(df_file) 
    print('wrote: ', df_file)

    # plot the results
    lputamen_bbox = find_objects(lps_flip(aparc) == 12)
    sl0 = int(0.5*(lputamen_bbox[0][0].start + lputamen_bbox[0][0].stop))
    sl1 = int(0.5*(lputamen_bbox[0][1].start + lputamen_bbox[0][1].stop))
    sl2 = int(0.5*(lputamen_bbox[0][2].start + lputamen_bbox[0][2].stop))

    mr_imshow_kwargs  = {'vmin':0, 'vmax':np.percentile(mr,99.99), 'cmap':py.cm.Greys_r}
    pet_imshow_kwargs = {'vmin':0, 'vmax':np.percentile(cnn_bow[aparc>0],99.99)}
    vi = ThreeAxisViewer([lps_flip(mr),lps_flip(osem),lps_flip(bow),lps_flip(cnn_bow)], 
                          sl_x = sl0, sl_y = sl1, sl_z = sl2, ls = '', rowlabels = ['T1 MR','OSEM','BOW','$BOW_{CNN}$'],
                          imshow_kwargs = [mr_imshow_kwargs] + 3*[pet_imshow_kwargs])
    vi.fig.savefig(os.path.splitext(prediction_file)[0] + '.png')

    py.close(vi.fig)
  else:
    print('reading : ', df_file)
    df = pd.read_csv(df_file)
 
  df['tracer'] = tracer 
  reg_results = reg_results.append([df], ignore_index = True)
 
 
#---------------------------------------------------------------------------------
# filter background ROIs
reg_results = reg_results.loc[(reg_results['region'] != 'other') & 
                              (reg_results['region'] != 'background')]

#---------------------------------------------------------------------------------
# make plots

order = ['frontal cortex','temporal cortex','occipital cortex','parietal cortex',
          'hippocampus','cingulate cortex','thalamus','basal ganglia',
         'cerebellum','white matter','ventricle']

fp = dict(marker = 'o', markerfacecolor = '0.3', markeredgewidth = 0, markersize = 2.5) 

fig, ax = py.subplots(2,1, figsize = (12,6), sharex = True)
bplot1 = sns.boxplot(x='region',  y ='rc_mean', data = reg_results, ax = ax[0], hue = 'tracer', flierprops = fp,
                     order = order)
bplot2 = sns.boxplot(x='region',  y ='ssim',    data = reg_results, ax = ax[1], hue = 'tracer', flierprops = fp,
                     order = order)

# make better legend
for plot in [bplot1, bplot2]:
  handles, labels = plot.axes.get_legend_handles_labels()
  plot.legend().remove()

bplot1.legend(handles, labels, ncol=len(tracer), loc='upper center')
bplot2.legend(handles, labels, ncol=len(tracer), loc='lower right')

ax[0].set_ylabel('RC_mean')
ax[1].set_ylabel('SSIM_mean')

for axx in ax: 
  axx.set_xticklabels(axx.get_xticklabels(),rotation=15)
  axx.grid(ls = ':')
fig.tight_layout()
fig.savefig(os.path.join('figs', f'regions_{model_name}_{tracer}.pdf'))
fig.show()

fig2, ax2 = py.subplots(2,1, figsize = (12,6), sharex = True)
sns.boxplot(x='subject',  y ='rc_mean', data = reg_results, ax = ax2[0], hue = 'tracer', flierprops = fp)
sns.boxplot(x='subject',  y ='ssim',    data = reg_results, ax = ax2[1], hue = 'tracer', flierprops = fp)
for axx in ax2: 
  axx.set_xticklabels(axx.get_xticklabels(),rotation=90)
  axx.grid(ls = ':')
fig2.tight_layout()
fig2.savefig(os.path.join('figs', f'subjects_{model_name}_{tracer}.pdf'))
fig2.show()

# make the data tables
sum_data = pd.DataFrame()
tmp = pd.DataFrame({('nvox' + ' ' + pd.DataFrame.mean.__name__): reg_results.groupby(['tracer','region']).apply(pd.DataFrame.mean)['nvox'].astype(int)})
sum_data = pd.concat([sum_data, tmp], axis = 1)
for metric in ['rc_mean','ssim']:
  for stat in [pd.DataFrame.mean, pd.DataFrame.std, pd.DataFrame.min, pd.DataFrame.max]:
    tmp = pd.DataFrame({(metric + ' ' + stat.__name__): reg_results.groupby(['tracer','region']).apply(stat)[metric]})
    sum_data = pd.concat([sum_data, tmp], axis = 1)

sum_data.to_latex(os.path.join('figs',f'{model_name}_{tracer}.tex'), float_format = '{:,.3f}'.format) 
