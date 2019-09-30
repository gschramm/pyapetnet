import os
import time
import shutil
import logging
import argparse

import pyapetnet
from   pyapetnet.predictors import predict
from   pyapetnet.read_dicom import dicom_search

from fnmatch import fnmatch
from logging.handlers import TimedRotatingFileHandler

parser = argparse.ArgumentParser(description = 'Bowsher CNN queue',
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dir', default = '/data/dicom/',     
                    help = 'Incoming dicom dir')
parser.add_argument('--process_dir', default = '/data/temp/', 
                    help = 'Temporary processing dir')
parser.add_argument('--pacs_dir', default = '/data/ready/', 
                    help = 'Spool directory for PACS import')
parser.add_argument('--archiv_dir', default = '/data/archiv/', 
                    help = 'archiving directory')
parser.add_argument('--t_wait', type = int, default = 60, 
                    help = 'waiting time between processings (s)')
parser.add_argument('--log_file', default = '/data/log/bowcnn_queue.log', 
                    help = 'file used for logging')
parser.add_argument('--petac_pattern', default = '*_agr_ac*', 
                    help = 'pattern used to find AC PET dicom series')
parser.add_argument('--mprage_pattern', default = '*mprage*_agr*', 
                    help = 'pattern used to find MPRAGE dicom series')
parser.add_argument('--model_name', default = '190904_osem_nopsf_bet_10_bs_52_ps_29.h5', 
                    help = 'basename of the CNN model')
parser.add_argument('--model_dir', default = '/data/models', 
                    help = 'basename of the CNN model')

args = parser.parse_args()

#---- set up the logger
logger = logging.getLogger('simple')
logger.setLevel(logging.DEBUG)

handler = TimedRotatingFileHandler(args.log_file, when = 'W0', backupCount = 52)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s-%(process)d-%(levelname)s : %(message)s'))

logger.addHandler(handler)
#----

t_wait         = args.t_wait
incoming_dir   = args.in_dir
processing_dir = args.process_dir
pacs_spool_dir = args.pacs_dir
archiv_dir     = args.archiv_dir
petac_pattern  = args.petac_pattern
mprage_pattern = args.mprage_pattern

model_name     = args.model_name 
model_dir      = args.model_dir 

keepRunning = True

while keepRunning:
  input_dirs = next(os.walk(incoming_dir))[1]

  time.sleep(t_wait)
  
  if len(input_dirs) == 0:
    logger.info('waiting for incoming data')
  else:
    logger.info(f'')
    logger.info(f'======================================================')
    logger.info(f'processing: {os.path.join(incoming_dir,input_dirs[0])}')

    # move data away from the incoming dir to a temp dir for processing
    pdir = os.path.join(processing_dir,input_dirs[0])
    shutil.move(os.path.join(incoming_dir,input_dirs[0]), pdir)

    logger.info(f'moved {os.path.join(incoming_dir,input_dirs[0])} to {pdir}')

    # find the correct PET and MR dicom files
    pet_dcm_files = None
    mr_dcm_files  = None

    dcm_info = dicom_search(os.path.join(pdir,'*'))
    if len(dcm_info) >= 2:
      seriesDescriptions = [x["SeriesDescription"] for x in dcm_info]

      pet_seriesDesc = [x for x in seriesDescriptions if fnmatch(x.lower(),petac_pattern)]
      if len(pet_seriesDesc) > 0: 
        pet_seriesDesc = pet_seriesDesc[0]
        pet_dcm_files  = dcm_info[seriesDescriptions.index(pet_seriesDesc)]["files"]
        logger.info(f'found PET dicom series {pet_seriesDesc}')
      else:
        logger.error(f'Cannot find PET AC dicom series')

      mr_seriesDesc = [x for x in seriesDescriptions if fnmatch(x.lower(),mprage_pattern)]
      if len(mr_seriesDesc) > 0: 
        mr_seriesDesc = mr_seriesDesc[0]
        mr_dcm_files  = dcm_info[seriesDescriptions.index(mr_seriesDesc)]["files"]
        logger.info(f'found MR dicom series {mr_seriesDesc}')
      else:
        logger.error(f'Cannot find MPRAGE dicom series')

    if (pet_dcm_files is not None) and (mr_dcm_files is not None):

      odir = os.path.join(processing_dir, 'cnnbow_' + os.path.basename(pdir))

      predict(mr_input      = mr_dcm_files, 
              pet_input     = pet_dcm_files,            
              input_format  = 'dicom',                                     
              odir          = odir,
              model_name    = model_name,           
              model_dir     = model_dir,  
              perc          = 99.99,                                       
              patchsize     = (128,128,128),                               
              verbose       = True,                                        
              clip_neg      = True,                                        
              coreg         = True,                                        
              affine        = None,                                        
              crop_mr       = False,
              mr_ps_fwhm_mm = 1.5,                                       
              debug_mode    = True)  

      # move dicom output to the PACS import spool directory
      if os.path.exists(odir):
        shutil.move(odir, pacs_spool_dir)
        logger.info(f'moved {odir} to {pacs_spool_dir}')

      # move output niftis to archiv dir
      for fname in [odir + x for x in ['.nii','_debug_mr.nii','_debug_pet.nii','.png']]:
        if os.path.exists(fname):
          shutil.move(fname, archiv_dir)
          logger.info(f'moved {fname} to {archiv_dir}')
 
    # move input data from processing dir to archiv 
    if not os.path.exists(os.path.join(archiv_dir,os.path.basename(pdir))):
      if os.path.exists(pdir):
        shutil.move(pdir, archiv_dir)
        logger.info(f'moved {pdir} to {archiv_dir}')
