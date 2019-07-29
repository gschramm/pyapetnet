import os
import datetime

import warnings
import numpy as np

import pydicom as dicom
from pydicom.dataset import Dataset, FileDataset

from   time import time

def write_dicom_slice(pixel_array, # 2D array in LP orientation
                      filename                               = None,
                      outputdir                              = 'mydcmdir',
                      suffix                                 = '.dcm',
                      modality                               = 'PT',
                      SecondaryCaptureDeviceManufctur        = 'KUL',
                      uid_base                               = '1.2.826.0.1.3680043.9.7147.', # UID root for Georg Schramm
                      PatientName                            = 'Test^Patient',
                      PatientID                              = '08150815',
                      AccessionNumber                        = '08150815',
                      StudyDescription                       = 'test study',
                      SeriesDescription                      = 'test series',
                      PixelSpacing                           = ['1','1'],
                      SliceThickness                         = '1',
                      ImagePositionPatient                   = ['0','0','0'],
                      ImageOrientationPatient                = ['1','0','0','0','1','0'],
                      CorrectedImage                         = ['DECY','ATTN','SCAT','DTIM','LIN','CLN'],
                      ImageType                              = 'STATIC',
                      RescaleSlope                           = None,
                      RescaleIntercept                       = None,
                      StudyInstanceUID                       = None,
                      SeriesInstanceUID                      = None,
                      SOPInstanceUID                         = None,
                      FrameOfReferenceUID                    = None,
                      RadiopharmaceuticalInformationSequence = None,
                      PatientGantryRelationshipCodeSequence  = None, 
                      verbose                                = False,
                      **kwargs):
  """write a 2D PET dicom slice

  Parameters
  ---------

  pixel_array : 2d numpy array 
    array that contains the image values


  filename : str, optional  
    name of the output dicom file (default: None -> automatically generated)
  
  outputdir : string, optional 
    output directory fir dicom file (default: mydcmdir)

  suffix : string, optional 
    suffix for dicom file (default '.dcm')

  SecondaryCaptureDeviceManufctur       --|  
  uid_base                                | 
  PatientName                             | 
  PatientID                               | 
  AccessionNumber                         | 
  StudyDescription                        | 
  SeriesDescription                       | 
  PixelSpacing                            | 
  SliceThickness                          | 
  ImagePositionPatient                    | 
  ImageOrientationPatient                 | 
  CorrectedImage                          | ... dicom tags that should be present in a minimal
  ImageType                               |     dicom header
  RescaleSlope                            |     see function definition for default values
  RescaleIntercept                        |     default None means that they are creacted automatically
  StudyInstanceUID                        | 
  SeriesInstanceUID                       | 
  SOPInstanceUID                          | 
  FrameOfReferenceUID                     | 
  RadiopharmaceuticalInformationSequence  | 
  PatientGantryRelationshipCodeSequence --| 

  **kwargs : additional tags from the standard dicom dictionary to write
             the following tags could be useful:
  StudyDate                              
  StudyTime                              
  SeriesDate                             
  SeriesTime                             
  AcquisitionDate                        
  AcquisitionTime                        
  PatientBirthDate                       
  PatientSex                             
  PatientAge                             
  PatientSize                            
  PatientWeight                          
  ActualFrameDuration                    
  PatientPosition                        
  DecayCorrectionDateTime                
  ImagesInAcquisition                    
  SliceLocation                          
  NumberOfSlices                         
  Units                                  
  DecayCorrection                        
  ReconstructionMethod                   
  FrameReferenceTime                     
  DecayFactor                             
  DoseCalibrationFactor                  
  ImageIndex                             

  Returns
  -------
  str
    containing the ouput file name  

  """
  # create output dir if it does not exist
  if not os.path.exists(outputdir): os.mkdir(outputdir)

  # Populate required values for file meta information
  file_meta = Dataset()

  if   modality == 'PT': file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.128'
  elif modality == 'NM': file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.20'
  elif modality == 'CT': file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.2'
  elif modality == 'MR': file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.4'
  else:                  file_meta.MediaStorageSOPClassUID    = '1.2.840.10008.5.1.4.1.1.4'

  # the MediaStorageSOPInstanceUID sould be the same as SOPInstanceUID
  # however it is stored in the meta information header to have faster access
  if SOPInstanceUID == None: SOPInstanceUID = dicom.uid.generate_uid(uid_base)
  file_meta.MediaStorageSOPInstanceUID = SOPInstanceUID
  file_meta.ImplementationClassUID     = uid_base + '1.1.1'
 
  filename = modality + '.' + SOPInstanceUID + suffix
 
  # Create the FileDataset instance (initially no data elements, but file_meta
  # supplied)
  ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
 
  # Add the data elements -- not trying to set all required here. Check DICOM
  # standard
  ds.PatientName = PatientName
  ds.PatientID   = PatientID

  ds.Modality          = modality
  ds.StudyDescription  = StudyDescription
  ds.SeriesDescription = SeriesDescription

  if StudyInstanceUID    == None: StudyInstanceUID    = dicom.uid.generate_uid(uid_base) 
  ds.StudyInstanceUID    = StudyInstanceUID   

  if SeriesInstanceUID   == None: SeriesInstanceUID   = dicom.uid.generate_uid(uid_base)
  ds.SeriesInstanceUID   = SeriesInstanceUID  

  if FrameOfReferenceUID == None: FrameOfReferenceUID = dicom.uid.generate_uid(uid_base)
  ds.FrameOfReferenceUID = FrameOfReferenceUID

  ds.SOPInstanceUID      = SOPInstanceUID     
  ds.SOPClassUID         = file_meta.MediaStorageSOPClassUID

  ds.SecondaryCaptureDeviceManufctur = SecondaryCaptureDeviceManufctur

  ## These are the necessary imaging components of the FileDataset object.
  ds.SamplesPerPixel           = 1

  if modality == 'PT' or modality == 'NM': ds.PhotometricInterpretation = "MONOCHROME1"
  else:                                    ds.PhotometricInterpretation = "MONOCHROME2"

  ds.HighBit                   = 15
  ds.BitsStored                = 16
  ds.BitsAllocated             = 16

  # PixelRepresentation is 0 for uint16, 1 for int16

  if pixel_array.dtype == np.uint16:
    ds.PixelRepresentation = 0                
    ds.RescaleIntercept    = 0    
    ds.RescaleSlope        = 1    
  elif pixel_array.dtype == np.int16:
    ds.PixelRepresentation = 1                
    ds.RescaleIntercept    = 0    
    ds.RescaleSlope        = 1    
  else:
    ds.PixelRepresentation = 0                

    # rescale the input pixel array to uint16 if needed
    if RescaleIntercept == None: RescaleIntercept = pixel_array.min()
    if RescaleIntercept != 0:    pixel_array      = 1.0*pixel_array - RescaleIntercept
      
    if RescaleSlope == None: 
      if pixel_array.max() != 0: RescaleSlope = 1.0*pixel_array.max()/(2**16 - 1)
      else:                      RescaleSlope = 1.0
    if RescaleSlope != 1: pixel_array  = 1.0*pixel_array/RescaleSlope

    pixel_array = pixel_array.astype(np.uint16)

    ds.RescaleIntercept = RescaleIntercept    
    ds.RescaleSlope     = RescaleSlope

  # we have to transpose the column and row direction in the dicoms
  ds.PixelData      = pixel_array.transpose().tobytes()
  ds.Columns        = pixel_array.shape[0]
  ds.Rows           = pixel_array.shape[1]
 
  # the pixel spacing also has to be inverted (transposed array saved!) 
  ds.PixelSpacing   = PixelSpacing[::-1]
  ds.SliceThickness = SliceThickness

  # Set the transfer syntax
  ds.is_little_endian = True
  ds.is_implicit_VR   = True
  
  # Set creation date/time
  dt             = datetime.datetime.now()
  timeStr        = dt.strftime('%H%M%S.%f')  # long format with micro seconds
  ds.ContentDate = dt.strftime('%Y%m%d')
  ds.ContentTime = timeStr

  # voxel coordinate tags 
  ds.ImagePositionPatient    = ImagePositionPatient   
  ds.ImageOrientationPatient = ImageOrientationPatient

  # special NM tags
  if modality == 'PT' or modality == 'NM':
    ds.CorrectedImage = CorrectedImage  
    ds.ImageType      = ImageType       

    if RadiopharmaceuticalInformationSequence != None: 
      rpi = RadiopharmaceuticalInformationSequence
      # this is needed otherwise the dicoms cannot be read
      rpi.is_undefined_length = True
      rpi[0].RadionuclideCodeSequence.is_undefined_length = True
      
      ds.RadiopharmaceuticalInformationSequence = rpi

  # add all key word arguments to dicom structure
  for key, value in kwargs.items(): 
    if dicom.datadict.tag_for_name(key) != None: setattr(ds,key,value)
    else: warnings.warn(key + ' not in standard dicom dictionary -> will not be written')

  if verbose: print("Writing file", os.path.join(outputdir,filename))
  dicom.filewriter.write_file(os.path.join(outputdir,filename), ds, write_like_original = False)

  return os.path.join(outputdir,filename)

###########################################################################################

def write_3d_static_dicom(vol_lps, 
                          outputdir,
                          affine              = np.eye(4),
                          uid_base            = '1.2.826.0.1.3680043.9.7147.',
                          RescaleSlope        = None,
                          RescaleIntercept    = None,
                          StudyInstanceUID    = None, 
                          SeriesInstanceUID   = None,
                          FrameOfReferenceUID = None,
                          **kwargs):
  """write a 3d PET volume to 2D dicom files

  Parameters
  ----------

  vol_lps : 3d numpy array   
    a 3D array in LPS orientation containing the image

  outputdir : str, optional
    the output directory for the dicom files

  affine : 2d 4x4 numpy array, optional
    affine transformation mapping from voxel to LPS coordinates
    The voxel sizes, the direction vectors and the LPS origin
    are derived from it.

  uid_base : str, optional 
    base string for UID (default 1.2.826.0.1.3680043.9.7147)

  RescaleSlope : float, optional
    rescale Slope (default None -> maximum of image / (2**16 - 1)) 

  RescaleIntercept : float, optional
    resalce Intercept (default None -> 0) 

  StudyInstanceUID : str, optional
    dicom study instance UID (default None -> autom. created)

  SeriesInstanceUID : str, optional 
    dicom series instance UID (default None -> autom. created)   

  FrameOfReferenceUID : str, optional 
    dicom frame of reference UID (default None -> autom. created)       

  **kwargs : dict, optional
    passed to write_dicom_slice

  Returns
  -------
  list
    containing the file names of the written 2D dicom files

  Note
  ----
  This function is a wrapper around write_dicom_slice.
  """
  # get the voxel sizes, direction vectors and the origin from the affine in case given
  ux = affine[:-1,0]
  uy = affine[:-1,1]
  uz = affine[:-1,2]

  xvoxsize = np.sqrt((ux**2).sum()) 
  yvoxsize = np.sqrt((uy**2).sum()) 
  zvoxsize = np.sqrt((uz**2).sum()) 
  
  nx = ux / xvoxsize        
  ny = uy / yvoxsize        

  lps_origin = affine[:-1,-1]

  if RescaleSlope == None: RescaleSlope = (vol_lps.max() - vol_lps.min()) / (2**16 - 1)
  if RescaleIntercept == None: RescaleIntercept = vol_lps.min()

  # calculate the normalized direction vector in z direction
  nz = np.cross(nx,ny)

  if StudyInstanceUID    == None: StudyInstanceUID    = dicom.uid.generate_uid(uid_base) 
  if SeriesInstanceUID   == None: SeriesInstanceUID   = dicom.uid.generate_uid(uid_base)
  if FrameOfReferenceUID == None: FrameOfReferenceUID = dicom.uid.generate_uid(uid_base)

  numSlices = vol_lps.shape[2]

  fnames = []

  for i in range(vol_lps.shape[-1]):
    fnames.append(write_dicom_slice(vol_lps[:,:,i], 
                                    uid_base                = uid_base,
                                    ImagePositionPatient    = (lps_origin + i*zvoxsize*nz).astype('str').tolist(),
                                    ImageOrientationPatient = np.concatenate((nx,ny)).astype('str').tolist(),
                                    PixelSpacing            = [str(xvoxsize), str(yvoxsize)],
                                    SliceThickness          = str(zvoxsize),
                                    RescaleSlope            = RescaleSlope,
                                    RescaleIntercept        = RescaleIntercept,
                                    StudyInstanceUID        = StudyInstanceUID, 
                                    SeriesInstanceUID       = SeriesInstanceUID, 
                                    FrameOfReferenceUID     = FrameOfReferenceUID,
                                    outputdir               = outputdir,
                                    NumberOfSlices          = numSlices,
                                    **kwargs))

  return fnames

###################################################################################################

def write_4d_dicom(vol_lps, 
                   outputdir,
                   uid_base   = '1.2.826.0.1.3680043.9.7147.',
                   SeriesType = 'DYNAMIC',
                   **kwargs):

  """ write 4D volume to 2D dicom files

  Parameters
  ----------

  vol_lps : 4d numpy array
    a 4D array in LPST orientation containing the image.
    the timing axis has to be the left most axis.

  outputdir : str 
    the output directory for the dicom files
 
  uid_base : str, optional
    base string for UID (default 1.2.826.0.1.3680043.9.7147)

  **kwargs : dict
    passed to write_3d_static_dicom
    note: Every kwarg can be a list of length nframes or a single value.
          In the first case, each time frame gets a different value
          (e.g. useful for AcquisitionTime or ActualFrameDuration).
          In the second case, each time frame gets the same values
          (e.g. for PatientWeight or affine)        

  Returns
  -------
  list of lists
    containing the filenames of the written 2D dicom files
    each element containg the filenames of one frame

  Note
  ----
  This function is a wrapper around write_3d_static_dicom.
  """
  numFrames = vol_lps.shape[0]

  StudyInstanceUID    = dicom.uid.generate_uid(uid_base) 
  SeriesInstanceUID   = dicom.uid.generate_uid(uid_base)
  FrameOfReferenceUID = dicom.uid.generate_uid(uid_base)

  numSlices = vol_lps.shape[-1]

  fnames = []

  for i in range(numFrames):
    kw = {}
    for key, value in kwargs.items():
      if type(value) is list: kw[key] = value[i]
      else:                   kw[key] = value

    fnames.append(write_3d_static_dicom(vol_lps[i,...], 
                                        outputdir,
                                        uid_base                   = uid_base,
                                        TemporalPositionIdentifier = i + 1,
                                        NumberOfTemporalPositions  = numFrames,
                                        StudyInstanceUID           = StudyInstanceUID,   
                                        SeriesInstanceUID          = SeriesInstanceUID,  
                                        FrameOfReferenceUID        = FrameOfReferenceUID,
                                        SeriesType                 = SeriesType,
                                        **kw))

  return fnames
