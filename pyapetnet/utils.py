import os
import numpy as np
import nibabel as nib

from pymirc.fileio import write_3d_static_dicom


#----------------------------------------------------------------------------------------------
def cross_product_matrix(a):
    # cross product matrix of a vector a
    # useful to contruct an affine matrix for a

    ndim = a.shape[0]
    cpm = np.zeros((ndim, ndim))

    for i in range(ndim):
        uv = np.zeros(ndim)
        uv[i] = 1
        cpm[:, i] = np.cross(a, uv)

    return cpm


#----------------------------------------------------------------------------------------------
def rotation_matrix(uv, theta):
    # general rotation matrix for rotation around unit vector uv

    ndim = uv.shape[0]

    tmp = np.cos(theta) * np.eye(ndim) + np.sin(theta) * cross_product_matrix(
        uv) + (1 - np.cos(theta)) * np.outer(uv, uv)

    R = np.zeros((ndim + 1, ndim + 1))
    R[:-1, :-1] = tmp
    R[-1, -1] = 1

    return R


#----------------------------------------------------------------------------------------------
def affine_center_rotation(uv, theta, uv_origin=None, offset=None):
    # affine trasnformation for rotation around unit vector uv through the center followd by shift

    ndim = uv.shape[0]

    # set up affine to transform origin of rotation axis into image center
    T = np.eye(ndim + 1, ndim + 1)
    if uv_origin is not None: T[:-1, -1] -= uv_origin

    R = rotation_matrix(uv, theta)

    aff = np.linalg.inv(T) @ (R @ T)

    if offset is not None: aff[:-1, -1] += offset

    return aff


#----------------------------------------------------------------------
def load_nii_in_ras(fname):
    """ function that loads nifti file and returns the volume and affine in 
        RAS orientation
    """
    nii = nib.load(fname)
    nii = nib.as_closest_canonical(nii)
    vol = nii.get_fdata()

    return vol, nii.affine


#----------------------------------------------------------------------
def flip_ras_lps(vol, affine):
    """ flip a volume and its affine from RAS to LPS, or from LPS to RAS
    """
    vol_flipped = np.flip(vol, (0, 1))
    affine_flipped = affine.copy()
    affine_flipped[0, -1] = (-1 *
                             affine @ np.array([vol.shape[0] - 1, 0, 0, 1]))[0]
    affine_flipped[1, -1] = (-1 *
                             affine @ np.array([0, vol.shape[1] - 1, 0, 1]))[1]

    return vol_flipped, affine_flipped


def create_demo_dcm_data(dcm_dir):
    """ convert the test data set from nifti to dicom
    """
    pet_fname = os.path.join(os.path.dirname(__file__), 'data',
                             'brainweb_06_osem.nii')
    mr_fname = os.path.join(os.path.dirname(__file__), 'data',
                            'brainweb_06_t1.nii')

    pet, pet_affine = flip_ras_lps(*load_nii_in_ras(pet_fname))
    mr, mr_affine = flip_ras_lps(*load_nii_in_ras(mr_fname))

    os.mkdir(dcm_dir)
    write_3d_static_dicom(pet,
                          os.path.join(dcm_dir, 'PT'),
                          pet_affine,
                          modality='PT')
    write_3d_static_dicom(mr,
                          os.path.join(dcm_dir, 'MR'),
                          mr_affine,
                          modality='MR')




def pet_dcm_keys_to_copy():
    """  return a list of usefule PET dicom tags to copy """
    return [
        'AcquisitionDate', 'AcquisitionTime', 'ActualFrameDuration',
        'AccessionNumber', 'DecayCorrection', 'DecayCorrectionDateTime',
        'DecayFactor', 'DoseCalibrationFactor', 'FrameOfReferenceUID',
        'FrameReferenceTime', 'InstitutionName', 'ManufacturerModelName',
        'OtherPatientIDs', 'PatientAge', 'PatientBirthDate', 'PatientID',
        'PatientName', 'PatientPosition', 'PatientSex', 'PatientWeight',
        'ProtocolName', 'RadiopharmaceuticalInformationSequence',
        'RescaleType', 'SeriesDate', 'SeriesTime', 'StudyDate',
        'StudyDescription', 'StudyID', 'StudyInstanceUID', 'StudyTime', 'Units'
    ]
