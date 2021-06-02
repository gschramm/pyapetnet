import nibabel as nib
import numpy   as np
import os

from glob import glob

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage         import affine_transform

def brainweb(brainweb_raw_dir = os.path.join('..','data','training_data','brainweb','raw'),
             subject          = 'subject54',
             gm_contrast      = 4,
             wm_contrast      = 1,
             csf_contrast     = 0.05,
             skin_contrast    = 0.5,
             fat_contrast     = 0.25,
             bone_contrast    = 0.1,
             blood_contrast   = 0.8):

  dmodel_path = os.path.join(brainweb_raw_dir, subject + '_crisp_v.mnc.gz')
  t1_path     = os.path.join(brainweb_raw_dir, subject + '_t1w_p4.mnc.gz')
  
  # the simulated t1 has different voxel size and FOV)
  dmodel_affine = nib.load(dmodel_path).affine.copy()
  t1_affine     = nib.load(t1_path).affine.copy()
  
  dmodel_voxsize = np.sqrt((dmodel_affine**2).sum(0))[:-1]
  t1_voxsize     = np.sqrt((t1_affine**2).sum(0))[:-1]
 
  dmodel = nib.load(dmodel_path).get_data()
  t1     = nib.load(t1_path).get_data()

  # create low frequent variation in GM
  v  = gaussian_filter(np.random.rand(*dmodel.shape), 30)
  v *= (0.1/v.std())
  v += (1 - v.mean())
 
  pet_gt = (gm_contrast*v*(dmodel == 2) + 
            wm_contrast*(dmodel == 3) + 
            skin_contrast*(dmodel == 5) + 
            skin_contrast*(dmodel == 6) + 
            fat_contrast*(dmodel == 4) + 
            bone_contrast*(dmodel == 7) + 
            bone_contrast*(dmodel == 11) + 
            blood_contrast*(dmodel == 8) + 
            csf_contrast*(dmodel == 1))

  # the dmodel has half the voxel size of the T1
  # we average neighboring columns, rows and planes

  pet_gt = 0.5*(pet_gt[::2,:,:] + pet_gt[1::2,:,:])
  pet_gt = 0.5*(pet_gt[:,::2,:] + pet_gt[:,1::2,:])
  pet_gt = 0.5*(pet_gt[:,:,::2] + pet_gt[:,:,1::2])

  # the offset of the T1 is different, we crop it
  aff     = np.linalg.inv(dmodel_affine) @ t1_affine
  offset  = -aff[:-1,-1].astype(int) 
  hoffset = (offset//2)

  t1_crop = t1[:,(offset[1]-hoffset[1]):(-hoffset[1]),(offset[2]-hoffset[2]):(-hoffset[2])]

  return np.array([np.flip(t1_crop,1)/t1_crop.max(),np.flip(pet_gt,1)])


#--------------------------------------------------------------------------------------------------

def brainweb2d(brainweb_raw_dir = os.path.join('..','data','training_data','brainweb','raw'),
               subject          = 'subject54',
               gm_contrast      = (1,4),
               wm_contrast      = (1,1.5),
               csf_contrast     = (0,0),
               skin_contrast    = (0.2,1),
               fat_contrast     = (0.2,1),
               bone_contrast    = (0.1,0.2),
               seed             = 0):

  np.random.seed(seed)

  dmodel_path = os.path.join(brainweb_raw_dir, subject + '_crisp_v.mnc.gz')
  gm_path     = os.path.join(brainweb_raw_dir, subject + '_gm_v.mnc.gz')
  wm_path     = os.path.join(brainweb_raw_dir, subject + '_wm_v.mnc.gz')
  t1_path     = os.path.join(brainweb_raw_dir, subject + '_t1w_p4.mnc.gz')
  
  # the simulated t1 has different voxel size and FOV)
  dmodel_affine = nib.load(dmodel_path).affine.copy()
  t1_affine     = nib.load(t1_path).affine.copy()
  
  dmodel_voxsize = np.sqrt((dmodel_affine**2).sum(0))[:-1]
  t1_voxsize     = np.sqrt((t1_affine**2).sum(0))[:-1]
  
  dmodel = nib.load(dmodel_path).get_data()
  gm     = nib.load(gm_path).get_data()
  wm     = nib.load(wm_path).get_data()
  
  t1     = nib.load(t1_path).get_data()

  # the dmodel has half the voxel size of the T1
  # we average neighboring columns, rows and planes

  max_gm_slice  = np.argmax(gm.sum(-1).sum(-1))
  start_slice   = max(0,max_gm_slice-60)
  end_slice     = min(dmodel.shape[0], max_gm_slice + 130)

  pet_gt = np.zeros(dmodel.shape)

  for sl in range(start_slice,end_slice):
  
    gmc = (gm_contrast[1]   - gm_contrast[0])*np.random.rand()   + gm_contrast[0]
    wmc = (wm_contrast[1]   - wm_contrast[0])*np.random.rand()   + wm_contrast[0]
    sc  = (skin_contrast[1] - skin_contrast[0])*np.random.rand() + skin_contrast[0]
    fc  = (fat_contrast[1]  - fat_contrast[0])*np.random.rand()  + fat_contrast[0]
    bc  = (bone_contrast[1] - bone_contrast[0])*np.random.rand() + bone_contrast[0]
    cc  = (csf_contrast[1]  - csf_contrast[0])*np.random.rand()  + csf_contrast[0]

    flip_contrast = np.random.rand()
    if flip_contrast > 0.7:
      gmc, wmc = wmc, gmc

    pet_gt[sl,...] = (gmc*gm[sl,...] + 
                      wmc*wm[sl,...] + 
                      sc*(dmodel[sl,...] == 5) + 
                      sc*(dmodel[sl,...] == 6) + 
                      fc*(dmodel[sl,...] == 4) + 
                      bc*(dmodel[sl,...] == 7) + 
                      bc*(dmodel[sl,...] == 11) + 
                      cc*(dmodel[sl,...] == 1))


  # the dmodel has half the voxel size of the T1
  # we average neighboring columns, rows and planes

  pet_gt = pet_gt[::2,:,:]
  pet_gt = pet_gt[:,::2,:]
  pet_gt = pet_gt[:,:,::2]

  dmodel = dmodel[::2,:,:]
  dmodel = dmodel[:,::2,:]
  dmodel = dmodel[:,:,::2]

  # the offset of the T1 is different, we crop it
  aff     = np.linalg.inv(dmodel_affine) @ t1_affine
  offset  = -aff[:-1,-1].astype(int) 
  hoffset = (offset//2)

  t1_crop = t1[:,(offset[1]-hoffset[1]):(-hoffset[1]),(offset[2]-hoffset[2]):(-hoffset[2])]


  # crop images to start and end slice
  pet_gt  = pet_gt[(start_slice//2 + 1):(end_slice//2),:,:]
  t1_crop = t1_crop[(start_slice//2 + 1):(end_slice//2),:,:]
  dmodel  = dmodel[(start_slice//2 + 1):(end_slice//2),:,:]

  # pad images to 256x256
  p1 = 256 - pet_gt.shape[1]
  p2 = 256 - pet_gt.shape[2]

  pet_gt  = np.pad(pet_gt,  ((0,0),(p1//2, p1 - p1//2),(p2//2, p2 - p2//2)), mode = 'constant')
  t1_crop = np.pad(t1_crop, ((0,0),(p1//2, p1 - p1//2),(p2//2, p2 - p2//2)), mode = 'constant')
  dmodel  = np.pad(dmodel, ((0,0),(p1//2, p1 - p1//2),(p2//2, p2 - p2//2)), mode = 'constant')

  return np.array([np.flip(t1_crop,1)/t1_crop.max(), np.flip(pet_gt,1), np.flip(dmodel,1)], dtype = np.float32)


