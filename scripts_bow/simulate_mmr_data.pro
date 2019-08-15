pixelsize  = 1.
planesep   = 1.
FOV_mm     = 256
nrcols     = 256
nrrows     = 256
nrplanes   = ceil(127*2.03125/planesep)
nclip      = 100
nrdet      = 344 - 2*nclip

fwhm0_data       = [3.445,0] 
volumefwhm_data  = [0,3.35]

fwhm0_recon       = [3.445,0]
volumefwhm_recon  = [0,3.35]

small      = 1e-7

tmp = strsplit(nistring(fwhm0_recon[0]),'.',/extract)
psf_str = tmp[0] + '_' + strmid(tmp[1],0,1)
;----------------------------------------------------------------------------------------------------

subjects = (file_basename(file_search('../data/training_data/brainweb/raw/subject??_t1w_p4.mnc.gz'))).Map(lambda(x:strmid(x,0,9)))

FOREACH subject, subjects DO BEGIN
  print, subject

  subject_num = long(strmid(subject,7,2))
 
  FOR i = 0, 2 DO BEGIN
    init_seed = 1000*subject_num + i
    seed      = init_seed

    gm_contrast   =   5*(randomu(seed,1))[0] + 0.5
    wm_contrast   =   2*(randomu(seed,1))[0] + 0.5
    csf_contrast  = 0.2*(randomu(seed,1))[0] + 0.05
    skin_contrast =     (randomu(seed,1))[0] + 0.1
    fat_contrast  = 0.5*(randomu(seed,1))[0] + 0.1
    bone_contrast = 0.5*(randomu(seed,1))[0] + 0.1
    count_exp     =   2*(randomu(seed,1))[0] + 7.5
    counts        = 10^count_exp
 
    bw = Python.Import('brainweb')
    data = bw.brainweb(subject       =  'subject04',  $  
                       gm_contrast   =  gm_contrast,  $ 
                       wm_contrast   =  wm_contrast,  $ 
                       csf_contrast  = csf_contrast,  $ 
                       skin_contrast = skin_contrast, $
                       fat_contrast  = fat_contrast,  $ 
                       bone_contrast = bone_contrast)
    
    t1  = fltarr(nrcols,nrrows,nrplanes)
    pet = fltarr(nrcols,nrrows,nrplanes)
    
    
    start_plane = (nrplanes - (data.dim)[2])/2
    
    t1[*,*,start_plane:(start_plane +  (data.dim)[2] - 1)]  =  data[*,*,*,0]
    pet[*,*,start_plane:(start_plane + (data.dim)[2] - 1)] =  data[*,*,*,1]
 
    ;-----------------------------
    attn_img = 0.01*float(t1 GT 0.5)
    proj_mu = NIdef_proj(/pet3d_mmr, /siemensdefault,                        $
                                     child             = 1,                  $
                                     relreconsize      = pixelsize / 2.0445, $
                                     relreconplanesize = planesep  / 2.03125,$
                                     sumdet            = 1,                  $
                                     nrcols            = nrcols,             $
                                     nrrows            = nrcols,             $
                                     nrplanes          = nrplanes,           $
                                     nrdet             = nrdet,              $
                                     /distancedriven)
    
    niproj, attn_img, attn_fwd, proj = proj_mu
    attn_sino = exp(-attn_fwd)
    niproj_free, proj_mu
    
    sens_sino = fltarr(attn_sino.dim) + 1
    
    ;-----------------------------
    ; simulate the data
    proj_data = NIdef_proj(/pet3d_mmr, /siemensdefault,                             $
                           child             = 1,                                   $
                           relreconsize      = pixelsize / 2.0445,                  $
                           relreconplanesize = planesep  / 2.03125,                 $
                           sumdet            = 1,                                   $
                           fwhm0             = fwhm0_data/[2.0445,2.03125],         $
                           volumefwhm        = volumefwhm_data/[pixelsize,planesep],$
                           nrcols            = nrcols,                              $
                           nrrows            = nrcols,                              $
                           nrplanes          = nrplanes,                            $
                           nrdet             = nrdet,                               $
                           /distancedriven)
    
    niproj, pet, pet_fwd, proj = proj_data, attenuation = attn_sino
    niproj_free, proj_data
    
    emis_sino   = sens_sino * pet_fwd
    contam_sino = 0.4*niconvolgauss(emis_sino, fwhm = 25) + 0.01*max(emis_sino)
    
    emis_sino += contam_sino
   
    count_factor = counts / total(emis_sino)
    emis_sino   *= count_factor
    contam_sino *= count_factor
    sens_sino   *= count_factor
    
    emis_sino = nipoisson(seed, emis_sino)  
   
 
    ;-----------------------------
    ; do the recon
    proj_recon = NIdef_proj(/pet3d_mmr, /siemensdefault,                              $
                            child             = 1,                                    $
                            relreconsize      = pixelsize / 2.0445,                   $
                            relreconplanesize = planesep  / 2.03125,                  $
                            sumdet            = 1,                                    $
                            fwhm0             = fwhm0_recon/[2.0445,2.03125],         $
                            volumefwhm        = volumefwhm_recon/[pixelsize,planesep],$
                            nrcols            = nrcols,                               $
                            nrrows            = nrcols,                               $
                            nrplanes          = nrplanes,                             $
                            nrdet             = nrdet,                                $
                            /distancedriven)
    

    osem   = nimaposem(emis_sino,                             $
                       attenuation = attn_sino,               $
                       sensitivity = sens_sino,               $
                       contamsino  = contam_sino,             $  
                       projd       = proj_recon,              $
                       nriter      = 3,                       $
                       nrsubsets   = 21,                      $
                       /norm_per, /printgradinfo, /showsub, small = small)
     
    niproj_free, proj_recon

    ok = niwrite_nii(t1,   filepath('t1.nii',$ 
                                    root = '../data/training_data/brainweb', subdir = subject),$
                     orientation = 'LPS')
    ok = niwrite_nii(pet,  filepath('pet_' + nistring(init_seed) + '.nii',$ 
                                    root = '../data/training_data/brainweb', subdir = subject),$
                     orientation = 'LPS')
    ok = niwrite_nii(osem, filepath('osem_psf_' + psf_str + '_' + nistring(init_seed) + '.nii',$ 
                                    root = '../data/training_data/brainweb', subdir = subject),$
                     orientation = 'LPS')

  ENDFOR  
ENDFOREACH
END
