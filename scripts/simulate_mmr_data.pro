; before running this script the LD_LIBRARY_PATH has to contain the path to the correct python lib!
; to do so execute (before starting IDL): 
; (1) conda activate ZZZ  (where ZZZ is your desired conda env)
; (2) export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

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

small = 1e-7

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

    odir = filepath(subject, root = '../data/training_data/brainweb')

    IF file_test(odir) EQ 0 THEN file_mkdir, odir

    gm_contrast   =   5*(randomu(seed,1))[0] + 0.5
    wm_contrast   =   2*(randomu(seed,1))[0] + 0.5
    csf_contrast  = 0.2*(randomu(seed,1))[0] + 0.05
    skin_contrast =     (randomu(seed,1))[0] + 0.1
    fat_contrast  = 0.5*(randomu(seed,1))[0] + 0.1
    bone_contrast = 0.5*(randomu(seed,1))[0] + 0.1

    bw = Python.Import('brainweb')
    data = bw.brainweb(subject       = subject,       $  
                       gm_contrast   = gm_contrast,   $ 
                       wm_contrast   = wm_contrast,   $ 
                       csf_contrast  = csf_contrast,  $ 
                       skin_contrast = skin_contrast, $
                       fat_contrast  = fat_contrast,  $ 
                       bone_contrast = bone_contrast)

    t1  = fltarr(nrcols,nrrows,nrplanes)
    pet = fltarr(nrcols,nrrows,nrplanes)
    
    
    start_col = (nrcols - (data.dim)[0])/2
    start_row = (nrrows - (data.dim)[1])/2
    start_plane = (nrplanes - (data.dim)[2])/2
    
    t1[start_col:(start_col + (data.dim)[0] - 1), start_row:(start_row + (data.dim)[1] - 1), start_plane:(start_plane + (data.dim)[2] - 1)] = data[*,*,*,0]
    pet[start_col:(start_col + (data.dim)[0] - 1), start_row:(start_row + (data.dim)[1] - 1), start_plane:(start_plane + (data.dim)[2] - 1)] = data[*,*,*,1]

    attn_img = pixelsize*0.01*float(t1 GT 0.1*max(t1))


    ok = niwrite_nii(t1, filepath('t1.nii', root = odir), orientation = 'LPS')
    ok = niwrite_nii(attn_img, filepath('mu.nii', root = odir), orientation = 'LPS')
    ok = niwrite_nii(pet, filepath('true_pet_' + nistring(init_seed) + '.nii', root = odir),$
                     orientation = 'LPS')

    ;-----------------------------
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
    
    FOREACH counts, [0,1e7,5e8] DO BEGIN 
      emis_sino = pet_fwd
      contam_sino = 0.4*niconvolgauss(pet_fwd, fwhm = 25) + 0.01*max(pet_fwd)
      emis_sino += contam_sino

      IF counts GT 0 THEN BEGIN
        count_factor = counts / total(emis_sino)
        emis_sino   *= count_factor
        contam_sino *= count_factor
        sens_sino = fltarr(attn_sino.dim) + count_factor
        
        emis_sino = nipoisson(seed, emis_sino)  
      ENDIF ELSE BEGIN
        sens_sino = fltarr(attn_sino.dim) + 1
      ENDELSE
   
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

      ok = niwrite_nii(osem, filepath('osem_psf_' + psf_str + '_' + nistring(init_seed) + '_counts_' + nistring(counts, format = '(E0.1)') + '.nii', root = odir), orientation = 'LPS')

    ENDFOREACH
  ENDFOR  
ENDFOREACH
END
