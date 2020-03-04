; before running this script the LD_LIBRARY_PATH has to contain the path to the correct python lib!
; to do so execute (before starting IDL): 
; (1) conda activate ZZZ  (where ZZZ is your desired conda env)
; (2) export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

nrdet    = 256
nrangles = 224 

fwhm0_data  = 4.5
fwhm0_recon = 4.5

betbow = 5e0
counts = 1e7
small  = 1e-7

ofile = 'bow_2d_sim.h5'

;----------------------------------------------------------------------------------------------------

subjects = (file_basename(file_search('../data/training_data/brainweb/raw/subject??_t1w_p4.mnc.gz'))).Map(lambda(x:strmid(x,0,9)))

proj_mu    = nidef_proj(nrdet = nrdet, nrangles = nrangles, /parkul, pixelsizecm = 0.1)
proj_data  = nidef_proj(nrdet = nrdet, nrangles = nrangles, /parkul, pixelsizecm = 0.1, fwhm0 = fwhm0_data)
proj_recon = nidef_proj(nrdet = nrdet, nrangles = nrangles, /parkul, pixelsizecm = 0.1, fwhm0 = fwhm0_recon)

FOREACH subject, subjects DO BEGIN
  print, subject

  subject_num = long(strmid(subject,7,2))
 
  init_seed = subject_num*1000
  seed      = init_seed
 
  bw = Python.Import('brainweb')
  data = bw.brainweb2d(subject = subject, seed = seed)
  
  t1_3d     = data[*,*,*,0]
  pet_3d    = data[*,*,*,1]
  dmodel_3d = data[*,*,*,2]

  ;-----------------------------
  attn_img_3d = 0.01*float(dmodel_3d GT 0)
  binds = where(dmodel_3d EQ 11, /NULL)
  attn_img_3d[binds] = 0.015

  osem_3d = fltarr(pet_3d.dim)
  bow_3d  = fltarr(pet_3d.dim)

  FOR sl = 0, ((pet_3d.dim)[2] - 1) DO BEGIN
    pet      = pet_3d[*,*,sl]
    t1       = t1_3d[*,*,sl]
    attn_img = attn_img_3d[*,*,sl]

    attn_fwd = fltarr(nrdet,nrangles)
    niproj, attn_img, attn_fwd, proj = proj_mu
    attn_sino = exp(-attn_fwd)
    sens_sino = fltarr(attn_sino.dim) + 1

    ;-----------------------------
    ; simulate the data
    pet_fwd  = fltarr(nrdet,nrangles)
    niproj, pet, pet_fwd, proj = proj_data, attenuation = attn_sino

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

    osem_3d[*,*,sl] = nimaposem(emis_sino,     $
                     attenuation = attn_sino,  $
                     sensitivity = sens_sino,  $
                     contamsino  = contam_sino,$  
                     projd       = proj_recon, $
                     nriter      = 3,          $
                     nrsubsets   = 28,         $
                     /norm_per, /printgradinfo, /showsub, small = small)
 
    mask_bow = [[0,1,1,1,0],$
                [1,1,1,1,1],$
                [1,1,0,1,1],$
                [1,1,1,1,1],$
                [0,1,1,1,0]]
 
    prior = nidef_prior(asymbowsher = 1, markovtype = 'reldiff', markovweight = betbow,$ 
                        labelsimage = t1, neighborweights = mask_bow, markovparm = 0,$  
                        bowsher_nrneighbors = 4)

    bow_3d[*,*,sl]  = nimaposem(emis_sino,        $
                     attenuation  = attn_sino,    $
                     sensitivity  = sens_sino,    $
                     contamsino   = contam_sino,  $  
                     projd        = proj_recon,   $
                     nriter       = 20,           $
                     nrsubsets    = 28,           $
                     priordescrip = prior,        $
                     recon        = osem_3d[*,*,sl], $
                     /norm_per, /printgradinfo, /showsub, small = small)
  ENDFOR


  niwrite_hdf5, t1_3d,   ofile, 's_' + nistring(subject_num), 't1_3d'
  niwrite_hdf5, osem_3d, ofile, 's_' + nistring(subject_num), 'osem_3d'
  niwrite_hdf5, bow_3d,  ofile, 's_' + nistring(subject_num), 'bow_3d'
  niwrite_hdf5, pet_3d,  ofile, 's_' + nistring(subject_num), 'pet_3d'
ENDFOREACH

niwrite_hdf5, 1.,     ofile, 'header', 'pixelsize_mm'
niwrite_hdf5, betbow, ofile, 'header', 'beta_bowsher'
niwrite_hdf5, fwhm0_data,  ofile, 'header', 'fwhm0_data'
niwrite_hdf5, fwhm0_recon, ofile, 'header', 'fwhm0_recon'
niwrite_hdf5, counts,      ofile, 'header', 'counts'  

niproj_free, proj_mu
niproj_free, proj_data
niproj_free, proj_recon

END
