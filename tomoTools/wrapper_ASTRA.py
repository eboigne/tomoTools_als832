import numpy as np
import numexpr as ne
import sys
import pkgutil
import os
import time
import astra
import tifffile
import scipy.ndimage.morphology

def FP(rec, angles, supersampling = 1, use_cuda = True, pixel_width = 1.0, voxel_size = 1.0, ang_noise = 0):

    if isinstance(angles, (int, float)):
        angles = np.linspace(0, np.pi, angles, False)
    n_angles = len(angles)
    n_rays = rec.shape[0]
        
    if ang_noise > 0:
        angles += ang_noise / 0.144 * 0.5*(np.random.rand(n_angles)-0.5)*np.pi/180  # add noise of about ~0.15 deg each
        print('\n\n \t ====== Adding noise on angle on sinogram ======\n\n')
        
    if use_cuda:
        cfg = astra.astra_dict('FP_CUDA')
    else:
        cfg = astra.astra_dict('FP')

    proj_geom = astra.create_proj_geom('parallel', pixel_width/voxel_size, n_rays, angles)
    vol_geom = astra.create_vol_geom(n_rays, n_rays)
    
    # Other method for high resolution image
#     vol_geom_highRes = astra.create_vol_geom(n_rays*highResFactor, n_rays*highResFactor)
#     proj_geom_highRes = astra.create_proj_geom('parallel', 1.0, n_rays*highResFactor, angles)
#     proj_id = astra.create_projector('cuda',proj_geom_highRes,vol_geom_highRes)
#     im = rec[:,:]*pixel_width_on_projector/highResFactor
#     im_highRes = scipy.ndimage.zoom(im, highResFactor,order=3)
#     sino_id, sino = astra.create_sino(im_highRes, proj_id)
#     sino_lowRes = scipy.ndimage.zoom(sino, [1, 1./highResFactor], order = 3)
                  
    sino_id = astra.data2d.create('-sino', proj_geom)
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg['VolumeDataId'] = rec_id
#     cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {}
    # cfg['option']['PixelSuperSampling'] = supersampling
    cfg['option']['DetectorSuperSampling'] = supersampling
    
    if not(use_cuda):
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        cfg['ProjectorId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.data2d.store(rec_id, rec * pixel_width)
    astra.algorithm.run(alg_id)

    sino = astra.data2d.get(sino_id)

    astra.algorithm.delete(alg_id) 
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)   
    
    if not(use_cuda):
        astra.projector.delete(proj_id)
        
    return(sino)

def BP(sino, supersampling = 1, use_cuda = True, pixel_width = 1.0, voxel_size = 1.0, center_rot = 0.0, mask_array = 0, crop_outer_circle = False):

    n_angles, n_rays = sino.shape
    angles = np.linspace(0, np.pi, n_angles, False)
        
    if use_cuda:
        cfg = astra.astra_dict('BP_CUDA')
    else:
        cfg = astra.astra_dict('BP')

    vol_geom = astra.create_vol_geom(n_rays,n_rays)
    proj_geom = astra.create_proj_geom('parallel', pixel_width/voxel_size, n_rays, angles)
    #proj_geom['option']={'ExtraDetectorOffset':(centerRot-n_rays/2.)*np.ones(n_angles)}
    proj_geom['option']={'ExtraDetectorOffset':(center_rot)*np.ones(n_angles)}
    
    sino_id = astra.data2d.create('-sino', proj_geom)
    rec_id = astra.data2d.create('-vol', vol_geom)

#     cfg['VolumeDataId'] = rec_id_basic
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {}
    cfg['option']['PixelSuperSampling'] = supersampling
    cfg['option']['DetectorSuperSampling'] = supersampling
    
    if mask_array != 0:
        mask_id = astra.data2d.create('-vol', vol_geom, mask_array)
        cfg['option']['ReconstructionMaskId'] = mask_id
        
    if not(use_cuda):
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        cfg['ProjectorId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.data2d.store(sino_id, sino)
    astra.algorithm.run(alg_id)

    rec = astra.data2d.get(rec_id) / pixel_width
    
    astra.algorithm.delete(alg_id)    
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    
    if not(use_cuda):
        astra.projector.delete(projector_id)
        
    if crop_outer_circle:
        rec = np.squeeze(tomopy.circ_mask(np.reshape(rec, [1, n_rays,n_rays]),0))
        
    return(rec)

def FBP(sino, filter_type = 'ram-lak', supersampling = 1, use_cuda = True, pixel_width = 1.0, voxel_size = 1.0, center_rot = 0.0, mask_array = 0, crop_outer_circle = False, n_depth = 0, angles = []):

    n_angles, n_rays = sino.shape
    
    if len(angles) == 0:
        angles = np.linspace(0, np.pi, n_angles, False)
    else:
        angles = np.array(angles)
        
#     if n_angles > 2047:
#         print('\n \tDanger -- More than 2047 angles -- FBP implementation of ASTRA will yield weird values.')
        
    if use_cuda:
        cfg = astra.astra_dict('FBP_CUDA')
    else:
        cfg = astra.astra_dict('FBP')

    if n_depth == 0:
        n_depth = n_rays
    
    vol_geom = astra.create_vol_geom(n_depth,n_rays)
    proj_geom = astra.create_proj_geom('parallel', pixel_width/voxel_size, n_rays, angles)

    # proj_geom['option']={'ExtraDetectorOffset':(center_rot)*np.ones(n_angles)} # Deleted in ASTRA 1.9.0 (see https://github.com/astra-toolbox/astra-toolbox/blob/c02ff2d5f1f8b64525ea3bd5a6552e88b46baa9a/NEWS.txt)
    sino = scipy.ndimage.interpolation.shift(sino,(0, -center_rot), order = 3, mode='constant', cval = 0.0) # Instead using this, minus sign for continuity with ExtraDetectorOffset

    rec_id = astra.data2d.create('-vol', vol_geom)
    sino_id = astra.data2d.create('-sino', proj_geom)

#     cfg['VolumeDataId'] = rec_id_basic
    cfg['FilterType'] = filter_type
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {}
    cfg['option']['PixelSuperSampling'] = supersampling
#     cfg['option']['DetectorSuperSampling'] = supersampling
    
    if mask_array != 0:
        mask_id = astra.data2d.create('-vol', vol_geom, mask_array)
        cfg['option']['ReconstructionMaskId'] = mask_id
        
    if not(use_cuda):
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        cfg['ProjectorId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.data2d.store(sino_id, sino)
    astra.algorithm.run(alg_id)

    rec = astra.data2d.get(rec_id) / pixel_width
    
    astra.algorithm.delete(alg_id)    
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    
    if not(use_cuda):
        astra.projector.delete(projector_id)
        
    if crop_outer_circle:
        rec = np.squeeze(tomopy.circ_mask(np.reshape(rec, [1, n_rays,n_rays]),0))
        
    return(rec)

def SIRT(sino, nb_it, supersampling = 1, use_cuda = True, pixel_width = 1.0, voxel_size = 1.0, center_rot = 0.0, mask_array = 0, crop_outer_circle = False, n_depth = 0, angles = [], other_algo = ''):

    n_angles, n_rays = sino.shape
    
    if len(angles) == 0:
        angles = np.linspace(0, np.pi, n_angles, False)
    else:
        angles = np.array(angles)
        
    if use_cuda:
        if other_algo == '':
            cfg = astra.astra_dict('SIRT_CUDA')
        else:
            cfg = astra.astra_dict(other_algo)
    else:
        cfg = astra.astra_dict('SIRT')

    if n_depth == 0:
        n_depth = n_rays
    
    vol_geom = astra.create_vol_geom(n_depth,n_rays)
    proj_geom = astra.create_proj_geom('parallel', pixel_width/voxel_size, n_rays, angles)
    #proj_geom['option']={'ExtraDetectorOffset':(centerRot-n_rays/2.)*np.ones(n_angles)}
    proj_geom['option']={'ExtraDetectorOffset':(center_rot)*np.ones(n_angles)}
    
    sino_id = astra.data2d.create('-sino', proj_geom)
    rec_id = astra.data2d.create('-vol', vol_geom)

#     cfg['VolumeDataId'] = rec_id_basic
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {}
    cfg['option']['PixelSuperSampling'] = supersampling
    cfg['option']['DetectorSuperSampling'] = supersampling
    
    if mask_array != 0:
        mask_id = astra.data2d.create('-vol', vol_geom, mask_array)
        cfg['option']['ReconstructionMaskId'] = mask_id
        
    if not(use_cuda):
        proj_id = astra.create_projector('strip', proj_geom, vol_geom)
        cfg['ProjectorId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.data2d.store(sino_id, sino)
    astra.algorithm.run(alg_id, nb_it)

    rec = astra.data2d.get(rec_id) / pixel_width
    
    astra.algorithm.delete(alg_id)    
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sino_id)
    
    if not(use_cuda):
        astra.projector.delete(projector_id)
        
    if crop_outer_circle:
        rec = np.squeeze(tomopy.circ_mask(np.reshape(rec, [1, n_rays,n_rays]),0))
        
    return(rec)
