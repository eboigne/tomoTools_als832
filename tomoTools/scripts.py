import numpy as np
import sys, os
import time
import tifffile
import tomopy
import matplotlib.pyplot as plt
import scipy
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
import torch
import skimage
from skimage import measure
from skimage import filters
import multiprocessing
import wrapper_ASTRA
import h5py
import File
import gc # Garbage collected

from rigidTransform3D import *

def print2(str, end = '\n'):
    sys.stdout.write(str+end)
    sys.stdout.flush()

def clearFolder(pathFolder, boolClear=False):
    # Create new folder and clear it if boolClear is true

    if not os.path.exists(pathFolder):
        os.makedirs(pathFolder)
    elif boolClear:
        # Delete previous slices
        for the_file in os.listdir(pathFolder):
            file_path = os.path.join(pathFolder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    clearFolder(file_path, boolClear=True)
                    os.rmdir(file_path)
            except Exception as e:
                sprint(e+'\n')
# Setup and registerCall function

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def apply_3d_image_processing_on_subvolumes(vol, fct, chunk_size_max = (100, 100, 100), padding = 'extend', overlap = 3, bool_try_smaller_chunks = True, *args, **kwargs):

    try:
        verbose = False

        # kernel_array = custom_3d_kernel_sphere(radius)
        # pad_size = (np.array(kernel_array.shape)+1)//2
        pad_size = np.array([overlap, overlap, overlap])

        vol_out = np.zeros_like(vol)
        N = vol.shape[0]
        nb_chunks = vol.shape // np.array(chunk_size_max)+1
        chunk_size = vol.shape // nb_chunks+1
        print('\t Processing '+str(np.prod(nb_chunks))+' chunks of size '+str(chunk_size-1)+' ', end = '')

        i1 = 0
        i2 = min(chunk_size[0], vol.shape[0])

        for i in range(nb_chunks[0]):
            j1 = 0
            j2 = min(chunk_size[1], vol.shape[1])

            for j in range(nb_chunks[1]):
                k1 = 0
                k2 = min(chunk_size[2], vol.shape[2])

                for k in range(nb_chunks[2]):
                    # Select subvolume, with overlap
                    sub_vol = vol[max(0, i1-pad_size[0]):min(vol.shape[0], i2+pad_size[0]),max(0, j1-pad_size[1]):min(vol.shape[1], j2+pad_size[1]), max(0, k1-pad_size[2]):min(vol.shape[2], k2+pad_size[2])]

                    # Add padding (to deal with subvolumes on sides)
                    sub_vol = apply_padding(sub_vol, padding, pad_size)

                    # Apply function to subvolume
                    sub_vol = fct(sub_vol, *args, **kwargs)

                    # Remove padding
                    sub_vol = sub_vol[pad_size[0]:sub_vol.shape[0]-pad_size[0], pad_size[1]:sub_vol.shape[1]-pad_size[1], pad_size[2]:sub_vol.shape[2]-pad_size[2]]

                    # Remove overlap
                    sub_vol = sub_vol[(i1-pad_size[0] > 0) * pad_size[0]:sub_vol.shape[0]-(i2+pad_size[0] < vol.shape[0]) * pad_size[0], :, :]
                    sub_vol = sub_vol[:, (j1-pad_size[1] > 0) * pad_size[1]:sub_vol.shape[1]-(j2+pad_size[1] < vol.shape[1]) * pad_size[1], :]
                    sub_vol = sub_vol[:, :, (k1-pad_size[2] > 0) * pad_size[2]:sub_vol.shape[2]-(k2+pad_size[2] < vol.shape[2]) * pad_size[2]]
                    vol_out[i1:i2, j1:j2, k1:k2] = sub_vol

                    print('.', end = '')
                    k1 = k2
                    k2 = min(k2+chunk_size[2], vol.shape[2])
                j1 = j2
                j2 = min(j2+chunk_size[1], vol.shape[1])
            i1 = i2
            i2 = min(i2+chunk_size[0], vol.shape[0])
        print(' Done')

        return(vol_out)

    except ValueError as error:
        if np.array(chunk_size_max).min() > 1 and bool_try_smaller_chunks:
            print('\n\tError with chunk size, trying smaller chunks')
            return(apply_3d_image_processing_on_subvolumes(vol, fct, chunk_size_max = np.array(chunk_size_max)-1, padding = padding, overlap = overlap, *args, **kwargs))
        else:
            print('\n\tError with chunk size')
            raise(error)

def apply_padding(vol, padding, pad_size):

    vol_out = np.zeros(vol.shape+2*np.array(pad_size))
    vol_out[pad_size[0]:vol_out.shape[0]-pad_size[0], pad_size[1]:vol_out.shape[1]-pad_size[1], pad_size[2]:vol_out.shape[2]-pad_size[2]] = vol

    if padding == 'extend':
            vol_out[0:pad_size[0],:,:] += vol_out[[pad_size[0]],:,:]
            vol_out[vol_out.shape[0]-pad_size[0]:,:,:] += vol_out[vol_out.shape[0]-(pad_size[0]+1):vol_out.shape[0]-pad_size[0],:,:]
            vol_out[:,0:pad_size[1],:] += vol_out[:,[pad_size[1]],:]
            vol_out[:,vol_out.shape[1]-pad_size[1]:,:] += vol_out[:,vol_out.shape[1]-(pad_size[1]+1):vol_out.shape[1]-pad_size[1],:]
            vol_out[:,:,0:pad_size[2]] += vol_out[:,:,[pad_size[2]]]
            vol_out[:,:,vol_out.shape[2]-pad_size[2]:] += vol_out[:,:,vol_out.shape[2]-(pad_size[2]+1):vol_out.shape[2]-pad_size[2]]

    elif isinstance(padding, int) | isinstance(padding, float):
            vol_out[0:pad_size[0],:,:] = padding
            vol_out[vol_out.shape[0]-pad_size[0]:,:,:] = padding
            vol_out[:,0:pad_size[1],:] = padding
            vol_out[:,vol_out.shape[1]-pad_size[1]:,:] = padding
            vol_out[:,:,0:pad_size[2]] = padding
            vol_out[:,:,vol_out.shape[2]-pad_size[2]:] = padding

    return(vol_out)

def image_angles(h5py_file):
    img_angles = []
    meta = h5py_file['defaults']['group_attrs']['metadata']
    for img_str in meta:
        img_angles.append(meta[img_str]['rot_angle'].value)
    img_angles = np.sort(np.array(img_angles).astype('float32'))
    return(img_angles[img_angles>0])

def moving_average(x, w):
    return np.convolve(x.squeeze(), np.ones(w), 'same') / w

def fast_pytorch_convolution(img, kernel_array, verbose = False, chunk_size = 71):

    tic1 = time.time()
    if verbose:
        print('Starting pytorch convolution (for filtering)')

    # if img.dtype == np.dtype('bool'):
    #     kernel_array = np.reshape(kernel_array, (1,1)+kernel_array.shape).astype('bool')
    #     img = np.reshape(img, (1,1)+img.shape).astype('bool')
    # else:
    kernel_array = np.reshape(kernel_array, (1,1)+kernel_array.shape).astype('float32')
    img = np.reshape(img, (1,1)+img.shape).astype('float32')

    # torch.as_tensor gives pointers, not copies
    tic = time.time()
    tensor_kernel = torch.as_tensor(kernel_array).cuda()
    tensor_img = torch.as_tensor(img).cuda()
    if verbose:
        print('\tNumpy to Cuda (convolution) took: '+str(time.time()-tic)+' s')

    tic = time.time()
    try:
        out_tensor = torch.nn.functional.conv3d(tensor_img, tensor_kernel, bias=None, stride=1, padding=kernel_array.shape[-1]//2)
        if verbose:
            print('\tConvolution took: '+str(time.time()-tic)+' s')

        tic = time.time()
        out = out_tensor.cpu().detach().numpy()
        if verbose:
            print('\tCuda to Numpy (convolution) took: '+str(time.time()-tic)+' s')

        del tensor_img, tensor_kernel
        torch.cuda.empty_cache()

        if verbose:
            print('PyTorch filtering took: '+str(time.time()-tic1)+' s')
        return(np.squeeze(out))

    except Exception as error:
        del tensor_img, tensor_kernel
        torch.cuda.empty_cache()
        raise(error)

def fast_pytorch_mask_dilation(mask, radius, use_pyTorch = True):
    if radius > 0:
        kernel = custom_3d_kernel_sphere(radius)
        if use_pyTorch:
            mask_grown = fast_pytorch_convolution(mask.astype('bool'), kernel, verbose = False)
        else: # cpu, lot slower
            mask_grown = skimage.morphology.binary_dilation(mask.astype('bool'),kernel)
        return(mask_grown > 0)
    else:
        return(mask)

def mask_dilation_2d(mask, radius_dilation, factor_downsample_for_dilation = 1, use_successive_1pixel_dilation = False):

    if use_successive_1pixel_dilation:
        # Successive radius-1 dilations [Faster than 1 large dilatation (not equivalent, but ok)]
        if factor_downsample_for_dilation > 1:
            out = skimage.transform.rescale(mask, 1.0 / factor_downsample_for_dilation, multichannel=True) # Work on downsampled image to accelerate
        else:
            out = mask
        disk_dilation = skimage.morphology.disk(radius = 1)
        nb_dilations = int(1.0 * radius_dilation / factor_downsample_for_dilation)
        for k in range(nb_dilations):
            out = skimage.morphology.binary_dilation(out,disk_dilation)
        if factor_downsample_for_dilation > 1:
            out = skimage.transform.resize(out, self.init_image.shape)
        out = out > 0 # To bool array
    else:
        # 1 large dilation
        if factor_downsample_for_dilation > 1:
            out = skimage.transform.rescale(out, 1.0 / factor_downsample_for_dilation, multichannel=True) # Work on downsampled image to accelerate
        disk_dilation = skimage.morphology.disk(radius = radius_dilation)
        out = skimage.morphology.binary_dilation(out,disk_dilation)
        if factor_downsample_for_dilation > 1:
            out = skimage.transform.resize(out, self.init_image.shape)
        out = out > 0 # To bool array

    return(out)

def read_h5_file(path_save, path_files = []):

    if path_files == []:
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        path_files = askopenfilenames(initialdir=path_save) # show an "Open" dialog box and return the path to the selected file

    h5py_files = []
    for path_file in path_files:
        print('Using file: '+str(path_file))
        h5py_file = h5py.File(path_file,"r")
        try:
            theta_white = np.array(h5py_file['exchange']['theta_white'])
            theta_dark = np.array(h5py_file['exchange']['theta_dark'])
            theta = np.array(h5py_file['exchange']['theta']) # Equal to np.linspace(0,180,#angles) within 1e-7 error (negligeable)
            print('\t- '+str(len(theta_dark))+' dark fields')
            print('\t- '+str(len(theta_white))+' white fields')
            print('\t- '+str(len(theta))+' projections')
        except:
            print('\t- Some trouble reading data sizes')

        size_proj = h5py_file['exchange']['data'].shape[1:]
        print('\t- Image size: '+str(size_proj))

        h5py_file.path_file = path_file
        h5py_file.path_folder = path_save
        h5py_file.file_name = (h5py_file.path_file).split('/')[-1]
        h5py_file.file_name_noExtension = '.'.join((h5py_file.file_name).split('.')[:-1])
        h5py_file.height = size_proj[0]
        h5py_file.width = size_proj[1]

        h5py_files.append(h5py_file)
    return(h5py_files)

def print_gpu_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('GPU Memory usage:')
    print('\tTotal memory: '+str(t/2**20)+' MB')
    print('\tMemory reserved by PyTorch: '+str(r/2**20)+' MB')
    print('\tMemory allocated by PyTorch: '+str(a/2**20)+' MB')
    print('\tFree allocated memory : '+str(f/2**20)+' MB')
    print('\tTotal free memory: '+str((t-a)/2**20)+' MB')

def fast_pytorch_median_filter2d(img, kernels): # Give torch kernels, or an integer for the kernel_half_width

    if ~isinstance(kernels, torch.Tensor):
        N = kernels*2 + 1 # MUST BE ODD
        kernels = []
        for i in range(N):
            for j in range(N):
                kernel_zero = np.zeros([N,N])
                kernel_zero[i,j] = 1.0
                kernels.append(kernel_zero)
        kernels = np.array(kernels)
        kernels = np.reshape(kernels, (N*N, 1)+kernels.shape[1:]).astype('float32')
        kernels = torch.as_tensor(kernels).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()

    img_tensor_arrayForMedian = torch.nn.functional.conv2d(img_tensor, kernels, bias=None, stride=1, padding=N//2)
    img_tensor_median = img_tensor_arrayForMedian.median(dim=1)[0]
    return(img_tensor_median.cpu().detach().numpy().squeeze())

def fast_pytorch_remove_zingers_chunk(img_stack, kernel_half_width, threshold_zinger, chunk_size = 50):

    N = img_stack.shape[0]
    nb_chunk = N//chunk_size+1
    img_stack2 = np.zeros_like(img_stack)

    if img_stack.shape[0] < chunk_size:
        return(fast_pytorch_remove_zingers_v2(img_stack, kernel_half_width, threshold_zinger, verbose = False))
    i1 = 0
    i2 = chunk_size
    img_stack2[i1:i2] = fast_pytorch_remove_zingers_v2(img_stack[i1:i2+kernel_half_width], kernel_half_width, threshold_zinger, verbose = False)[:-kernel_half_width]
    print('.', end = '')
    for ind_chunk in range(nb_chunk-2):
        i1 += chunk_size
        i2 += chunk_size
        img_stack2[i1:i2] = fast_pytorch_remove_zingers_v2(img_stack[i1-kernel_half_width:i2+kernel_half_width], kernel_half_width, threshold_zinger, verbose = False)[kernel_half_width:-kernel_half_width]
        print('.', end = '')

    i1 += chunk_size
    img_stack2[i1:] = fast_pytorch_remove_zingers_v2(img_stack[i1-kernel_half_width:], kernel_half_width, threshold_zinger, verbose = False)[kernel_half_width:]
    img_stack = img_stack2
    print('.', end = '')
    return(img_stack)

def fast_pytorch_remove_zingers_v1(img_stack, kernel_half_width, threshold_zinger, verbose = True):

    # v1: Mean filter. Issue: mean is biased by zinger from neighbor slice

    N = kernel_half_width*2 + 1 # MUST BE ODD

    kernel = np.ones([N,1,1])
    kernel[N//2] = 0.0
    kernel /= np.sum(kernel)
    kernel = np.reshape(kernel, (1, 1)+kernel.shape).astype('float32')
    kernel = torch.as_tensor(kernel).cuda()

    x = np.reshape(img_stack, (1,1)+img_stack.shape).astype('float32')
    x_tensor = torch.as_tensor(x).cuda()

#     x2 = torch.nn.functional.conv3d(x_tensor, kernel, bias=None, stride=1, padding=(N//2, 0, 0))

    # Manual padding
    N2 = N//2
    pad_fct = torch.nn.ReplicationPad3d((0,0,0,0,N2,N2))
    x2 = pad_fct(x_tensor)
    for i in range(N2):
        x2[0,0,i,:,:] = x2[0,0,N2*2-i]
        x2[0,0,-(i+1),:,:] = x2[0,0,-(N2*2-i)]
#     print_gpu_memory()
    x2 = torch.nn.functional.conv3d(x2, kernel, bias=None, stride=1, padding=0)

    bool_error = (((x_tensor-x2) / x2).abs() > threshold_zinger)
    if verbose:
        print('\tCorrected '+str(bool_error.sum().cpu().numpy())+' outlier pixels')
    for i in range(bool_error.shape[2]):
        this_image_x = x_tensor[0,0,i]
        this_image_x2 = x2[0,0,i]
        this_image_x[bool_error[0,0,i]] = this_image_x2[bool_error[0,0,i]]
        x_tensor[0,0,i] = this_image_x
    x = x_tensor.cpu().detach().numpy().squeeze()

    del x_tensor, kernel, x2, bool_error
    torch.cuda.empty_cache()
    return(x)

def fast_pytorch_remove_zingers_v2(img_stack, kernel_half_width, threshold_zinger, verbose = True):

    x = np.reshape(img_stack, (1,1)+img_stack.shape).astype('float32')
    x_tensor = torch.as_tensor(x).cuda()

    # Manual padding
    pad_fct = torch.nn.ReplicationPad3d((0,0,0,0,kernel_half_width,kernel_half_width))
    x2 = pad_fct(x_tensor)
    for i in range(kernel_half_width):
        x2[0,0,i] = x2[0,0,kernel_half_width*2-i]
        x2[0,0,-(i+1)] = x2[0,0,-(kernel_half_width*2-i)]

    x2 = x2.squeeze()
    for i in range(x_tensor.shape[2]):
    #     if i < kernel_half_width:
        this_med = x2[i:i+2*kernel_half_width+1].median(dim=0)[0]
    #     elif i+kernel_half_width+1 > x2.shape[0]:
    #         x2_med = x2[i:i+N].median(dim=0)[0]
    #     else:
    #         x2_med = x2[i-kernel_half_width:i+kernel_half_width+1].median(dim=0)[0]
        bool_error = (((x_tensor[0,0,i]-this_med) / this_med).abs()) > threshold_zinger
        this_image_x = x_tensor[0,0,i]
        this_image_x[bool_error] = this_med[bool_error]
        x_tensor[0,0,i] = this_image_x

    x = x_tensor.cpu().detach().numpy().squeeze()

    del x_tensor, x2, bool_error, this_image_x, this_med, pad_fct
    torch.cuda.empty_cache()
    return(x)

def fast_pytorch_bin(img_stack,bin_factor, bin_factor_angle = 1):
    img_stack_tensor = torch.as_tensor(np.reshape(img_stack, (1,1)+img_stack.shape).astype('float32')).cuda()
    kernel = torch.as_tensor(np.ones([1,1,bin_factor_angle,bin_factor,bin_factor]).astype('float32')).cuda()
    kernel /= kernel.sum()

    out_stack = torch.nn.functional.conv3d(img_stack_tensor, kernel, bias=None, stride=(bin_factor_angle,bin_factor,bin_factor), padding=0)
    out = out_stack.cpu().detach().numpy().squeeze()
    del kernel, img_stack_tensor, out_stack
    torch.cuda.empty_cache()
    return(out)

def fast_pytorch_bin_2d(img,bin_factor, cpu = False):
    kernel = torch.as_tensor(np.ones([1,1,bin_factor,bin_factor]).astype('float32'))
    if not cpu:
        kernel = kernel.cuda()
    kernel /= kernel.sum()

    img_tensor = torch.as_tensor(np.reshape(img, (1,1)+img.shape).astype('float32'))
    if not cpu:
        img_tensor = img_tensor.cuda()
    out_tensor = torch.nn.functional.conv2d(img_tensor, kernel, bias=None, stride=(bin_factor,bin_factor), padding=0)
    out = out_tensor.cpu().detach().numpy().squeeze()

    del img_tensor, out_tensor, kernel
    torch.cuda.empty_cache()
    return(out)

def fast_pytorch_mean_2d(img,mean_factor):
    kernel = torch.as_tensor(np.ones([1,1,mean_factor,mean_factor]).astype('float32')).cuda()
    kernel /= kernel.sum()

    img_tensor = torch.as_tensor(np.reshape(img, (1,1)+img.shape).astype('float32')).cuda()
    out_tensor = torch.nn.functional.conv2d(img_tensor, kernel, bias=None, stride=(1, 1), padding=mean_factor//2)
    out = out_tensor.cpu().detach().numpy().squeeze()

    del img_tensor, out_tensor, kernel
    torch.cuda.empty_cache()
    return(out[:img.shape[0], :img.shape[1]])

def fast_pytorch_bin_3d(vol_stack,bin_factor, chunk_size = 4*68, cpu = False):
    try:
        N = vol_stack.shape[0]
        nb_chunk = N//chunk_size+1
        chunk_size = N//nb_chunk

        out_stack = []
        i1 = 0
        i2 = min(chunk_size, N)
        # print('.', end = '')

        kernel = torch.as_tensor(np.ones([1,1,bin_factor,bin_factor,bin_factor]).astype('float32'))
        if not cpu:
            kernel = kernel.cuda()

        kernel /= kernel.sum()

        for ind_chunk in range(nb_chunk):
            vol_stack_tensor = torch.as_tensor(np.reshape(vol_stack[i1:i2], (1,1)+vol_stack[i1:i2].shape).astype('float32'))
            if not cpu:
                vol_stack_tensor = vol_stack_tensor.cuda()
            out = torch.nn.functional.conv3d(vol_stack_tensor, kernel, bias=None, stride=(bin_factor,bin_factor,bin_factor), padding=0)
            out_stack.append(out.cpu().detach().numpy().squeeze())

            i1 += chunk_size
            i2 = min(i2+chunk_size, N)
            # print('.', end = '')
            del vol_stack_tensor, out
        del kernel
        torch.cuda.empty_cache()

        return(np.concatenate(out_stack))
    except Exception as e:
        print('\tError with this chunk size, trying smaller')
        print(str(e))
        if chunk_size > 2:
            return(fast_pytorch_bin_3d(vol_stack,bin_factor, chunk_size = chunk_size - 1))
        else:
            raise Exception('Error with chunk size')

def fast_pytorch_bin_chunk(img_stack,bin_factor, bin_factor_angle = 1, chunk_size = 500, verbose = True):

    N = img_stack.shape[0]
    nb_chunk = N//chunk_size+1
    out_stack = []
    i1 = 0
    i2 = chunk_size
    if verbose:
        print('.', end = '')
    for ind_chunk in range(nb_chunk):
        out_stack.append(fast_pytorch_bin(img_stack[i1:i2], bin_factor, bin_factor_angle = bin_factor_angle))
        i1 += chunk_size
        i2 = min(i2+chunk_size, N)
        if verbose:
            print('.', end = '')
    return(np.concatenate(out_stack))

def apply_offset(image, offset): # Apply a center-of-rotation offset to the sinogram.
    # This is to correct for when the exact COR is not aligned with the center of the detector

    if offset == 0:
        return(image)

    image_out = np.zeros_like(image)
    floor = np.int(np.floor(offset))
    ratio = np.abs(offset-floor)
    floor = np.abs(floor)
    if len(image.shape) == 2:
        if offset > 0:
            image_out[:,:-(floor+1)] = (1-ratio)*image[:,floor:-1] + ratio*image[:,floor+1:]
        elif offset >= -1.0:
            image_out[:,floor:] = (1-ratio)*image[:,:-floor] + ratio*image[:,1:]
        else:
            image_out[:,floor:] = (1-ratio)*image[:,:-floor] + ratio*image[:,1:-(floor-1)]
    else:
        for i in range(image.shape[0]):
            image_out[i,:,:] = apply_offset(image[i,:,:], offset)


    return(image_out)





def reconstruct_scan(path_in, scan, path_save = '', ):

    if path_save == '':
        path_save = path_in
    else:
        path_save += '/'

    if not 'split_half_circles' in scan.keys():
        scan['split_half_circles'] = False

    print('\n')
    use_dark_from_scan = scan['use_dark_from_scan']
    if not use_dark_from_scan:
        h5py_file_dark = read_h5_file(path_in, path_files=[scan['path_dark'],])[0]
    use_flat_from_scan = scan['use_flat_from_scan']
    if not use_flat_from_scan:
        h5py_file_flat = read_h5_file(path_in, path_files=[scan['path_flat'],])[0]
    h5py_files = read_h5_file(path_in, path_files=[scan['path_proj'],])
    h5py_file = h5py_files[0]

    # Acquisition parameters
    pixel_size_microns = 3.24 # [microns]
    N_angles_per_half_circle = scan['N_angles_per_half_circle']
    N_half_circles = 2 # With half-circles, and 0-180 range: the 180 angle is repeated (acquired twice)
    COR = scan['COR'] # / bin_factor

    # Pre-processing parameters
    bin_factor = 2 # Bin projection, flat & dark images by this factor
    bin_factor_angle = 1 # Bin projection images along the angle direction by this factor
    bin_gpu_chunk_size = 30 * bin_factor * bin_factor + 7 # Max number of slices simultaneously processed on the GPU
    filter_type = 'ram-lak' # FBP filter. Note from ram-lak, apply Gaussian blur with radius sigma = 1 yields results really close to Parzen.
    skip_first_flats = 20 # Skip the first few N flats, usually not as good quality.

    # Double normalization parameters:
    # normalize out fluctuations in photon counts between successive projection images
    # using mean value in a window at the left and right of the FOV (where no sample is)
    boolean_use_DN = True # Whether to use the double normalization
    window_width_DN = 25 // bin_factor # Width of the window [in pixels]
    window_cutEdge_DN = 10 // bin_factor # How far the window is from the left and right edges of the FOV [in pixels]
    # ind_range_DN = np.concatenate((np.arange(window_cutEdge_DN, window_width_DN,1),np.arange(h5py_file.width//bin_factor-(window_width_DN),2560//bin_factor-window_cutEdge_DN,1)))
    ind_range_DN = np.concatenate((np.arange(window_cutEdge_DN, window_width_DN,1),np.arange(2560//bin_factor-(window_width_DN),2560//bin_factor-window_cutEdge_DN,1)))


    # Outlier correction
    use_outlier_correction = True
    gpu_median_filter_chunk_size = 100 * bin_factor * bin_factor + 7 # Avoid divider of Nangles and Nflats. Even smaller because median filter requires large memory use. Check nvidia-smi
    outlier_kernel_half_width = 2 # Make it at least bin_factor, otherwise can miss zinglers on first/last slices
    outlier_zinger_threshold = 0.3

    # Saving parameters:
    proj_save_every = 31
    save_sinogram = False
    sino_save_every = 1

    # For fast post-processing
    crop_image = False
    ind_image_crop_bottom = 490
    ind_image_crop_top = 510

    # Not recommended unless necessary
    boolean_use_TN = False # True

    # Pre-process sinogram data: SHOULDN'T CHANGE ANYTHING IN THIS CELL
    height = h5py_file.height

    # Reconstruction
    if N_half_circles == 2:
    # For N_half_circles = 2, use ind_last_angle = 1 as last angle is the same as the first angle, and should thus be discarded
        ind_last_angle = 1 # Counting backwards. 0: using all angles, 1: Skip last angle, ...
    else:
    # For N_half_circles = 1, unsure since acquisition scheme changed in 2021.
        ind_last_angle = 0

    N_avg = 1
    angles_all = np.linspace(0, 2*np.pi, 2*(N_angles_per_half_circle*N_avg-(N_avg-1))-1)
    angles_to_use = angles_all
    angles_to_use1 = angles_to_use[:N_angles_per_half_circle-1]
    if ind_last_angle > 0:
        angles_to_use2 = angles_to_use[N_angles_per_half_circle-1:-ind_last_angle]
    else:
        angles_to_use2 = angles_to_use[N_angles_per_half_circle-1:]

    pixel_size_cm = pixel_size_microns / 1.0e4 * bin_factor

    ind_save_PMDOF_DN = np.concatenate((np.arange(0, (N_angles_per_half_circle-ind_last_angle)//bin_factor_angle, proj_save_every), np.arange((N_angles_per_half_circle-ind_last_angle)//bin_factor_angle-4,(N_angles_per_half_circle-ind_last_angle)//bin_factor_angle,1))).astype('int')
    ind_save_sinogram = np.arange(0, h5py_file.height, sino_save_every).astype('int')

    # ======================================== #

    print2('\tImporting dark -- ',end='')
    if use_dark_from_scan:
        dark = np.array(h5py_file['exchange']['data_dark']).astype('float32')
    else:
        dark = np.array(h5py_file_dark['exchange']['data']).astype('float32')
    print2('Done')

    if crop_image:
        dark = dark[:,ind_image_crop_bottom:ind_image_crop_top,:]

    if bin_factor>1:
        print2('\tBinning dark data ',end='')
        dark = fast_pytorch_bin_chunk(dark,bin_factor, chunk_size = bin_gpu_chunk_size)
        print2(' Done')
    dark_avg = np.mean(dark,axis = 0)

    ind_save_sinogram = ind_save_sinogram[ind_save_sinogram < dark.shape[1]]

    tic = time.time()
    print2('Processing file: '+str(h5py_file.path_file))

    print2('\tImporting flat -- ',end='')
    if use_flat_from_scan:
        flat = h5py_file['exchange']['data_white']
    else:
        flat = h5py_file_flat['exchange']['data']
    print2('Done')

    flat = flat[skip_first_flats:]
    if crop_image:
        flat = flat[:,ind_image_crop_bottom:ind_image_crop_top,:]

    print2('\tImporting projections -- ',end='')
    proj = h5py_file['exchange']['data']
    print2('Done')

    if crop_image:
        proj = proj[:,ind_image_crop_bottom:ind_image_crop_top,:]

    bin_str = str(bin_factor_angle)+'x'+str(bin_factor)+'x'+str(bin_factor)

    if bin_factor>1:
        print2('\tBinning data ',end='')
        flat = fast_pytorch_bin_chunk(flat,bin_factor, chunk_size = bin_gpu_chunk_size)
        if N_half_circles == 1:
            if ind_last_angle > 0:
                proj = proj[:-ind_last_angle]
            proj = fast_pytorch_bin_chunk(proj,bin_factor, bin_factor_angle = bin_factor_angle, chunk_size = bin_gpu_chunk_size)
        elif N_half_circles == 2:
            if ind_last_angle > 0:
                proj1 = fast_pytorch_bin_chunk(proj[:N_angles_per_half_circle-1],bin_factor, bin_factor_angle = bin_factor_angle, chunk_size = bin_gpu_chunk_size)
                proj2 = fast_pytorch_bin_chunk(proj[N_angles_per_half_circle-1:-ind_last_angle],bin_factor, bin_factor_angle = bin_factor_angle, chunk_size = bin_gpu_chunk_size)
            else:
                proj1 = fast_pytorch_bin_chunk(proj[:N_angles_per_half_circle-1],bin_factor, bin_factor_angle = bin_factor_angle, chunk_size = bin_gpu_chunk_size)
                proj2 = fast_pytorch_bin_chunk(proj[N_angles_per_half_circle-1:],bin_factor, bin_factor_angle = bin_factor_angle, chunk_size = bin_gpu_chunk_size)
        print2(' Done')
        ind_save_PMDOF_DN = ind_save_PMDOF_DN[ind_save_PMDOF_DN<proj2.shape[0]]
    else:
        if N_half_circles == 2:
            if ind_last_angle > 0:
                proj1 = proj[:N_angles_per_half_circle-1]
                proj2 = proj[N_angles_per_half_circle-1:-ind_last_angle]
            else:
                proj1 = proj[:N_angles_per_half_circle-1]
                proj2 = proj[N_angles_per_half_circle-1:]

    if use_outlier_correction:
        print2('\tRemoving outliers (zingers) for flats ',end='')
        flat = fast_pytorch_remove_zingers_chunk(flat, outlier_kernel_half_width, outlier_zinger_threshold, chunk_size = gpu_median_filter_chunk_size).astype('float32')
        print2(' Done')

        print2('\tRemoving outliers (zingers) for proj ',end='')
        if N_half_circles == 1:
            proj = fast_pytorch_remove_zingers_chunk(proj, outlier_kernel_half_width, outlier_zinger_threshold, chunk_size = gpu_median_filter_chunk_size).astype('float32')
        elif N_half_circles == 2:
            proj1 = fast_pytorch_remove_zingers_chunk(proj1, outlier_kernel_half_width, outlier_zinger_threshold, chunk_size = gpu_median_filter_chunk_size).astype('float32')
            proj2 = fast_pytorch_remove_zingers_chunk(proj2, outlier_kernel_half_width, outlier_zinger_threshold, chunk_size = gpu_median_filter_chunk_size).astype('float32')
        print2(' Done')

    print2('\tComputing transmission -- ',end='')
    flat += -dark_avg
    flat_avg = np.mean(flat,axis = 0).astype('float32')
    File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+'_bin'+bin_str+'_flat_avg', clear = True).saveTiff(flat_avg)
    if N_half_circles == 1:
        proj += -dark_avg
        proj /= flat_avg
    elif N_half_circles == 2:
        proj1 += -dark_avg
        proj2 += -dark_avg
        N_flat = flat.shape[0]
        proj1 /= flat_avg
        proj2 /= flat_avg
    print2('Done')

    if boolean_use_DN:
        print2('\tComputing double normalization -- ',end='')
        if N_half_circles == 1:
            double_norm = np.reshape(np.mean(proj[:,:,ind_range_DN], axis=(1,2)), [proj.shape[0], 1,1]) # One coeff per angle
            proj /= double_norm

            if boolean_use_TN:
                triple_norm = np.reshape(np.mean(proj[:,:,ind_range_DN], axis=(0,2)), [1, proj.shape[1],1]) # One coeff per slice (y-axis)
                triple_normB = np.copy(triple_norm)
                triple_normB[triple_normB<0.985] = 0.985
                triple_normB[triple_normB>1.015] = 1.015
                proj /= triple_normB

        elif N_half_circles == 2:
            double_norm1 = np.reshape(np.mean(proj1[:,:,ind_range_DN], axis=(1,2)), [proj1.shape[0], 1,1])
            double_norm2 = np.reshape(np.mean(proj2[:,:,ind_range_DN], axis=(1,2)), [proj2.shape[0], 1,1])

            proj1 /= double_norm1
            proj2 /= double_norm2

            if boolean_use_TN:
                triple_norm1 = np.reshape(np.mean(proj1[:,:,ind_range_DN], axis=(0,2)), [1, proj1.shape[1],1])
                triple_norm2 = np.reshape(np.mean(proj2[:,:,ind_range_DN], axis=(0,2)), [1, proj2.shape[1],1])

                triple_norm1B = np.copy(triple_norm1)
                triple_norm1B[triple_norm1B<0.985] = 0.985
                triple_norm1B[triple_norm1B>1.015] = 1.015

                triple_norm2B = np.copy(triple_norm2)
                triple_norm2B[triple_norm2B<0.985] = 0.985
                triple_norm2B[triple_norm2B>1.015] = 1.015

                proj1 /= triple_norm1B
                proj2 /= triple_norm2B
        print2('Done')
    else:
        print2('\tSkipping double normalization')

    print2('\tWriting PMDOF_DN to tif -- ',end='')
    if N_half_circles == 1:
        suffix = '_bin'+bin_str+'_PMDOF_DN/'
        File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(proj[ind_save_PMDOF_DN,:,:],ind=ind_save_PMDOF_DN)

    elif N_half_circles == 2:
        suffix = '_bin'+bin_str+'_PMDOF_DN_a_0-180/'
        File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(proj1[ind_save_PMDOF_DN,:,:],ind=ind_save_PMDOF_DN)

        suffix = '_bin'+bin_str+'_PMDOF_DN_b_180-360/'
        File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(proj2[ind_save_PMDOF_DN,:,:],ind=ind_save_PMDOF_DN)
    print2('Done')

    print2('\tMaking sinograms -- ',end='')
    if N_half_circles == 1:
        sino = -np.log(np.transpose(proj[:,ind_save_sinogram,:], [1, 0, 2]))
    elif N_half_circles == 2:
        sino1 = -np.log(np.transpose(proj1[:,ind_save_sinogram,:], [1, 0, 2]))
        sino2 = -np.log(np.transpose(proj2[:,ind_save_sinogram,:], [1, 0, 2]))
        sino = np.concatenate((sino1,sino2),axis = 1)
    print2('Done')

    del proj, proj1, proj2
    gc.collect()

    if save_sinogram:
        print2('\tWriting sinograms to tif -- ',end='')
        if N_half_circles == 1:
            suffix = '_bin'+bin_str+'_sino/'
            File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(sino,ind=ind_save_sinogram)
        elif N_half_circles == 2:
            suffix = '_bin'+bin_str+'_sino_a_0-180/'
            File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(sino1,ind=ind_save_sinogram)
            suffix = '_bin'+bin_str+'_sino_b_180-360_flipped/'
            File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(sino2,ind=ind_save_sinogram)
        print2('Done')


    skip_every_slice = 1

    if N_half_circles == 1:

        print2('\tDoing FBP reconstruction -- ',end='')
        rec = []
        angles = np.linspace(0, np.pi, sino.shape[1], False)
        for this_sino in sino:
            rec.append(wrapper_ASTRA.FBP(this_sino/pixel_size_cm, filter_type=filter_type, angles = angles, center_rot = COR))
        rec = np.array(rec)
        print2('Done')

        print2('\tSaving FBP reconstruction ('+str(rec.shape[0])+' slices) -- ',end='')
        suffix = '_bin'+bin_str+'_FBP_COR_'+str(COR).zfill(2)+'/'
        File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(rec,ind=ind_save_sinogram)
        print2('Done')

    elif N_half_circles == 2:

        print2('\tDoing FBP reconstruction (1/2) -- ',end='')
        rec1 = []
        angles = angles_to_use1
        for this_sino in sino1[::skip_every_slice]:
            rec1.append(wrapper_ASTRA.FBP(this_sino/pixel_size_cm, filter_type=filter_type, angles = angles, center_rot = COR))
        rec1 = np.array(rec1)
        print2('Done (1/2)')

        if scan['split_half_circles'] == True:
            print2('\tSaving FBP reconstruction (1/2) -- ',end='')
            suffix = '_bin'+bin_str+'_FBP_COR_'+str(COR).zfill(2)+'/'
            File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(rec1,ind=ind_save_sinogram)
            print2('Done (1/2)')

        print2('\tDoing FBP reconstruction (2/2) -- ',end='')
        rec2 = []
        angles = angles_to_use2
        for this_sino in sino2[::skip_every_slice]:
            rec2.append(wrapper_ASTRA.FBP(this_sino/pixel_size_cm, filter_type=filter_type, angles = angles, center_rot = COR))
        rec2 = np.array(rec2)
        print2('Done (2/2)')

        if scan['split_half_circles'] == True:
            print2('\tSaving FBP reconstruction (2/2) -- ',end='')
            suffix = '_bin'+bin_str+'_FBP_COR_'+str(COR).zfill(2)+'_b_180-360_flipped/'
            File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(rec2,ind=ind_save_sinogram)
            print2('Done (2/2)')
        else:
            print2('\tSaving averaged of two half circles -- ',end='')
            suffix = '_bin'+bin_str+'_FBP_COR_'+str(COR).zfill(2)+'_c_averaged_half_circles/'
            File(h5py_file.path_folder+h5py_file.file_name_noExtension+'/'+h5py_file.file_name_noExtension+suffix, clear = True).saveTiffStack(0.5*(rec1+rec2),ind=ind_save_sinogram)
            print2('Done')

    print2('This took '+str(time.time()-tic)+' s')


def import_data(path_save, scan, suffix = 'c_averaged_half_circles'):

    print('Importing data:')
    tic = time.time()
    path_scan = path_save+scan['path_proj'].split('/')[-1][:-3]
    path_scan = path_scan+'/'+scan['FBP_folder'] #[e for e in os.listdir(path_scan) if 'FBP' in e and not 'indLastAngle' in e and suffix in e][0]

    rec = File(path_scan).readAll()
    toc = time.time()
    print('This took '+str(toc-tic)+' s')

    return(rec)

def run_registration(folder_name, static, moving, path_mask, save_str = 'd', return_transform = False, nb_it = 5, smoothingSigmas = [3, 2, 2, 2, 2, 2, 2, 1],shrinkFactors = [8, 8, 4, 4, 4, 4, 4, 2], ind_z_start = 20):

    ind_z_end = static.shape[0]-ind_z_start
    ind_y_start = 0
    ind_y_end = static.shape[1]-ind_y_start
    ind_x_start = 0
    ind_x_end = static.shape[2]-ind_x_start
    ind_z_slice_to_follow = static.shape[0]//2 - ind_z_start
    ind_ROI_registration = (ind_z_start, ind_z_end, ind_y_start, ind_y_end, ind_x_start, ind_x_end)

    tic = time.time()
    clearFolder(folder_name+'/', boolClear=True)
    volRegistered, transform = registerCall(nb_it, static, moving, folder_name, path_mask, ind_ROI_registration = ind_ROI_registration, ind_z_slice_to_follow = ind_z_slice_to_follow, shrinkFactors = shrinkFactors, smoothingSigmas = smoothingSigmas, save_str = save_str)
    print('\t This took: '+str(time.time()-tic)+' s')

    if return_transform:
        return(volRegistered, transform)
    else:
        return(volRegistered)

def reduceMask(mask_solid, pxBorder):
    # Enlarge the solid mask by a little margin
    dilate_struct = scipy.ndimage.morphology.generate_binary_structure(2,pxBorder) # 2 is for 2D structure.
    mask_out = np.zeros_like(mask_solid)
    for i in range(mask_solid.shape[0]):
        mask_out[i,:,:] = scipy.ndimage.morphology.binary_erosion(mask_solid[i,:,:],dilate_struct, iterations=1)
    return mask_out

def modify_transform_binning_factor(transform, bin_factor):
    transform = safe_copy_transform(transform)

    center_x, center_y, center_z, center_angle = transform.GetFixedParameters()
    transform.SetFixedParameters((center_x*bin_factor, center_y*bin_factor, center_z*bin_factor, center_angle)) # Not a function of offset_bottom
    (eulerAngleX, eulerAngleY, eulerAngleZ, offsetX, offsetY, offsetZ) = transform.GetParameters()
    transform.SetParameters((eulerAngleX, eulerAngleY, eulerAngleZ, offsetX*bin_factor, offsetY*bin_factor, offsetZ*bin_factor))
    return(transform)

def registerCall(nb_it,refVol, vol, prefix_out, mask_path, ind_ROI_registration = (), ind_z_slice_to_follow = 0, shrinkFactors = [2, 1], smoothingSigmas = [2, 1], save_str = 'd'):

    maskTubeFile = File(mask_path)
    maskTube = maskTubeFile.read()

    def registerSubsetOfVolume(refVol, vol, refVolMask, volMask, pathSaveTransform, pathMoving = '', verbose = True, offset_it_number = 0, ind_z_slice_to_follow = 0, bin_factor = 1, initial_transform = '', **kwargs):

        refVolReg = refVol.copy()
        refVolReg[~refVolMask] = False
        volReg = vol.copy()
        volReg[~volMask] = False

        mask = reduceMask(volMask, 5)

        # Compute registration transform
        tic =  time.time()
        transform = computeTransform(refVolReg, volReg, pathSaveTransform, bool_save = True, mask = mask, pathMoving = pathMoving, verbose = verbose, ind_z_slice_to_follow = ind_z_slice_to_follow, offset_it_number = offset_it_number, bin_factor = bin_factor, initial_transform = initial_transform, **kwargs)
        toc =  time.time()
        print('\n\nComputed transform, took '+str(toc-tic)+' s')

        return(transform)

    pathMoving = prefix_out+'0_progress/'
    clearFolder(pathMoving)

    # Run registration

    N = len(shrinkFactors)
    transform = ''

    for i in range(N):
        kwargs = {}
        kwargs['learningRate']=1.0
        kwargs['numberOfIterations']= nb_it
        kwargs['convergenceMinimumValue'] = 1e-20 # 1e-5000 Threshold below which iterations stop
        kwargs['convergenceWindowSize'] = 2 # (>1) Nb of successive iterations used to compute the convergence minimum value (this exists because metric is usually not 100%, but because it is for me: 2 works fine /!| DONT USE 1). ALso number of minimal iterations.

        bin_factor = shrinkFactors[i]
        kwargs['shrinkFactors'] = [1,]
        kwargs['smoothingSigmas'] = [smoothingSigmas[i],]

        if i > 0:
            kwargs['initialTransform']= transform
            transform = modify_transform_binning_factor(transform, 1.0/bin_factor)

        vol_bin = fast_pytorch_bin_3d(vol,bin_factor, chunk_size = min(88, 34*bin_factor), cpu = ~torch.cuda.is_available())
        refVol_bin = fast_pytorch_bin_3d(refVol,bin_factor, chunk_size = min(88, 34*bin_factor), cpu = ~torch.cuda.is_available())
        maskTube_bin = (fast_pytorch_bin_2d(maskTube,bin_factor, cpu = ~torch.cuda.is_available())>0).astype('uint8')

        # Mask: Both gas and solid phases inside the tube
        refVolMask_bin = np.tile(maskTube_bin, [vol_bin.shape[0], 1, 1]).astype('bool')
        volMask_bin = np.tile(maskTube_bin, [vol_bin.shape[0], 1, 1]).astype('bool')

        pathSaveTransform = prefix_out+'transform.tfm'

        if len(ind_ROI_registration) > 0:
            ind_z_start, ind_z_end, ind_y_start, ind_y_end, ind_x_start, ind_x_end = (np.array(ind_ROI_registration)//bin_factor).tolist()
            volMask_bin = volMask_bin[ind_z_start:ind_z_end, ind_y_start:ind_y_end, ind_x_start:ind_x_end]
            refVolMask_bin = refVolMask_bin[ind_z_start:ind_z_end, ind_y_start:ind_y_end, ind_x_start:ind_x_end]
            transform = registerSubsetOfVolume(refVol_bin[ind_z_start:ind_z_end, ind_y_start:ind_y_end, ind_x_start:ind_x_end], vol_bin[ind_z_start:ind_z_end, ind_y_start:ind_y_end, ind_x_start:ind_x_end], refVolMask_bin, volMask_bin, pathSaveTransform, pathMoving, verbose = True, offset_it_number = i*nb_it, ind_z_slice_to_follow = ind_z_slice_to_follow//bin_factor, bin_factor = bin_factor, initial_transform = transform, **kwargs)
            center_x, center_y, center_z, center_angle = transform.GetFixedParameters()
            transform.SetFixedParameters((center_x+ind_x_start, center_y+ind_y_start, center_z+ind_z_start, center_angle)) # Not a function of offset_bottom
        else:
            transform = registerSubsetOfVolume(refVol_bin, vol_bin, refVolMask_bin, volMask_bin, pathSaveTransform, pathMoving, verbose = True, offset_it_number = i*nb_it, ind_z_slice_to_follow = ind_z_slice_to_follow//bin_factor, bin_factor = bin_factor, initial_transform = transform, **kwargs)

        # Modify transform to account for binning
        transform = modify_transform_binning_factor(transform, bin_factor)

        sitk.WriteTransform(transform, pathSaveTransform)


    # Setup saved vol

    tic =  time.time()
    volRegistered = applyTransformToVolume(refVol, vol, transform)
    toc =  time.time()
    print('Applying transform to volume, took '+str(toc-tic)+' s')

    print_transform_parameters(transform)

    if 'a' in save_str:
        File(prefix_out+'a_static/').saveTiffStack(refVol)
    if 'b' in save_str:
        File(prefix_out+'b_movingRegisteredToStatic/').saveTiffStack(volRegistered)
    if 'c' in save_str:
        File(prefix_out+'c_moving/').saveTiffStack(vol)
    if 'd' in save_str:
        File(prefix_out+'d_movingRegistered_minus_static/').saveTiffStack(volRegistered-refVol)

    return(volRegistered, transform)

def custom_3d_kernel_sphere(half_width):
    kernel_size = 2*half_width + 1
    kernel = np.zeros([kernel_size,kernel_size,kernel_size])

    one_in_center = np.zeros([kernel_size,kernel_size,kernel_size])
    one_in_center[int(kernel_size/2), int(kernel_size/2), int(kernel_size/2)] = 1.0
    kernel = scipy.ndimage.gaussian_filter(one_in_center,half_width/2)

    Y, X, Z = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
    dist_from_center = np.sqrt((X - kernel_size//2)**2 + (Y-kernel_size//2)**2 + (Z-kernel_size//2)**2)

    radius = half_width
    kernel[dist_from_center > radius] = 0.0
    max_value = np.max(kernel[dist_from_center <= radius])
    min_value = np.min(kernel[dist_from_center <= radius])
    kernel[dist_from_center <= radius] += (max_value-min_value)
    kernel /= np.sum(kernel)

    return(kernel)

def custom_3d_gaussian_filter(filter_half_width):

    # Window size of gaussian filter is ~3x the standard deviation sigma. See https://stackoverflow.com/questions/16165666/how-to-determine-the-window-size-of-a-gaussian-filter
    window_size_parameter = 3

    dirac_array = np.zeros([1+2*filter_half_width*window_size_parameter, 1+2*filter_half_width*window_size_parameter, 1+2*filter_half_width*window_size_parameter])
    dirac_array[filter_half_width*window_size_parameter,filter_half_width*window_size_parameter,filter_half_width*window_size_parameter] = 1.0
    gaussian_filter = scipy.ndimage.gaussian_filter(dirac_array, filter_half_width)

    # plt.imshow(gaussian_filter[filter_half_width])

    return(gaussian_filter)


class Data():
    def __init__(self):
        pass
    def __getstate__(self):
        return(self.__dict__)
    def __setstate__(self, d):
        self.__dict__ = d




