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

def fast_pytorch_mask_erosion(mask, radius, use_pyTorch = True):
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

def fast_pytorch_bin_2d(img,bin_factor):
    kernel = torch.as_tensor(np.ones([1,1,bin_factor,bin_factor]).astype('float32')).cuda()
    kernel /= kernel.sum()
    
    img_tensor = torch.as_tensor(np.reshape(img, (1,1)+img.shape).astype('float32')).cuda()
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

def fast_pytorch_bin_3d(vol_stack,bin_factor, chunk_size = 4*68):
    try:
        N = vol_stack.shape[0]
        nb_chunk = N//chunk_size+1
        chunk_size = N//nb_chunk

        out_stack = []
        i1 = 0
        i2 = min(chunk_size, N)
        # print('.', end = '')

        kernel = torch.as_tensor(np.ones([1,1,bin_factor,bin_factor,bin_factor]).astype('float32')).cuda()
        kernel /= kernel.sum()

        for ind_chunk in range(nb_chunk):
            vol_stack_tensor = torch.as_tensor(np.reshape(vol_stack[i1:i2], (1,1)+vol_stack[i1:i2].shape).astype('float32')).cuda()
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
