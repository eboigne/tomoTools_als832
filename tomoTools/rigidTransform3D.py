import numpy as np
from math import sqrt
import os, sys
from scipy.ndimage import zoom

# from pystackreg import StackReg

script_path = '/run/media/eboigne/Data_Emeric/Scripts/Python/'
# script_path = '/home/ihme/eboigne/Python/'

path1='/run/media/eboigne/Data3_EB/XCT_BOS_CTC/190610_PMB_Fb_Fc_Fd_AvgAll/projSolid/solid_01_Fa_proj02_rec_testReg/'
path2='/run/media/eboigne/Data3_EB/XCT_BOS_CTC/190610_PMB_Fb_Fc_Fd_AvgAll/projSolid/solid_02_Fb_proj01_rec_testReg/'


sys.path.append(script_path)
sys.path.append(script_path+'utils/')
sys.path.append(script_path+'io/')

from File import *

import SimpleITK as sitk



def safe_copy_transform(transform):
    type_transform = type(transform)
    
    if type_transform == sitk.SimpleITK.Euler3DTransform or sitk.SimpleITK.Transform:
        transform2 = sitk.Euler3DTransform()
        transform2.SetFixedParameters(transform.GetFixedParameters())
        transform2.SetParameters(transform.GetParameters())
    elif type_transform == sitk.SimpleITK.Euler2DTransform:
        transform2 = sitk.Euler2DTransform()
        transform2.SetFixedParameters(transform.GetFixedParameters())
        transform2.SetParameters(transform.GetParameters())
    elif type_transform == sitk.TranslationTransform:
        transform2 = sitk.TranslationTransform(transform.GetDimension())
        transform2.SetParameters(transform.GetParameters())
        # No fixed parameters
    else:
        print('Error: safe_copy_transform of this type of transform is not implemented')
        transform2 = None
    return(transform2)


def print_transform_parameters(transform):
    type_transform = type(transform)
    print('Transformation of type: '+str(type_transform))
    if type_transform == sitk.SimpleITK.Euler3DTransform:
        center_x, center_y, center_z, center_angle = transform.GetFixedParameters()
        (eulerAngleX, eulerAngleY, eulerAngleZ, offsetX, offsetY, offsetZ) = transform.GetParameters()
        eulerAngleX *= 180/np.pi
        eulerAngleY *= 180/np.pi
        eulerAngleZ *= 180/np.pi
        print('\t Registration parameters:')
        print('Euler angle in degrees (X, Y, Z): ('+str(eulerAngleX)+', '+str(eulerAngleY)+', '+str(eulerAngleZ)+')')
        print('Offset in voxels (X, Y, Z): ('+str(offsetX)+', '+str(offsetY)+', '+str(offsetZ)+')')
        print('Center of rotation (X, Y, Z, A): ('+str(center_x)+', '+str(center_y)+', '+str(center_z)+', '+str(center_angle)+')')
    elif type_transform == sitk.SimpleITK.Euler2DTransform:
        (a,b,c) = transform.GetParameters() # 2 offsets, 1 angle, not sure which is which
        (center_x,center_y) = transform.GetFixedParameters()
        print('\t Registration parameters:')
        print('Parameters (2 offsets, 1 angle, not sure which is which): '+str(a)+', '+str(b)+', '+str(c))
        print('Center of rotation (X, Y): ('+str(center_x)+', '+str(center_y)+')')
    elif type_transform == sitk.TranslationTransform:
        print('\t Registration parameters:')
        if transform.GetDimension() == 3:
            (offsetX, offsetY, offsetZ) = transform.GetParameters()
            print('Offset in voxels (X, Y, Z): ('+str(offsetX)+', '+str(offsetY)+', '+str(offsetZ)+')')
        else: 
            (offsetX, offsetY) = transform.GetParameters()
            print('Offset in voxels (X, Y): ('+str(offsetX)+', '+str(offsetY)+')')
    else: 
        try:
            center_x, center_y, center_z, center_angle = transform.GetFixedParameters()
            (eulerAngleX, eulerAngleY, eulerAngleZ, offsetX, offsetY, offsetZ) = transform.GetParameters()
            eulerAngleX *= 180/np.pi
            eulerAngleY *= 180/np.pi
            eulerAngleZ *= 180/np.pi
            print('\t Registration parameters:')
            print('Euler angle in degrees (X, Y, Z): ('+str(eulerAngleX)+', '+str(eulerAngleY)+', '+str(eulerAngleZ)+')')
            print('Offset in voxels (X, Y, Z): ('+str(offsetX)+', '+str(offsetY)+', '+str(offsetZ)+')')
            print('Center of rotation (X, Y, Z, A): ('+str(center_x)+', '+str(center_y)+', '+str(center_z)+', '+str(center_angle)+')')
        except:
            print('\tPrinting this type of transform not implemented')
            print(transform.GetParameters())
            print(transform.GetFixedParameters())
        
        
        
        

def computeTransform(volFixed, volMoving, pathSaveTransform = '', bool_save = False, mask = [], pathMoving = '', initial_transform = '', shrinkFactors = [], smoothingSigmas = [], verbose = False, ind_z_slice_to_follow = 0, offset_it_number = 0, bin_factor = 1, **kwargs):

    def save_combined_central_slice(fixed, moving, transform, file_name_prefix):
        global iteration_number

        central_indexes = [int(i/2) for i in fixed.GetSize()]
        
        if ind_z_slice_to_follow != 0:
            central_indexes[2] = ind_z_slice_to_follow
        moving_transformed = sitk.Resample(moving, fixed, transform,
                                                 sitk.sitkBSpline,# interpolator: sitk.sitkLinear / sitk.sitkBSpline
                                           0.0, moving.GetPixelIDValue()) # default pixel value, pixel type
        # fixedCombined = [fixed[:,:,central_indexes[2]], fixed[:,central_indexes[1],:], fixed[central_indexes[0],:,:]]
        movingCombined = [moving_transformed[:,:,central_indexes[2]], moving_transformed[:,central_indexes[1],:], moving_transformed[central_indexes[0],:,:]]
        for ind, img in enumerate(movingCombined):
            img_numpy = sitk.GetArrayFromImage(img)
            img_numpy = zoom(img_numpy, (bin_factor, bin_factor))
            movingCombined[ind] = sitk.GetImageFromArray(img_numpy)

        save_name = file_name_prefix+'/Moving_'+str(iteration_number+offset_it_number).zfill(2)+'.tif'
        sitk.WriteImage(sitk.Tile(movingCombined, (1,3)), save_name)

        iteration_number+=1


    def printIteration(method):
        transform = method.GetInitialTransform()

        if (method.GetOptimizerIteration() == 0):
            print("\n\tOffset iteration number: {0}".format(offset_it_number))
            print("\tLevel: {0}".format(method.GetCurrentLevel()))
            print("\tScales: {0}".format(method.GetOptimizerScales()))
            print('Fixed parameters (image center): '+str(transform.GetFixedParameters()))
        print('\nIteration: '+str(method.GetOptimizerIteration())+' - Cost function: '+str(method.GetMetricValue())+' - LearningRate: '+str(method.GetOptimizerLearningRate()))
        print_transform_parameters(transform)

    #read the images
    fixed_image = sitk.GetImageFromArray(volFixed.astype('float32'))
    moving_image = sitk.GetImageFromArray(volMoving.astype('float32'))

    # Initialize with geometry / moments
    initialTransform = sitk.Euler3DTransform()
    
    transform = sitk.CenteredTransformInitializer(fixed_image,
                                                  moving_image,
                                                  initialTransform,
                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    
    if type(initial_transform) != type(''):
        transform.SetParameters(initial_transform.GetParameters())
        transform.SetFixedParameters(initial_transform.GetFixedParameters())

    #
    # print('\tInitial transform parameters:')
    # (eulerAngleX, eulerAngleY, eulerAngleZ, offsetX, offsetY, offsetZ) = transform.GetParameters()
    # eulerAngleX *= 180/np.pi
    # eulerAngleY *= 180/np.pi
    # eulerAngleZ *= 180/np.pi
    # # (eulerAngleX, eulerAngleY, eulerAngleZ) = 180/np.pi * (eulerAngleX, eulerAngleY, eulerAngleZ)
    # print('\t Registration parameters:')
    # print('Euler angle in degrees (X, Y, Z): ('+str(eulerAngleX)+', '+str(eulerAngleY)+', '+str(eulerAngleZ)+')')
    # print('Offset in voxels (X, Y, Z): ('+str(offsetX)+', '+str(offsetY)+', '+str(offsetZ)+')')

    # Initialize with identity
    # transform = sitk.Euler3DTransform()

    # Set initial transform
    # transform = sitk.ReadTransform('/run/media/eboigne/Data2_EB/als/simpleItkTest/transform2_inv.tfm')
    # transform = sitk.ReadTransform('/run/media/eboigne/Data2_EB/als/simpleItkTest/transform2.tfm')


    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsCorrelation()
    
    if len(mask) > 0 and not isinstance(mask, sitk.Image):
        sitk_Image = sitk.GetImageFromArray(mask.astype('int'))
        registration_method.SetMetricFixedMask(sitk_Image)

    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR) # Same as random, but with fixed seed
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetMetricSamplingPercentage(1)

    registration_method.SetInterpolator(sitk.sitkLinear) # See https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
    



    # registration_method.SetOptimizerAsGradientDescent(learningRate=0.1,
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=kwargs['learningRate'],
                                                      numberOfIterations=kwargs['numberOfIterations'],
                                                      convergenceMinimumValue=kwargs['convergenceMinimumValue'], # Threshold below which iterations stop
                                                      convergenceWindowSize=kwargs['convergenceWindowSize']) 
    
#     registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=kwargs['learningRate'],
#                                                       numberOfIterations=kwargs['numberOfIterations'],
#                                                       convergenceMinimumValue=kwargs['convergenceMinimumValue'], # Threshold below which iterations stop
#                                                       convergenceWindowSize=kwargs['convergenceWindowSize']) # Nb of last iteration used to compute the convergence minimum value


    # registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate = 0.1, minStep = 1e-5000,
    #                                                              relaxationFactor = 0.9,
    #                                                              gradientMagnitudeTolerance = 1e-5000,
    #                                                              numberOfIterations = 50)
    # See doc on optimizer at ITK: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html
    # And https://simpleitk.github.io/SPIE2018_COURSE/basic_registration.pdf
    # https://simpleitk.github.io/SPIE2018_COURSE/advanced_registration.pdf
    # And see: https://github.com/SimpleITK/SimpleITK/blob/c71be30fd0da5ab8cf3dda41675798f844c233e3/Code/Registration/include/sitkImageRegistrationMethod.h


    registration_method.SetOptimizerScalesFromPhysicalShift() # Weight the learning rate over different parameters (especially offsets vs angle) using a notion of "physical shift".
    if shrinkFactors == []:
        shrinkFactors = (shrink_coeff * np.array([4,4,4,4,2,2,2,1,1])).astype('int32').tolist()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = shrinkFactors) # At each level, downsample in isotropic way by the given factor: 1 yiels original resolution.
    if smoothingSigmas == []:
        smoothingSigmas = [2,2,2,2,1,1,1,0,0]
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothingSigmas) # At each level, a Gaussian filter is used to smooth the two fields (moved and fixed) with given sigma.
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn() # Off for voxel units.
    
    registration_method.SetInitialTransform(transform)

    #add iteration callback, save central slice in xy, xz, yz planes
    global iteration_number
    iteration_number = 0

    if bool_save:
        registration_method.AddCommand(sitk.sitkIterationEvent,lambda: save_combined_central_slice(fixed_image, moving_image,transform,pathMoving))
                                                                           # '/run/media/eboigne/Data2_EB/als/simpleItkTest/output/'))
    if verbose == True:
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: printIteration(registration_method))
        print('Initial cost function: '+str(registration_method.GetMetricValue()))

    registration_method.Execute(fixed_image, moving_image)

    if pathSaveTransform != '':
        sitk.WriteTransform(transform, pathSaveTransform)

    return(transform)



def computeTransformTranslation(volFixed, volMoving, pathSaveTransform = '', bool_save = False, mask = [], pathMoving = '', initial_transform = '', verbose = False, **kwargs):

    def save_combined_central_slice(fixed, moving, transform, file_name_prefix):
        global iteration_number

        central_indexes = [int(i/2) for i in fixed.GetSize()]
    
        moving_transformed = sitk.Resample(moving, fixed, transform,
                                                 sitk.sitkBSpline,# interpolator: sitk.sitkLinear / sitk.sitkBSpline
                                           0.0, moving.GetPixelIDValue()) # default pixel value, pixel type

        # fixedCombined = [fixed[:,:,central_indexes[2]], fixed[:,central_indexes[1],:], fixed[central_indexes[0],:,:]]
        movingCombined = [moving_transformed[:,:,central_indexes[2]], moving_transformed[:,central_indexes[1],:], moving_transformed[central_indexes[0],:,:]]
        #
        # sitk.WriteImage(sitk.Tile(fixedCombined, (1,3)),
        #                 file_name_prefix+'Fixed/Fixed_'+ format(iteration_number, '03d') + '.tif')

        save_name = file_name_prefix+'/Moving_'+str(iteration_number).zfill(2)+'.tif'
        sitk.WriteImage(sitk.Tile(movingCombined, (1,3)), save_name)

        iteration_number+=1


    def printIteration(method):
        transform = method.GetInitialTransform()

        if (method.GetOptimizerIteration() == 0):
            print("\tLevel: {0}".format(method.GetCurrentLevel()))
            print("\tScales: {0}".format(method.GetOptimizerScales()))
            print('Fixed parameters (image center): '+str(transform.GetFixedParameters()))
        print('Iteration: '+str(method.GetOptimizerIteration())+' - Cost function: '+str(method.GetMetricValue())+' - LearningRate: '+str(method.GetOptimizerLearningRate()))
        print_transform_parameters(transform)

    #read the images
    fixed_image = sitk.GetImageFromArray(volFixed.astype('float32'))
    moving_image = sitk.GetImageFromArray(volMoving.astype('float32'))

    if type(initial_transform) != type(''):
        transform = initial_transform
    else:
        transform = sitk.TranslationTransform(fixed_image.GetDimension())

    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsCorrelation()

    if len(mask) > 0 and not isinstance(mask, sitk.Image):
        registration_method.SetMetricFixedMask(mask)

    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR) # Same as random, but with fixed seed
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetMetricSamplingPercentage(1)

    registration_method.SetInterpolator(sitk.sitkLinear) # See https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
    



    # registration_method.SetOptimizerAsGradientDescent(learningRate=0.1,
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=kwargs['learningRate'],
                                                      numberOfIterations=kwargs['numberOfIterations'],
                                                      convergenceMinimumValue=kwargs['convergenceMinimumValue'], # Threshold below which iterations stop
                                                      convergenceWindowSize=kwargs['convergenceWindowSize']) # Nb of last iteration used to compute the convergence minimum value

    # registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate = 0.1, minStep = 1e-5000,
    #                                                              relaxationFactor = 0.9,
    #                                                              gradientMagnitudeTolerance = 1e-5000,
    #                                                              numberOfIterations = 50)
    # See doc on optimizer at ITK: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html
    # And https://simpleitk.github.io/SPIE2018_COURSE/basic_registration.pdf
    # https://simpleitk.github.io/SPIE2018_COURSE/advanced_registration.pdf
    # And see: https://github.com/SimpleITK/SimpleITK/blob/c71be30fd0da5ab8cf3dda41675798f844c233e3/Code/Registration/include/sitkImageRegistrationMethod.h

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1]) # At each level, downsample in isotropic way by the given factor: 1 yiels original resolution.
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0]) # At each level, a Gaussian filter is used to smooth the two fields (moved and fixed) with given sigma.
    registration_method.SetInitialTransform(transform)

    #add iteration callback, save central slice in xy, xz, yz planes
    global iteration_number
    iteration_number = 0

    if bool_save:
        registration_method.AddCommand(sitk.sitkIterationEvent,lambda: save_combined_central_slice(fixed_image, moving_image,transform,pathMoving))
                                                                           # '/run/media/eboigne/Data2_EB/als/simpleItkTest/output/'))

    if verbose == True:
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: printIteration(registration_method))
        print('Initial cost function: '+str(registration_method.GetMetricValue()))

    registration_method.Execute(fixed_image, moving_image)

    if pathSaveTransform != '':
        sitk.WriteTransform(transform, pathSaveTransform)

    return(transform)


def computeTransform2D(volFixed, volMoving, pathSaveTransform = '', bool_save = False, mask = [], pathMoving = '', initial_transform = '', verbose = False, **kwargs):

    def printIteration(method):
        transform = method.GetInitialTransform()

        if (method.GetOptimizerIteration() == 0):
            print("\tLevel: {0}".format(method.GetCurrentLevel()))
            print("\tScales: {0}".format(method.GetOptimizerScales()))
            print('Fixed parameters (image center): '+str(transform.GetFixedParameters()))
        print('Iteration: '+str(method.GetOptimizerIteration())+' - Cost function: '+str(method.GetMetricValue())+' - LearningRate: '+str(method.GetOptimizerLearningRate()))
        print_transform_parameters(transform)

    #read the images
    fixed_image = sitk.GetImageFromArray(volFixed.astype('float32'))
    moving_image = sitk.GetImageFromArray(volMoving.astype('float32'))

    # Initialize with geometry / moments
    initialTransform = sitk.Euler2DTransform()
    
    transform = sitk.CenteredTransformInitializer(fixed_image,
                                                  moving_image,
                                                  initialTransform,
                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    
    if type(initial_transform) != type(''):
        transform.SetParameters(initial_transform.GetParameters())

    #
    # print('\tInitial transform parameters:')
    # (eulerAngleX, eulerAngleY, eulerAngleZ, offsetX, offsetY, offsetZ) = transform.GetParameters()
    # eulerAngleX *= 180/np.pi
    # eulerAngleY *= 180/np.pi
    # eulerAngleZ *= 180/np.pi
    # # (eulerAngleX, eulerAngleY, eulerAngleZ) = 180/np.pi * (eulerAngleX, eulerAngleY, eulerAngleZ)
    # print('\t Registration parameters:')
    # print('Euler angle in degrees (X, Y, Z): ('+str(eulerAngleX)+', '+str(eulerAngleY)+', '+str(eulerAngleZ)+')')
    # print('Offset in voxels (X, Y, Z): ('+str(offsetX)+', '+str(offsetY)+', '+str(offsetZ)+')')

    # Initialize with identity
    # transform = sitk.Euler3DTransform()

    # Set initial transform
    # transform = sitk.ReadTransform('/run/media/eboigne/Data2_EB/als/simpleItkTest/transform2_inv.tfm')
    # transform = sitk.ReadTransform('/run/media/eboigne/Data2_EB/als/simpleItkTest/transform2.tfm')


    # print type(transform)
    # print type(initialTransform)

    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsCorrelation()

    if len(mask) > 0 and not isinstance(mask, sitk.Image):
        registration_method.SetMetricFixedMask(mask)

    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR) # Same as random, but with fixed seed
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetMetricSamplingPercentage(1)

    registration_method.SetInterpolator(sitk.sitkLinear) # See https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
    


    # registration_method.SetOptimizerAsGradientDescent(learningRate=0.1,
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=kwargs['learningRate'],
                                                      numberOfIterations=kwargs['numberOfIterations'],
                                                      convergenceMinimumValue=kwargs['convergenceMinimumValue'], # Threshold below which iterations stop
                                                      convergenceWindowSize=kwargs['convergenceWindowSize']) # Nb of last iteration used to compute the convergence minimum value


    # registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate = 0.1, minStep = 1e-5000,
    #                                                              relaxationFactor = 0.9,
    #                                                              gradientMagnitudeTolerance = 1e-5000,
    #                                                              numberOfIterations = 50)
    # See doc on optimizer at ITK: https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html
    # And https://simpleitk.github.io/SPIE2018_COURSE/basic_registration.pdf
    # https://simpleitk.github.io/SPIE2018_COURSE/advanced_registration.pdf
    # And see: https://github.com/SimpleITK/SimpleITK/blob/c71be30fd0da5ab8cf3dda41675798f844c233e3/Code/Registration/include/sitkImageRegistrationMethod.h


    registration_method.SetOptimizerScalesFromPhysicalShift() # Weight the learning rate over different parameters (especially offsets vs angle) using a notion of "physical shift".
    
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [2,1,1]) # At each level, downsample in isotropic way by the given factor: 1 yiels original resolution.
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1,0,0]) # At each level, a Gaussian filter is used to smooth the two fields (moved and fixed) with given sigma.
    
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,4,4,4,2,2,2,1,1]) # At each level, downsample in isotropic way by the given factor: 1 yiels original resolution.
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,2,2,2,1,1,1,0,0]) # At each level, a Gaussian filter is used to smooth the two fields (moved and fixed) with given sigma.
    
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn() # Off for voxel units.
    
    registration_method.SetInitialTransform(transform)

    #add iteration callback, save central slice in xy, xz, yz planes
    global iteration_number
    iteration_number = 0

    if verbose == True:
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: printIteration(registration_method))
        print('Initial cost function: '+str(registration_method.GetMetricValue()))

    registration_method.Execute(fixed_image, moving_image)

    if pathSaveTransform != '':
        sitk.WriteTransform(transform, pathSaveTransform)

    return(transform)



def applyTransformToVolume(volFixed, volMoving, transform):


    if type(volFixed) == np.ndarray:
        fixed = sitk.GetImageFromArray(volFixed.astype('float32'))
    else:
        fixed = volFixed
    if type(volMoving) == np.ndarray:
        moving = sitk.GetImageFromArray(volMoving.astype('float32'))
    else:
        moving = volMoving

    moving_transformed = sitk.Resample(moving, fixed, transform,
                                             sitk.sitkLinear,# interpolator: sitk.sitkLinear / sitk.sitkBSpline
                                       0.0, moving.GetPixelIDValue()) # default pixel value, pixel type
    # Interpolator: https://simpleitk.org/doxygen/latest/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5

    return(sitk.GetArrayFromImage(moving_transformed))


if __name__ == '__main__':

    folder1 = File(path1)
    folder2 = File(path2)

    A = folder1.readAll()
    B = folder2.readAll()

    transform = computeTransform(A, B)

    print(transform)

    tifffile.imsave('test1.tif', A.astype('float32'))
    tifffile.imsave('test2.tif', B.astype('float32'))
    tifffile.imsave('test3.tif',applyTransformToVolume(A,B,transform).astype('float32'))
