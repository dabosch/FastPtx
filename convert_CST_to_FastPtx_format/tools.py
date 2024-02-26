import numpy as np
import h5py
import SimpleITK as sitk
from scipy.io import loadmat


def coreg_masks(cor_mask,tissue,brain,b0=None):
    fixed = sitk.GetImageFromArray(cor_mask * 1.0)
    moving = sitk.GetImageFromArray(tissue * 1.0)
    
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkBSpline)
    registration_method.SetOptimizerAsRegularStepGradientDescent(10, 1e-3, 20000, 0.5, 1e-9)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # initialise transformation
    initial_transform = sitk.CenteredTransformInitializer(fixed, 
                                                          moving, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed, moving)
    
    moving_t = sitk.Resample(moving, final_transform, interpolator=sitk.sitkLinear)
    
    tissue_mask = sitk.GetArrayFromImage(moving_t)
    
    tissue_mask = tissue_mask[:cor_mask.shape[0],:cor_mask.shape[1],:cor_mask.shape[2]]
    
    
    print(brain.shape)
    brain_itk = sitk.GetImageFromArray(brain)
    brain_mask = sitk.GetArrayFromImage(sitk.Resample(brain_itk,final_transform,interpolator=sitk.sitkLinear))
    
    brain_mask = brain_mask[:cor_mask.shape[0],:cor_mask.shape[1],:cor_mask.shape[2]]
    
    if b0 is not None:
        b0_itk = sitk.GetImageFromArray(b0)
        deltaB0_Hz = sitk.GetArrayFromImage(sitk.Resample(b0_itk,final_transform,interpolator=sitk.sitkLinear))
    
        deltaB0_Hz = deltaB0_Hz[:cor_mask.shape[0],:cor_mask.shape[1],:cor_mask.shape[2]]
    else:
        deltaB0_Hz = np.zeros(cor_mask.shape)
    return tissue_mask,brain_mask,deltaB0_Hz


def readData(fpath,nCh=8):
    with h5py.File(f'{fpath}/SAR_CP_Mode.h5', "r") as f:
        stepSize = np.zeros(3)
        stepSize[0] = f['Mesh line x'][()][1]-f['Mesh line x'][()][0]
        stepSize[1] = f['Mesh line y'][()][1]-f['Mesh line y'][()][0]
        stepSize[2] = f['Mesh line z'][()][1]-f['Mesh line z'][()][0]
        SAR = f['SAR'][()]
        cor_mask = SAR>0

    hx = []
    for i in range(nCh):
        print(f'reading Ch {i}')
        with h5py.File(f'{fpath}/H_Ch{i+1}.h5', "r") as f:
            #hx[:,:,:,0] = f['H-Field']['x']['re'][()] + 1j * f['H-Field']['x']['im'][()] + 1j
            if i==0:
                sh = [i for i in f['H-Field']['x']['re'][()].shape]
                sh.append(nCh)
                hx = np.zeros(sh,dtype=np.complex64)
                hy = np.zeros(sh,dtype=np.complex64)
                hz = np.zeros(sh,dtype=np.complex64)
                
            hx[:,:,:,i] = f['H-Field']['x']['re'][()] + 1j * f['H-Field']['x']['im'][()]
            hy[:,:,:,i] = f['H-Field']['y']['re'][()] + 1j * f['H-Field']['y']['im'][()]
            hz[:,:,:,i] = f['H-Field']['z']['re'][()] + 1j * f['H-Field']['z']['im'][()]
    # Calculate Bx and By and convert from 0.5W RMS to V at plug level
    mu0=1.25663706212e-6 # magnetic field constant
    Bx = mu0 * hx / 5
    By = mu0 * hy / 5
    B_plus  = ( Bx + 1j*By )/2

    print('loading B0')
    dk = loadmat(f'inputData/Duke_400_b0.mat')
    duke = dk['tissueMask']
    brain = dk['brainMask']
    b0 = dk['deltaB0_Hz']
    # remove spinal cord
    brain[:,:,:85] *= 0

    print('starting coregistration')
    tissue_mask,brain_mask,deltaB0_Hz = coreg_masks(cor_mask,duke,brain,b0)

    # get ready for returning
    cor_mask = np.transpose(cor_mask,[2,1,0])
    brain_mask = np.transpose(brain_mask,[2,1,0])
    B_plus = np.transpose(B_plus,[2,1,0,3])
    deltaB0_Hz = np.transpose(deltaB0_Hz,[2,1,0])
    stepSize = stepSize[[2,1,0]]
    m = {"B_plus":B_plus,"stepSize":stepSize,"cor_mask":cor_mask,"brain_mask":brain_mask,"deltaB0_Hz":deltaB0_Hz}
    return m