from  tools.plot3D import plot3D
from tools.trajectories import gtokTX, ktogTX, spins
from tools.io import readData, combineData, vectorizeData, \
    writePulse, readVOPfile, interpData
#from tools.spatialDomainMethod import optimize_RF_grad
from tools.smallTipAngle import smallTipAngle
from tools.bloch import bloch
import numpy as np
import torch
import matplotlib.pyplot as plt 
from matplotlib import image
import cv2
#from tools.bloch import bloch
#import tools.bloch as bloch
import os
import time

from scipy.io import loadmat,savemat

start_time = time.time()

#imname = './data/EW4G-63IR.png'
pulsename = "db_TFLb1"

do_plot = True
do_UP = False

if do_UP:
    nIter = 7500
else:
    nIter = 3500

fname = ['./data/4VRE-G6Q4.mat']
#fname = ['./data/currentData.mat']

if do_UP:
    fname = []
    fname.append('./data/EBNY-7PGC.mat')
    fname.append('./data/ILMH-BG6A.mat')
    fname.append('./data/IPPR-4F37.mat')
    fname.append('./data/JPY5-SWAO.mat')
    fname.append('./data/KYT2-DQX6.mat')
    fname.append('./data/M75A-FAX5.mat')
    fname.append('./data/EW4G-63IR.mat')
    fname.append('./data/IY5C-GQWQ.mat')
    fname.append('./data/DJOY-ISET.mat')

vopfile = './data/RFSWDZZMat9001.bin.mat'

downsample=False
Unorm = None
sphericalROI = False
maskName = 'brainMask'
#maskName = 'mask'
#maskName = 'tissueMask'

dev = torch.device('cuda:1')
dtc = torch.complex64
dt = torch.float32

# for circular mask
ctr = [32,32,40]
r = 8

# %% read data
print('reading data')
dat = []
for f in fname:
    dat.append(readData(f,dt=dt,dev=dev,Unorm=Unorm,maskName=maskName))

VOPs = readVOPfile(vopfile,dev=dev,dtc=dtc)

# %% preprocess data (fix mask)
for d in dat:
    if sphericalROI:
        # do spherical ROI
        def create_circular_mask(w,h,d, center=None, radius=None):
            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2), d//2)
            if radius is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], center[2], w-center[0], h-center[1],d-center[2])

            Y, X, Z = np.ogrid[:h, :w, :d]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)

            mask = dist_from_center <= radius
            return mask
    
        d['mask'][ctr[0]-r-2:ctr[0]+r+2,ctr[1]-r-2:ctr[1]+r+2,ctr[2]-r-2:ctr[2]+r+2] = 0
        d['mask'][ctr[0]-r-2:ctr[0]+r+2,ctr[1]-r-2:ctr[1]+r+2,ctr[2]-r-2:ctr[2]+r+2] = torch.tensor(create_circular_mask(2*r+4,2*r+4,2*r+4,radius=r),device=dev)
    pass
#dat['mask'][:,:,45:] = 0
print('processing data')
#dat['s'] = dat['s'] #* (223/365)

# combine datasets (UP)
datComb = dat[0];
for i in range(1,len(fname)):
    datComb = combineData(datComb,dat[i])

if downsample:
    datComb = interpData(datComb,0.5)

dat_v = vectorizeData(datComb)

if False:
    # reduce number of Tx channels
    dat_v['nTx'] = 2
    t = torch.ones([dat_v['nVox'],2],device=dev,dtype=dtc)
    t[:,0] = dat_v['s'][:,0::2].sum(1)
    t[:,1] = dat_v['s'][:,1::2].sum(1)
    dat_v['s'] = t


#%% define target

target = 1.0 * dat_v['mask3D'].clone()

s1 = 35
s2 = 50
if downsample:
    s1=int(s1/2)
    s2=int(s2/2)

# target[:,:,:s1] = 0
# target[:,:,s1] *= 0.5
# target[:,:,s2] *= 0.5
# target[:,:,s2+1:] = 0

if False:
    # do image
    image = image.imread(imname).astype(np.float32)
    image = image[...,:3].sum(2)
    image = torch.from_numpy(image).to(target.device).rot90(1).flip(0)
    image /= image.max()
    for z in range(target.shape[2]):
        target[:,:,z] *= image

if sphericalROI:
    # spherical ROI
    target = create_circular_mask(target.shape[0],target.shape[1],target.shape[2],center=ctr,radius=r)
    target = torch.from_numpy(target).to(dev)
        
targetFA = 90
if True: 
    target = target * targetFA
else: # TONE pulse (10..30 deg)
    l = torch.linspace(0,20,s2-s1,device=dev,dtype=dt)
    target[:,:,s1:s2] = target[:,:,s1:s2] * (l.unsqueeze(0).unsqueeze(0)+10)

if do_plot:
    f = plot3D(target.cpu(),pos=[32,32,42])
    f.savefig('./img/target.png')

#%% set start value
stepdur=20e-6
nsteps = 100
TR = 1 # TFL
try:
    mat = loadmat(f'_{pulsename}.mat')
    g_start = torch.from_numpy(mat['g']).to(dev).to(dt)
    pulse_start = torch.from_numpy(mat['pulse']).to(dev).to(dtc)
    stepdur = mat['stepdur'][0,0]
    # fill zeros
    pulse_start = pulse_start.masked_fill(pulse_start==0,1e-2)
    g_start = g_start.masked_fill(g_start==0,1e-6)
        
except:
    print('couldnt read file,starting with default values')
    g_start = None
    k = spins(nsteps,kmax=140,alpha=10,beta=0.5,u=8*3,v=5*3,dev=dat_v['s'].device)*0.5
    #g_start = ktogTX(k,stepdur)
    #g_start = torch.randn(3,nsteps,device=dev,dtype=dt) * 0.0001
    #pulse_start = None
    pulse_start = torch.ones(nsteps,16,device=dev,dtype=dtc) * targetFA / (nsteps*stepdur) / 2000 * 1.2
            # CP mode
    phs = torch.linspace(2*torch.pi/dat_v['nTx'],2*torch.pi,dat_v['nTx'],device=dev,dtype=dt)
    phs[::2] += -98/180*torch.pi
    pulse_start = pulse_start * torch.exp(1j * phs).reshape(1,-1)
    

torch.autograd.set_detect_anomaly(False)


# system limits (SAR, maxRF, etc)
limits = smallTipAngle.getLimits()
limits['plotscale'] = targetFA * 1.2
limits['targetFA'] = targetFA
limits['maxSAR'] = 9.5
limits['maxRF'] = 170
limits['maxSlew'] = 190 # T/m/s
limits['sliceProfWeight'] = 0
#limits['maxG'] = 1e-12
limits['phaseOpt'] = False
largeFA = (target.max() > 30)
# if FA > 30: design for small FA and up-scale
limits['plotiter'] = 400
limits['dispiter'] = 40

if largeFA:
    limits['maxSAR'] = limits['maxSAR']/100
    limits['maxRF'] = limits['maxRF']/12
    limits['targetFA'] = limits['targetFA']/10
    
    limits['plotscale'] = limits['plotscale']/10
    target = target/10
    if pulse_start is not None:
        pulse_start = pulse_start/10



# first optimize with spatial domain method since it's fast
pulse,g = smallTipAngle.optimize_RF_grad(dat_v,target,niter=nIter,nsteps=nsteps,pulse_start=pulse_start,
                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits,do_plot=do_plot)


if False:
    rep = int(stepdur/10e-6);
    #gout = torch.zeros(3,rep*g.shape[1])
    g = torch.nn.functional.interpolate(g.unsqueeze(1),scale_factor=rep, mode='linear',align_corners=True).squeeze()
    pouti = torch.nn.functional.interpolate(pulse.imag.T.unsqueeze(1),scale_factor=rep, mode='linear',align_corners=True).squeeze().T
    poutr = torch.nn.functional.interpolate(pulse.real.T.unsqueeze(1),scale_factor=rep, mode='linear',align_corners=True).squeeze().T
    pulse = poutr + 1j * pouti
    stepdur = stepdur / rep

# then refine with full bloch sim
pulse_start = pulse.clone().detach()
g_start = g.clone().detach()
if largeFA:
    target = target * 10
    pulse_start *= 10
    limits['maxSAR'] = limits['maxSAR']*100
    limits['maxRF'] = limits['maxRF']*12
    limits['plotscale'] = limits['plotscale']*10
    limits['targetFA'] = limits['targetFA'] * 10


if do_UP:
    pulse,g = bloch.optimize_RF_grad(dat_v,target,niter=2500,nsteps=nsteps,pulse_start=pulse_start,
                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits)
if largeFA:
    pulse,g = bloch.optimize_RF_grad(dat_v,target,niter=1000,nsteps=nsteps,pulse_start=pulse_start,
                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits)
if sphericalROI:
    pulse,g = bloch.optimize_RF_grad(dat_v,target,niter=200,nsteps=nsteps,pulse_start=pulse_start,
                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits)
#%%

k = gtokTX(g,stepdur)

#%% export pulse file
mat = dict()
mat['pulse'] = pulse.cpu().numpy()
mat['g'] = g.cpu().numpy()
mat['k'] = k.cpu().numpy()
mat['stepdur'] = stepdur
orientation='unity'
elapsed = time.time() - start_time
print(f"elapsed time: {elapsed:.2f} sec")
print('writing pulse files')
writePulse(f'{pulsename}.ini', pulse.cpu(), g.cpu(),orientation=orientation,stepdur=stepdur,FA=targetFA)
print('writing mat')
savemat(f'{pulsename}.mat',mat)

if True:
    print('transferring pulse files')
    os.system(f'scp ./{pulsename}.ini root@mars:/opt/medcom/MriCustomer/seq/RFPulses/{pulsename}.ini')
    print('file copied successfully')

#%%
SAR = smallTipAngle.calcPulseSAR(pulse, VOPs,TR=TR,stepdur=stepdur)
print(f'SAR = {SAR:6.3f}')
#%%
resvec = torch.cat((g.reshape(-1)/limits['maxG'],
                             (pulse/limits['maxRF']).real.reshape(-1),
                             (pulse/limits['maxRF']).imag.reshape(-1)))

#calcErr(dat_v,bestvec,target_v,stepdur,verbose=True,VOPs=VOPs,TR=TR)
#optvec = bestvec.clone().requires_grad_(True)
#spatialDomainMethod.plotVec(bestvec,dat_v,stepdur)
#plotVec(load,dat_v,stepdur)
if do_plot:
    f = bloch.plotVec(pulse=pulse,g=g,dat_v=dat_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR);
    filename = f'img/final_{pulsename}.png'
    f.savefig(filename)
    plt.close()

#torch.save(bestvec,'pulse.dat')

#pulse,g,k = splitVec(bestvec,stepdur)
#pulse_= pulse.reshape(dat_v['nTx'],-1)
#writePulse('./test.ini',pulse_,g,FA=7,stepdur=stepdur)
