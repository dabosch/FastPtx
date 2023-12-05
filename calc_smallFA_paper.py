# %% import packages as needed
from  tools.plot3D import plot3D
from tools.trajectories import gtokTX, ktogTX, spins
from tools.io import readData, combineData, vectorizeData, \
    writePulse, readVOPfile, interpData
from tools.smallTipAngle import smallTipAngle
from tools.bloch import bloch
import numpy as np
import torch
import matplotlib.pyplot as plt 
import cv2
import os
import time

from scipy.io import loadmat,savemat

# %% set parameters

do_plot = True
# load one file (for tailored pulse) or multiple (for universal pulse, UP)
do_UP = False
nsteps = 52           # pulse duration = 10us * nsteps
maxSAR = 2.3          # SAR limit in W/kG
TR     = 20 * 1e-3    # TR in ms
targetFA = 10         # target FA in deg

# which device to run the code on, and which datatype to use
dev = torch.device('cuda')
dtc = torch.complex64
dt = torch.float32

# output file name
if do_UP:
    nIter = 7500
    pulsename = f'UP_20deg_{nsteps*10}us_16Tx31Rx'
else:
    nIter = 2500
    pulsename = f'TP_20deg_{nsteps*10}us'

# B1+ data file name
fname = ['./data/I3OS-QDEW.mat']

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

# VOPs for SAR calculation
vopfile = './data/RFSWDZZMat9001.bin.mat'

# downsampling data by factor of 2 can speed up the pulse calculation
downsample=False
Unorm = None
maskName = 'brainMask' # 'mask', 'tissueMask' # which mask to load from the data file

# %% system limits (SAR, maxRF, etc)
limits = smallTipAngle.getLimits()
limits['plotscale'] = targetFA * 1.2
limits['targetFA'] = targetFA
limits['maxSAR'] = maxSAR
limits['maxRF'] = 185
limits['sliceProfWeight'] = 0
#limits['maxG'] = 1e-12
limits['phaseOpt'] = False
limits['plotiter'] = 400
limits['dispiter'] = 40

# %% read data
start_time = time.time()
print('reading data')
dat = []
for f in fname:
    dat.append(readData(f,dt=dt,dev=dev,Unorm=Unorm,maskName=maskName))

VOPs = readVOPfile(vopfile,dev=dev,dtc=dtc)

# %% preprocess data
print('processing data')

# combine datasets (UP)
datComb = dat[0];
for i in range(1,len(dat)):
    datComb = combineData(datComb,dat[i])

if downsample:
    datComb = interpData(datComb,0.5)

dat_v = vectorizeData(datComb)

if False:
    # reduce number of Tx channels for faster tests
    dat_v['nTx'] = 2
    t = torch.ones([dat_v['nVox'],2],device=dev,dtype=dtc)
    t[:,0] = dat_v['s'][:,0::2].sum(1)
    t[:,1] = dat_v['s'][:,1::2].sum(1)
    dat_v['s'] = t


#%% define target
# Here any target can be defined

target = 1.0 * dat_v['mask3D'].clone()

if True: 
    target = target * targetFA
else: # TONE pulse (10..30 deg)
    l = torch.linspace(0,20,s2-s1,device=dev,dtype=dt)
    target[:,:,s1:s2] = target[:,:,s1:s2] * (l.unsqueeze(0).unsqueeze(0)+10)

if do_plot:
    f = plot3D(target.cpu())
    f.savefig('./img/target.png')

#%% set start value
stepdur=10e-6
TR=19.7e-3 #
try:
    # we can start optimization from a working pulse...
    mat = loadmat('_pulse_MPRAGE_400us.mat')
    g_start = torch.from_numpy(mat['g']).to(dev).to(dt)
    pulse_start = torch.from_numpy(mat['pulse']).to(dev).to(dtc)
    stepdur = mat['stepdur'][0,0]
    # fill zeros
    pulse_start = pulse_start.masked_fill(pulse_start==0,1e-2)
    g_start = g_start.masked_fill(g_start==0,1e-6)
except:
    # ... or we start optimization with default values
    print('couldnt read start values,starting with default values')
    g_start = None
    k = spins(nsteps,kmax=140,alpha=10,beta=0.5,u=8*3,v=5*3,dev=dat_v['s'].device)*0.5
    pulse_start = torch.ones(nsteps,16,device=dev,dtype=dtc) * 10.4 * 0.9
            # CP mode
    phs = torch.linspace(2*torch.pi/dat_v['nTx'],2*torch.pi,dat_v['nTx'],device=dev,dtype=dt)
    phs[::2] += -98/180*torch.pi
    pulse_start = pulse_start * torch.exp(1j * phs).reshape(1,-1)

# if FA > 30: design for small FA and up-scale later
largeFA = (target.max() > 30)
if largeFA:
    limits['maxSAR'] = limits['maxSAR']/100
    limits['maxRF'] = limits['maxRF']/15
    limits['targetFA'] = limits['targetFA']/10
    
    limits['plotscale'] = limits['plotscale']/10
    target = target/10
    if pulse_start is not None:
        pulse_start = pulse_start/10


# here we use the small tip angle approximation to optimize a pulse
pulse,g = smallTipAngle.optimize_RF_grad(dat_v,target,niter=nIter,nsteps=nsteps,pulse_start=pulse_start,
                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits,do_plot=do_plot)

# then refine with full bloch sim
pulse_start = pulse.clone().detach()
g_start = g.clone().detach()
if largeFA:
    target = target * 10
    pulse_start *= 10
    limits['maxSAR'] = limits['maxSAR']*100
    limits['maxRF'] = limits['maxRF']*15
    limits['plotscale'] = limits['plotscale']*10
    limits['targetFA'] = limits['targetFA'] * 10

# if we want to optimize an UP, do 2500 steps optimizing RF only
if do_UP:
    pulse,g = bloch.optimize_RF_only(dat_v,target,niter=2500,nsteps=nsteps,pulse_start=pulse_start,
                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits)

# %% export pulse file
k = gtokTX(g,stepdur)
mat = dict()
mat['pulse'] = pulse.cpu().numpy()
mat['g'] = g.cpu().numpy()
mat['k'] = k.cpu().numpy()
mat['stepdur'] = stepdur
orientation='unity'
elapsed = time.time() - start_time
print(f"elapsed time: {elapsed:.2f} sec")
print('writing pulse files')
writePulse(f'{pulsename}.ini', pulse.cpu()/targetFA, g.cpu(),orientation=orientation,stepdur=stepdur,FA=1)
print('writing mat')
savemat(f'{pulsename}.mat',mat)

# we can also transfer the pulse file to a remote server
if False:
    print('transferring pulse files')
    os.system(f'scp ./{pulsename}.ini root@mars:/opt/medcom/MriCustomer/seq/RFPulses/{pulsename}.ini')
    print('file copied successfully')

# %% Display pulse
SAR = smallTipAngle.calcPulseSAR(pulse, VOPs,TR=TR,stepdur=stepdur)
print(f'SAR = {SAR:6.3f}')
resvec = torch.cat((g.reshape(-1)/limits['maxG'],
                             (pulse/limits['maxRF']).real.reshape(-1),
                             (pulse/limits['maxRF']).imag.reshape(-1)))

if do_plot or True:
    bloch.figtitle = ''
    f = bloch.plotVec(pulse=pulse,g=g,dat_v=dat_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR);
    filename = f'img/final_{pulsename}.png'
    f.savefig(filename)
    filename = f'img/test.eps'
    f.savefig(filename, format='eps',bbox_inches='tight')
    f.savefig('img/test.svg', format='svg',bbox_inches='tight')
