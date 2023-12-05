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

start_time = time.time()

imname = './circle.png'

do_plot = True
do_UP = False

if do_UP:
    nIter = 20000
    pulsename = 'slisel'
else:
    nIter = 9000
    pulsename = 'db_GRE'
    
#%% define excitation slab
pos = [0.0,0.0,-0.035] # m
thick = .05 # m
rotx = 45 # deg

fname = ['./data/EW4G-63IR.mat']
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
maskName = 'brainMask'
#maskName = 'mask'
#maskName = 'tissueMask'

dev = torch.device('cuda:1')
dtc = torch.complex64
dt = torch.float32

# %% read data
print('reading data')
dat = []
dat2 = []
for f in fname:
    dat.append(readData(f,dt=dt,dev=dev,Unorm=Unorm,maskName=maskName))
    dat2.append(readData(f,dt=dt,dev=dev,Unorm=Unorm,maskName='tissueMask'))

VOPs = readVOPfile(vopfile,dev=dev,dtc=dtc)

# %% preprocess data (fix mask)
#dat['mask'][:,:,:39] = 0
#dat['mask'][:,:,45:] = 0
print('processing data')
#dat['s'] = dat['s'] #* (223/365)

for i in range(len(dat)):
    print(f"Uref {i:2}: {dat[i]['meta']['Uref']}")

# combine datasets (UP)
# datComb = dat[0];
# for i in range(1,len(dat)):
#     datComb = combineData(datComb,dat[i])

# if downsample:
#     datComb = interpData(datComb,0.5)

    

def cosd(x):
    return np.cos(np.deg2rad(x))    
def sind(x):
    return np.sin(np.deg2rad(x))

# slab = torch.zeros(dat[0]['mask'].shape,device=dat[0]['mask'].device)
# nGap = 1.5
# for y in range(slab.shape[1]):
#     for z in range(slab.shape[2]):
#         dist = abs(cosd(rotx)*(z-ctr[1]) - sind(rotx)*(y-ctr[0]))
#         if dist < (thick/2 / dat[0]['meta']['voxSz'][1]):
#             slab[:,y,z] = 1.0
#         if dist < (thick/2 / dat[0]['meta']['voxSz'][1])+nGap and dist > (thick/2 / dat[0]['meta']['voxSz'][1])-nGap :
#             dat[0]['mask'][:,y,z] = 0
# plot3D(dat[0]['mask'].cpu())
#n=0
for n in range(len(dat)):
    ctr = dat[n]['meta']['isocenter'][1:3] + pos[1:3]/dat[n]['meta']['voxSz'][1:3]
    slab = torch.zeros(dat[n]['mask'].shape,device=dat[n]['mask'].device)
    nGap = 1.5
    for y in range(slab.shape[1]):
        for z in range(slab.shape[2]):
            dist = abs(cosd(rotx)*(z-ctr[1]) - sind(rotx)*(y-ctr[0]))
            if dist < (thick/2 / dat[n]['meta']['voxSz'][1])+nGap:
                dat2[n]['mask'][:,y,z] = 0
            if dist > (thick/2 / dat[n]['meta']['voxSz'][1])-nGap:
                dat[n]['mask'][:,y,z] = 0
    slab = dat[n]['mask'].clone()
    dat[n]['mask'] = dat[n]['mask'] + dat2[n]['mask']
    plot3D(dat[n]['mask'].cpu())
    if n==0:
        allSlab = slab.clone()
    else:
        allSlab = torch.cat((allSlab,slab),0)
slab = allSlab
#dat_v = vectorizeData(dat[0])

# combine datasets (UP)
datComb = dat[0];
for i in range(1,len(dat)):
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




#plot3D(slab.cpu())

#print(asdasfd)

#%% define target
target = 1.0 * dat_v['mask3D'].clone()
target *= slab;


targetFA = 10

target = target * targetFA

if do_plot:
    f = plot3D(target.cpu())
    f.savefig('./img/target.png')

#%% set start value
stepdur=30e-6
nsteps = 80
TR=25.4e-3 # MP2RAGE
TR=30e-3 # some 3DEPI
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
    g_start = ktogTX(k,stepdur)
    #pulse_start = None
    
    nsinc = nsteps//2
    bwt = 8
    
    # from bwt, calculate required gradient strength
    if True:
        empirical_factor=0.8
        bw = bwt / (nsinc * stepdur * empirical_factor)
        gradamp = bw / smallTipAngle.gamma / thick * 2 * np.pi
        print(f'bw = {bw}Hz, grad = {gradamp} T/m')
        g_start = torch.ones(3,nsteps,device=dev,dtype=dt) * 1e-6
        #gradamp = 0.003
        maxSlew = 185 / 2
        nramp = int(np.ceil((gradamp/maxSlew)/stepdur))+2
        # initialize x and y axis
        g_start[:2,:] = 0.0001 * 0.0
        g_start[2,1:nramp+1] = torch.linspace(0,gradamp,nramp)
        sinc_start = nramp
        sinc_end = sinc_start+nsinc
        g_start[2,sinc_start:sinc_end+1]  = gradamp
        nramp2 = int(np.ceil((gradamp*3/maxSlew)/stepdur))+1
        g_start[2,sinc_end:sinc_end+nramp2] = torch.linspace(gradamp,-gradamp,nramp2)
        reph_start = sinc_end+nramp2
        g_start[2,reph_start:reph_start+nsinc//2] = -gradamp
        reph_end = reph_start+nsinc//2
        g_start[2,reph_end:reph_end+nramp] = torch.linspace(-gradamp,0,nramp)
        #pulse
        inp = torch.linspace(-bwt/2,bwt/2,nsinc,device=dev,dtype=dtc)
        pulse_start = torch.ones(nsteps,16,device=dev,dtype=dtc) * 0
        sinc = torch.sinc(inp)
        filt = torch.hamming_window(nsinc).to(sinc.device);
        fsinc = sinc * filt
        # calculate correct amplitude integral
        ai90 = dat_v['meta']['Uref']*5e-4
        aiSinc = fsinc.sum()*stepdur
        sincscale = ai90/aiSinc/ 90*targetFA / np.sqrt(dat_v['meta']['nTx']) * 1.55 * 0.85 #* 2

        # freq shift
        df = -np.sqrt(pos[2]**2 + pos[1]**2) * gradamp * smallTipAngle.gamma * 0.7
        freqShift = torch.linspace(0,-df * stepdur * nsinc,nsinc,device=dev)
        
        fsincND = (fsinc * torch.exp(1j * freqShift)).unsqueeze(1).repeat_interleave(16,1) * sincscale
        #plt.plot(fsinc.cpu().real);
        #plt.show();
        rf_mask = torch.zeros(nsteps,device=dev,dtype=torch.bool)
        rf_mask[sinc_start:sinc_start+nsinc] = True
        pulse_start[sinc_start:sinc_start+nsinc,:] = fsincND
        # now rotate gradients
        gy = g_start[1,:]*cosd(-rotx) - g_start[2,:] * sind(-rotx)
        gz = g_start[1,:]*sind(-rotx) - g_start[2,:] * cosd(-rotx)
        g_start[1,:] = gy
        g_start[2,:] = gz
    
    #print(asdf)
    
    #plt.plot(g_start[2,:].cpu())
    #plt.show()
    if False:
        pulse_start = torch.ones(nsteps,16,device=dev,dtype=dtc) * targetFA / (nsteps*stepdur) / 2000
        # random gradients
        g_start = torch.randn(3,nsteps,device=dev,dtype=dt) * 0.0003

        # gradients along slice dir
        g_start = torch.ones(3,nsteps,device=dev,dtype=dt) * 0
        g_start[:2,:] = 1e-6
        g_start[2,:] = 1e-4
        s = int(nsteps/4)
        g_start[2,:s] = -5e-4
        g_start[2,s:3*s] = 5e-4
        g_start[2,3*s:] = -5e-4    

        s = int(nsteps/4)
        g_start[2,:s] = -5e-4
        g_start[2,s:2*s] = 5e-4
        g_start[2,2*s:3*s] = -5e-4
        g_start[2,3*s:] = 5e-4
        
        g_start[2,:] = -5e-4
        
        # random gradients
        g_start = torch.randn(3,nsteps,device=dev,dtype=dt) * 0.0003
        
        # spins
        k = spins(nsteps,kmax=280,alpha=10,beta=0.5,u=8*3,v=5*3,dev=dat_v['s'].device)*0.5
        g_start = ktogTX(k,stepdur)
        
        # now rotate gradients
        gy = g_start[1,:]*cosd(-rotx) - g_start[2,:] * sind(-rotx)
        gz = g_start[1,:]*sind(-rotx) - g_start[2,:] * cosd(-rotx)
        g_start[1,:] = gy
        g_start[2,:] = gz

    g_start[:,0] = 0.0
    g_start[:,-1] = 0.0
    
    # CP mode
    phs = torch.linspace(2*torch.pi/dat_v['nTx'],2*torch.pi,dat_v['nTx'],device=dev,dtype=dt)
    phs[::2] += -98/180*torch.pi
    pulse_start = pulse_start * torch.exp(1j * phs).reshape(1,-1)
    
    #pulse_start = None
    #g_start = None
    #g_start = torch.randn(3,nsteps,device=dev,dtype=dt) * 0.0001
    

torch.autograd.set_detect_anomaly(False)


# system limits (SAR, maxRF, etc)
limits = smallTipAngle.getLimits()
limits['plotscale'] = targetFA * 1.2
limits['targetFA'] = targetFA
limits['maxSAR'] = 9.5
limits['maxRF'] = 185
limits['sliceProfWeight'] = 20 #decrease to have less strict slice profile but maybe more homhog. excitation 
#limits['maxG'] = 1e-12
#limits['phaseOpt'] = True
largeFA = (target.max() > 30)
# if FA > 30: design for small FA and up-scale
limits['plotiter'] = 200
limits['dispiter'] = 20
limits['stepiter'] = 500
#limits['RF_mask'] = rf_mask # comment this line to be less restritive in terms of timing
#limits['gradMomZero'] = True
#limits['sumSig'] = 1/3
#limits['lr'] = 8e-5

if largeFA:
    limits['maxSAR'] = limits['maxSAR']/100
    limits['maxRF'] = limits['maxRF']/15
    limits['targetFA'] = limits['targetFA']/10
    
    limits['plotscale'] = limits['plotscale']/10
    target = target/10
    if pulse_start is not None:
        pulse_start = pulse_start/10


# si = limits['stepiter']
# limits['stepiter'] = 500
# pulse_start,g_start = smallTipAngle.optimize_RF_grad(dat_v,target,niter=1000,nsteps=nsteps,pulse_start=pulse_start,
#                            g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits,do_plot=do_plot,optim="ASGD")
# limits['stepiter'] = si
        
pulse,g = smallTipAngle.optimize_RF_grad(dat_v,target,niter=nIter,nsteps=nsteps,pulse_start=pulse_start,
                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits,do_plot=do_plot)

#pulse,g = smallTipAngle.optimize_spokes(dat_v,target,niter=nIter,nsteps=nsteps,nspokes=nspokes,pulse_start=pulse_start,
#                           g_start=g_start,stepdur=stepdur,VOPs=VOPs,TR=TR,limits=limits,do_plot=do_plot,thick=thick,rotx=rotx)




#torch.autograd.set_detect_anomaly(True)

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
    limits['maxRF'] = limits['maxRF']*15
    limits['plotscale'] = limits['plotscale']*10
    limits['targetFA'] = limits['targetFA'] * 10

# we can also optimize RF only:
if do_UP and False:
    # RMSE 1.13 1281 sec for RF +grad
    pulse,g = bloch.optimize_RF_only(dat_v,target,niter=1000,nsteps=nsteps,pulse_start=pulse_start,
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
writePulse(f'{pulsename}.ini', pulse.cpu()/targetFA, g.cpu(),orientation=orientation,stepdur=stepdur,FA=1)
#writePulse('db_MPRAGE.ini', pulse.cpu(), g.cpu(),orientation=orientation,stepdur=stepdur,FA=targetFA)
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
#smallTipAngle.plotVec(bestvec,dat_v,stepdur)
#plotVec(load,dat_v,stepdur)
if do_plot or True:
    f = bloch.plotVec(pulse=pulse,g=g,dat_v=dat_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR);
    filename = f'img/final_{pulsename}.png'
    f.savefig(filename)

#torch.save(bestvec,'pulse.dat')

#pulse,g,k = splitVec(bestvec,stepdur)
#pulse_= pulse.reshape(dat_v['nTx'],-1)
#writePulse('./test.ini',pulse_,g,FA=7,stepdur=stepdur)
