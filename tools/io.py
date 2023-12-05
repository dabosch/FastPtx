#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 14:53:26 2022

@author: dario
"""
from scipy.io import loadmat
import mat73
import torch
import configparser

import scipy.io as spio

def newloadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def readVOPfile(fname,dev=torch.device('cpu'),dtc=torch.complex64):
    try:
        mat = newloadmat(fname)
    except:
        mat = mat73.loadmat(fname)
    VOPs = torch.tensor(mat['VOPs'],device=dev,dtype=dtc)
    return VOPs

def readData(fname,dev=torch.device('cpu'),dt=torch.float32,Unorm=None,
             maskName='brainMask'):
    try:
        mat = newloadmat(fname)
    except:
        mat = mat73.loadmat(fname)
    # workaround: find corresponding complex data type
    if dt==torch.float16:
        dtc = torch.complex32
    elif dt==torch.float64:
        dtc = torch.complex128
    else:
        dtc = torch.complex64
        
    dat = dict()
    dat['s'] = torch.tensor(mat['nT_per_V'],device=dev,dtype=dtc) / 1e9
    dat['nTx'] = mat['nT_per_V'].shape[3]
    try:
        dat['b0'] = torch.tensor(mat['deltaB0_Hz'],device=dev,dtype=dt) * 2 * torch.pi
    except:
        dat['b0'] = torch.zeros(mat['nT_per_V'].shape[0:3],device=dev,dtype=dt)
        print('Warning: b0 maps not found, assuming zeros!')
        
    if maskName is not None:
        dat['mask'] = torch.tensor(mat[maskName],device=dev,dtype=bool)    
    else:
        dat['mask'] = torch.ones(dat['b0'].shape,device=dev,dtype=dt)>0
        print('Warning: mask not found!')
    dat['meta'] = mat['meta']
    dist = torch.zeros((3,)+dat['s'].shape[0:3],device=dev)
    
    for i in range(dist.shape[3]):
        dist[2,:,:,i] = (i-dat['meta']['isocenter'][2])*dat['meta']['voxSz'][2]
    for i in range(dist.shape[2]):
        dist[1,:,i,:] = (i-dat['meta']['isocenter'][1])*dat['meta']['voxSz'][1]
    for i in range(dist.shape[1]):
        dist[0,i,:,:] = (i-dat['meta']['isocenter'][0])*dat['meta']['voxSz'][0]
    dat['dist'] = dist
    # normalize maps to ref voltage
    if Unorm is not None:
        dat['s'] = dat['s'] / Unorm *  float(dat['meta']['Uref'])
    return dat

def interpData(dat,factor):
    sNewr = torch.nn.functional.interpolate(dat['s'].real.permute([3,0,1,2]).unsqueeze(1),scale_factor=(factor,factor,factor),recompute_scale_factor=True)
    sNewi = torch.nn.functional.interpolate(dat['s'].imag.permute([3,0,1,2]).unsqueeze(1),scale_factor=(factor,factor,factor),recompute_scale_factor=True)
    dat['s'] = (sNewr + 1j*sNewi).permute(2,3,4,0,1).squeeze()
    
    new = torch.nn.functional.interpolate(dat['b0'].unsqueeze(0).unsqueeze(0),scale_factor=(factor,factor,factor),recompute_scale_factor=True)
    dat['b0'] = new.squeeze()
    
    new = torch.nn.functional.interpolate(dat['dist'].unsqueeze(0),scale_factor=(factor,factor,factor),recompute_scale_factor=True)
    dat['dist'] = new.squeeze()
    
    new = torch.nn.functional.interpolate(1.0*dat['mask'].unsqueeze(0).unsqueeze(0),scale_factor=(factor,factor,factor),recompute_scale_factor=True)
    dat['mask'] = new.squeeze()>0.5
    
    return dat
    
def combineData(dat1,dat2):
    datO = dict();
    keys = ['mask','s','b0']
    for key in keys:
        datO[key] = torch.cat((dat1[key],dat2[key]),0)
        
    datO['dist'] = torch.cat((dat1['dist'],dat2['dist']),1)
    datO['nTx'] = dat1['nTx']
    datO['meta'] = dat1['meta']
    return datO

def vectorizeData(dat):
    dev = dat['s'].device
    dtc = dat['s'].dtype
    dt = dat['b0'].dtype
    dat_v = dict()
    dat_v['nVox'] = int(dat['mask'].reshape(1,-1).sum().cpu())
    dat_v['nTx'] = dat['nTx']
    dat_v['mask3D'] = dat['mask']
    dat_v['mask'] = torch.ones(dat_v['nVox'],dtype=bool,device=dev)
    dat_v['s'] = torch.zeros(dat_v['nVox'],dat_v['nTx'],device=dev,dtype=dtc)
        
    for i in range(dat_v['nTx']):
        dat_v['s'][:,i] = dat['s'][:,:,:,i].masked_select(dat_v['mask3D'])
    dat_v['b0'] = dat['b0'].masked_select(dat_v['mask3D'])
    dat_v['dist'] = torch.zeros(3,dat_v['nVox'],dtype=dt,device=dev)
    for i in range(3):
        dat_v['dist'][i] = dat['dist'][i,:,:,:].masked_select(dat_v['mask3D'])
    dat_v['meta'] = dat['meta']
    
    return dat_v

def writePulse(fname,pulse,g,FA=7,stepdur=10e-6,orientation='traAP',asymmetry=0.5):
    
    oversample = 1
    
    pulse = pulse.detach().cpu().clone()
    pulse = pulse.repeat_interleave(oversample,0)
    print(pulse.shape)
    #pulse = pulse.roll(0,1)
    g2 = g.detach().cpu().clone() *1e3 # mT/m
    g = g.detach().cpu().clone() *1e3 # mT/m
    g[:,-1] = 0 # set last gradient entries to absolute zero
    g2[:,-1] = 0 # set last gradient entries to absolute zero
    if orientation=='sagAP':
        g[2,:] = -g2[0,:]
        g[1,:] = g2[1,:]
        g[0,:] = -g2[2,:]
    elif orientation=='sagHF':
        g[2,:] = g2[0,:]
        g[1,:] = g2[2,:]
        g[0,:] = g2[1,:] 
    elif orientation=='unity':
        g[2,:] = g2[2,:] # f->h
        g[1,:] = -g2[0,:] # l->r
        g[0,:] = -g2[1,:] # a->p
    else: # traAP
        g[2,:] = -g2[2,:]
        g[1,:] = g2[1,:]
        g[0,:] = g2[0,:]
    
    # for some reason siemens doesn't accept grad raster times !=10 any more
    if stepdur != 10e-6:
        rep = round(stepdur/10e-6)
        pulse = pulse.repeat_interleave(rep,0)
        g = g.repeat_interleave(rep,1)
        stepdur = 10e-6
    
    nTx = pulse.shape[1]
    nSamples = g.shape[1]
    
    config = configparser.ConfigParser()
    # preserve case
    config.optionxform = lambda option: option
    config.add_section('pTXPulse')
    config['pTXPulse']['NUsedChannels'] = str(nTx)
    config['pTXPulse']['DimGradient'] = str(g.shape[0])
    config['pTXPulse']['DimRF'] = "2"
    config['pTXPulse']['MaxAbsRF'] = f'{pulse.abs().max():.3f}'
    config['pTXPulse']['InitialPhase'] = str(0)
    config['pTXPulse']['Asymmetry'] = str(asymmetry) # probably 1 or 0
    
    config['pTXPulse']['NominalFlipAngle'] = str(FA)
    config['pTXPulse']['Samples'] = str(nSamples)
    
    #config.add_section('pTXRFPulse')
    #config['pTXRFPulse']['Samples'] = str(nSamples * oversample)
    
    
    config['pTXPulse']['Oversample'] = str(oversample)
    
    config.add_section('Gradient')
    config['Gradient']['GradientSamples'] = str(nSamples)
    config['Gradient']['MaxAbsGradient[0]'] = f'{g[0,:].abs().max():.3f}  {g[1,:].abs().max():.3f}  {g[2,:].abs().max():.3f}' # mT/m
    config['Gradient']['GradRasterTime'] = str(int(stepdur*1e6)) # us
    
    for i in range(nSamples):
        config['Gradient'][f'G[{i}]'] = f'{g[0,i]:.3f}\t{g[1,i]:.3f}\t{g[2,i]:.3f}'
    
    for n in range(nTx):
        config.add_section(f'pTXPulse_ch{n}')
        if n==0:
            config[f'pTXPulse_ch{n}']['Samples'] = str(nSamples * oversample)
        phs = torch.pi/8*n
        for i in range(nSamples*oversample):
            # new: don't take CP mode phs into account
            config[f'pTXPulse_ch{n}'][f'RF[{i}]'] = f'{pulse[i,n].abs():.5f}\t{pulse[i,n].angle():.5f}'
            # old: write CP mode phase
            #config[f'pTXPulse_ch{n}'][f'RF[{i}]'] = f'{pulse[i,n].abs():.5f}\t{pulse[i,n].angle()+phs:.5f}'
    
    with open(fname, 'w') as configfile:    # save
        config.write(configfile)

def readPulse(fname,orientation='unity'):
    
    config = configparser.ConfigParser()
    # preserve case
    config.optionxform = lambda option: option
    config.read(fname)
    #pulse = pulse.detach().cpu()
    #g = g.detach().cpu() *1e3 # mT/m
    nTx = int(config['pTXPulse']['NUsedChannels'])
    nSamples = int(config['Gradient']['GradientSamples'])
    
    g = torch.zeros(3,nSamples)
    for i in range(nSamples):
        for n in range(3):
            g[n,i] = float(config['Gradient'][f'G[{i}]'].split('\t')[n])
    g = g / 1e3
    
    
    mag = torch.zeros(nSamples,nTx)
    phs = torch.zeros(nSamples,nTx)
    for i in range(nSamples):
        for n in range(nTx):
            mag[i,n] = float(config[f'pTXPulse_ch{n}'][f'RF[{i}]'].split('\t')[0])
            phs[i,n] = float(config[f'pTXPulse_ch{n}'][f'RF[{i}]'].split('\t')[1])
                       # - torch.pi/8*n
    pulse = mag * torch.exp(1j * phs)
    
    g2 = g.clone()
    if orientation=='sagAP':
        g[2,:] = -g2[0,:]
        g[1,:] = g2[1,:]
        g[0,:] = -g2[2,:]
    elif orientation=='sagHF':
        g[2,:] = g2[1,:]
        g[1,:] = g2[0,:]
        g[0,:] = g2[2,:] 
    elif orientation=='unity':
        g[2,:] = g2[2,:] # f->h
        g[1,:] = -g2[0,:] # l->r
        g[0,:] = -g2[1,:] # a->p
    else: # traAP
        g[2,:] = -g2[2,:]
        g[1,:] = g2[1,:]
        g[0,:] = g2[0,:]
    
    return pulse,g
    
    #for n in range(nTx):
    #    config.add_section(f'pTXPulse_ch{n}')
    #    for i in range(nSamples):
    #        config[f'pTXPulse_ch{n}'][f'RF[{i}]'] = f'{pulse[n,i].abs():.5f}\t{pulse[n,i].angle():.5f}'
    
    #with open(fname, 'w') as configfile:    # save
    #    config.write(configfile)
#readPulse('../pTXRFPulse8.ini')
