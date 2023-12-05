#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:01:33 2022

@author: dario
"""

#__all__ = ['createAmat']

import torch
from tools.trajectories import gtokTX, spins, ktogTX
import numpy as np
import matplotlib.pyplot as plt
import signal
#import time
import cv2
from scipy.io import savemat


class smallTipAngle:

    gamma = 267522189.9851 # in rad/s/T
    figtitle = "small tip angle"
    lr = 4e-4
    dispiter = 20
    @staticmethod
    def getLimits():
        lim = dict()
        lim['maxRF'] = 165 # V
        lim['maxG'] = 65e-3 # T/m
        lim['maxSlew'] = 185 # T/m/s
        lim['maxSlewB0'] = 35/(50e-6) / 1 # A/s # 35A@10kHz->35V in 50e-6s
        lim['maxSAR']= 9.5 # W/kg
        lim['plotscale']=15
        lim['t1'] = 2100e-3
        lim['t2'] = 35e-3
        lim['phaseOpt'] = False
        lim['sliceProfWeight'] = 0.0
        lim['targetFA'] = None
        lim['dispiter'] = 10
        lim['plotiter'] = 100
        lim['stepiter'] = 1000
        lim['gradMomZero'] = False
        lim['RF_mask'] = None
        lim['sumSig'] = 0
        lim['lr'] = 4e-4
        return lim
    
    def createAmat(self,dat_v,k,stepdur=10e-6):
        dev = dat_v['s'].device
        try:
            nTx = dat_v['s'].shape[1]
        except:
            nTx = 1
        ntimes = k.shape[1]
        
        M0 = 1
        
        #temp = torch.einsum('jk,ki->ji',-dat_v['dist'].T,k)
        temp = -dat_v['dist'].T@k
        remainingTime = (torch.linspace(ntimes-0.5,0.5,ntimes,device=dev)*stepdur)#.unsqueeze(0).tile(nVox,1)
        b0eff = -1 * torch.einsum('j,k->jk',dat_v['b0'], remainingTime)
        A = 1j * self.gamma * M0 * stepdur * torch.exp(1j * b0eff + 1j * temp)
        
        if nTx>1:
            Afull = A.tile(1,nTx) * dat_v['s'].repeat_interleave(ntimes,1)
        else:
            Afull = A * dat_v['s'].unsqueeze(1).repeat_interleave(ntimes,1)
        
        return Afull
    
    @classmethod
    def calcFA(self,dat_v,pulse,g=None,stepdur=10e-6,Amat=None,t1=None,t2=None):
        if Amat is None and g is None:
            raise Exception("both Amat and g are None, but one of them must be provided!")
        if Amat is None:
            k = gtokTX(g,stepdur)
            Amat = self.createAmat(self,dat_v,k,stepdur)
        FA = torch.einsum('k,jk->j',pulse.T.reshape(-1),Amat)*180/torch.pi 
        return FA
    
    @classmethod
    def calcFA3D(self,dat_v,pulse,g=None,stepdur=10e-6,Amat=None):
        FA = self.calcFA(dat_v,pulse,g,stepdur,Amat)
        dtc = dat_v['s'].dtype
        dev = dat_v['s'].device
        FA3D = torch.zeros(dat_v['mask3D'].shape,dtype=dtc,device=dev)
        FA3D[dat_v['mask3D']] = FA
        FA3D = FA3D.detach().cpu()
        return FA3D
    
    @classmethod
    def calcPulseSAR(self,pulse_in,VOPs,TR=20e-3,stepdur=10e-6):
        Att = -2.0 # dB
        pulse = pulse_in.clone() * 10**(Att/20)
        
        v = VOPs.unsqueeze(2).permute(3,2,1,0)   # v1kk
        p = pulse.conj().unsqueeze(0).unsqueeze(2) # 1t1k
        v1 = torch.matmul(p,v)        # tv11
        p = pulse.unsqueeze(0).unsqueeze(3) # 1tk1
        v2 = torch.matmul(v1,p)
        
        #v2 = v2.real.nan_to_num(0.0) + 1j * v2.imag.nan_to_num(0.0)
        v2 = v2.abs()
        #v2 = v2.nan_to_num(0.0)
        energy = v2.sum(1).squeeze()*stepdur # sum over time
        localSAR = energy.max() / TR
        return localSAR
    
    @classmethod
    def optimize_RF_grad(self,dat_v,target,niter=100000,nsteps=200,pulse_start=None,
                         g_start=None,stepdur=10e-6,VOPs=None,TR=20e-6,limits=None,do_plot=True,optim="AdamW"):
        
        if limits is None:
            limits = self.getLimits()
        dev = dat_v['s'].device
        dt = dat_v['dist'].dtype
        dtc = dat_v['s'].dtype
        lr=limits['lr']
        # define default values for g and pulse
        if g_start is None:
            g_start = torch.ones(3,nsteps,dtype=dt,device=dev)*limits['maxG']/1e3
            g_start[:,[0,-1]] = 0
            #g_start = (torch.rand(3,nsteps).to(dev).to(dt) - 0.5) * limits['maxG']/1e1
        if pulse_start is None:
            pulse_start = torch.ones(dat_v['nTx'],g_start.shape[1],dtype=dtc,device=dev)*limits['maxRF']/1e2
        
        optvec_template = torch.cat((g_start.reshape(-1)/limits['maxG'],
                                     (pulse_start/limits['maxRF']).reshape(-1) ) )
        # optvec_template = torch.cat((g_start.reshape(-1)/limits['maxG'],
        #                              (pulse_start/limits['maxRF']).real.reshape(-1),
        #                              (pulse_start/limits['maxRF']).imag.reshape(-1)))
        optvec = optvec_template.clone().requires_grad_(True)
        
        def setupOptim(optvec,lr,optim):
            if optim == "ASGD":
                optimizer = torch.optim.ASGD([optvec], lr=lr/1e4)
            elif optim=="SGD":
                optimizer = torch.optim.SGD([optvec], lr=lr/1e4)
            elif optim=="LBFGS":
                optimizer = torch.optim.LBFGS([optvec], lr=lr/1e3)
            else:
                # default: use AdamW
                optimizer = torch.optim.AdamW([optvec], lr=lr/1e0)
            return optimizer
        # setup optimizer
        optimizer = setupOptim(optvec,lr,optim)

        target_v = target.masked_select(dat_v['mask3D'])
        
        # calc and plot initial status
        pulse,g = self.splitVec(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,VOPs=VOPs,TR=TR)
        err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        
        bestErr = 1e32
        lastBest = 1e32
        lastbestI = 1
        
        # disp is slow, plotting is slower -> don't plot/disp too often
        #dispiter = 10
        #plotiter = 100
        # after how many steps do we re-initialize the optimizer?
        stepiter = limits['stepiter']
        # keep track of best ever result
        bestvec = optvec.clone().detach()
        
        # catch CTRL+C:
        #signal.signal(signal.SIGINT, signal.default_int_handler)
        #doExit = False
        i = 0
        while (i < niter+1):
            try:
                if i % stepiter == 0 or err.isnan():
                    if lastbestI <= (i-stepiter):
                        lr = lr/5
                    # reinitialize optimizer
                    optvec = bestvec.clone().requires_grad_(True)
                    optimizer = setupOptim(optvec,lr,optim)
                    print('reinitializing optimizer')
                optimizer.zero_grad()
                pulse,g = self.splitVec(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
                err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)
                
                if err < bestErr: 
                    # new all-time best
                    bestvec = optvec.clone().detach()
                    bestErr = err.detach().clone()
                
                err.backward(retain_graph=True)
                optimizer.step()
                if (i) % limits['dispiter'] == 0:
                    print(f'{i:6}: {err[0]:14.6f} - best: {bestErr[0]:10.6f} - lr: {lr}')
                    pulse,g = self.splitVec(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
                    #err = calcErr(pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)                
                if bestErr < lastBest and (i) % limits['plotiter'] == 0:
                    # plot best vec
                    pulse,g = self.splitVec(self,bestvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
                    if do_plot:
                        self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=True,i=i,VOPs=VOPs,TR=TR)
                    self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
                    lastBest = bestErr
                    lastbestI = i
                #if doExit:
                #    break
                i+=1
            except KeyboardInterrupt:
                if i == niter:
                    break
                i = niter
                #break
        #bestvec = bestvec.clone().detach()
        pulse,g = self.splitVec(self,bestvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
        self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=False,i=i,VOPs=VOPs,TR=TR)
        pulse = pulse.detach()
        g = g.detach()
        return pulse,g    
    
    @classmethod
    def optimize_RF_only(self,dat_v,target,niter=100000,nsteps=200,pulse_start=None,
                         g_start=None,stepdur=10e-6,VOPs=None,TR=20e-6,limits=None,do_plot=True):
        
        if limits is None:
            limits = self.getLimits()
        dev = dat_v['s'].device
        dt = dat_v['dist'].dtype
        dtc = dat_v['s'].dtype
        lr=limits['lr']
        # define default values for g and pulse
        if g_start is None:
            g_start = torch.ones(3,nsteps,dtype=dt,device=dev)*limits['maxG']/1e3
            g_start[:,[0,-1]] = 0
            #g_start = (torch.rand(3,nsteps).to(dev).to(dt) - 0.5) * limits['maxG']/1e1
        if pulse_start is None:
            pulse_start = torch.ones(dat_v['nTx'],g_start.shape[1],dtype=dtc,device=dev)*limits['maxRF']/1e2
        
        optvec_template = torch.cat(((pulse_start/limits['maxRF']).real.reshape(-1),
                                     (pulse_start/limits['maxRF']).imag.reshape(-1)))
        optvec = optvec_template.clone().requires_grad_(True)
        g = g_start.clone()
        # setup optimizer
        optimizer = torch.optim.AdamW([optvec], lr=lr/1e0)
        
        target_v = target.masked_select(dat_v['mask3D'])
        
        # calc and plot initial status
        pulse = self.splitPulse(self,optvec,dat_v['nTx'], limits=limits,stepdur=stepdur)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,VOPs=VOPs,TR=TR)
        err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        
        bestErr = 1e32
        lastBest = 1e32
        lastbestI = 1
        
        # after how many steps do we re-initialize the optimizer?
        stepiter = limits['stepiter']
        # keep track of best ever result
        bestvec = optvec.clone().detach()
        
        # catch CTRL+C:
        #signal.signal(signal.SIGINT, signal.default_int_handler)
        #doExit = False
        i = 1
        while (i < niter+1):
            try:
                if i % stepiter == 0 or err.isnan():
                    if lastbestI < (i-stepiter):
                        lr = lr/5
                    # reinitialize optimizer
                    optvec = bestvec.clone().requires_grad_(True)
                    optimizer = torch.optim.AdamW([optvec], lr=lr)
                    
                optimizer.zero_grad()
                pulse = self.splitPulse(self,optvec,dat_v['nTx'], limits=limits,stepdur=stepdur)
                err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)
                
                if err < bestErr: 
                    # new all-time best
                    bestvec = optvec.clone().detach()
                    bestErr = err.detach().clone()
                
                err.backward(retain_graph=True)
                optimizer.step()
                if (i) % limits['dispiter'] == 0:
                    print(f'{i:6}: {err[0]:14.6f} - best: {bestErr[0]:10.6f} - lr: {lr}')
                    
                if bestErr < lastBest and (i) % limits['plotiter'] == 0:
                    # plot best vec
                    pulse = self.splitPulse(self,bestvec,dat_v['nTx'], limits=limits,stepdur=stepdur)
                    if do_plot:
                        self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=True,i=i,VOPs=VOPs,TR=TR)
                    self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
                    lastBest = bestErr
                    lastbestI = i
                #if doExit:
                #    break
                i+=1
            except KeyboardInterrupt:
                if i == niter:
                    break
                i = niter
                #break
        #bestvec = bestvec.clone().detach()
        pulse = self.splitPulse(self,optvec,dat_v['nTx'], limits=limits,stepdur=stepdur)
        self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=False,i=i,VOPs=VOPs,TR=TR)
        pulse = pulse.detach()
        g = g.detach()
        return pulse,g    
    
    @classmethod
    def optimize_grad_only(self,dat_v,target,niter=100000,nsteps=200,pulse_start=None,
                         g_start=None,stepdur=10e-6,VOPs=None,TR=20e-6,limits=None,do_plot=True):
        
        if limits is None:
            limits = self.getLimits()
        dev = dat_v['s'].device
        dt = dat_v['dist'].dtype
        dtc = dat_v['s'].dtype
        lr=limits['lr']
        # define default values for g and pulse
        if g_start is None:
            g_start = torch.ones(3,nsteps,dtype=dt,device=dev)*limits['maxG']/1e3
            g_start[:,[0,-1]] = 0
            #g_start = (torch.rand(3,nsteps).to(dev).to(dt) - 0.5) * limits['maxG']/1e1
        if pulse_start is None:
            pulse_start = torch.ones(dat_v['nTx'],g_start.shape[1],dtype=dtc,device=dev)*limits['maxRF']/1e2
        
        optvec_template = torch.cat(((g_start/limits['maxG']).reshape(-1),))
        optvec = optvec_template.clone().requires_grad_(True)
        pulse = pulse_start.clone()
        # setup optimizer
        optimizer = torch.optim.AdamW([optvec], lr=lr/1e0)
        
        target_v = target.masked_select(dat_v['mask3D'])
        
        # calc and plot initial status
        g = optvec.reshape(3,-1)*limits['maxG']
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,VOPs=VOPs,TR=TR)
        err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        
        bestErr = 1e32
        lastBest = 1e32
        lastbestI = 1
        
        # disp is slow, plotting is slower -> don't plot/disp too often
        #dispiter = 10
        #plotiter = 100
        # after how many steps do we re-initialize the optimizer?
        stepiter = limits['stepiter']
        # keep track of best ever result
        bestvec = optvec.clone().detach()
        
        # catch CTRL+C:
        #signal.signal(signal.SIGINT, signal.default_int_handler)
        #doExit = False
        i = 1
        while (i < niter+1):
            try:
                if i % stepiter == 0 or err.isnan():
                    if lastbestI < (i-stepiter):
                        lr = lr/5
                    # reinitialize optimizer
                    optvec = bestvec.clone().requires_grad_(True)
                    optimizer = torch.optim.AdamW([optvec], lr=lr)
                    
                optimizer.zero_grad()
                g = optvec.reshape(3,-1)*limits['maxG']
                err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)
                
                if err < bestErr: 
                    # new all-time best
                    bestvec = optvec.clone().detach()
                    bestErr = err.detach().clone()
                
                err.backward(retain_graph=True)
                optimizer.step()
                if (i) % limits['dispiter'] == 0:
                    print(f'{i:6}: {err[0]:14.6f} - best: {bestErr[0]:10.6f} - lr: {lr}')
                    
                if bestErr < lastBest and (i) % limits['plotiter'] == 0:
                    # plot best vec
                    g = bestvec.reshape(3,-1)*limits['maxG']
                    if do_plot:
                        self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=True,i=i,VOPs=VOPs,TR=TR)
                    self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
                    lastBest = bestErr
                    lastbestI = i
                #if doExit:
                #    break
                i+=1
            except KeyboardInterrupt:
                if i == niter:
                    break
                i = niter
                #break
        #bestvec = bestvec.clone().detach()
        g = optvec.reshape(3,-1)*limits['maxG']
        self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=False,i=i,VOPs=VOPs,TR=TR)
        pulse = pulse.detach()
        g = g.detach()
        return pulse,g    
    
    @classmethod
    def optimize_RF_spins(self,dat_v,target,niter=100000,nsteps=200,pulse_start=None,
                         g_start=None,stepdur=10e-6,VOPs=None,TR=20e-6,limits=None,do_plot=True):
        
        if limits is None:
            limits = self.getLimits()
        dev = dat_v['s'].device
        dt = dat_v['dist'].dtype
        dtc = dat_v['s'].dtype
        lr=limits['lr']
        # define default values for g and pulse
        spins_param = torch.tensor([[50, 5, 0.5, 24, 15]],dtype=dt,device=dev)
        if pulse_start is None:
            pulse_start = torch.ones(dat_v['nTx'],g_start.shape[1],dtype=dtc,device=dev)*limits['maxRF']/1e2
        
        optvec_template = torch.cat((spins_param.reshape(-1),
                                     (pulse_start/limits['maxRF']).reshape(-1) ) )
        optvec = optvec_template.clone().requires_grad_(True)
        
        # setup optimizer
        optimizer = torch.optim.AdamW([optvec], lr=lr/1e0)
        
        target_v = target.masked_select(dat_v['mask3D'])
        
        # calc and plot initial status
        pulse,g = self.splitVecSpins(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,VOPs=VOPs,TR=TR)
        err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        
        bestErr = 1e32
        lastBest = 1e32
        lastbestI = 1
        
        # disp is slow, plotting is slower -> don't plot/disp too often
        #dispiter = 10
        #plotiter = 100
        # after how many steps do we re-initialize the optimizer?
        stepiter = limits['stepiter']
        # keep track of best ever result
        bestvec = optvec.clone().detach()
        
        # catch CTRL+C:
        #signal.signal(signal.SIGINT, signal.default_int_handler)
        #doExit = False
        i = 0
        while (i < niter+1):
            try:
                if i % stepiter == 0 or err.isnan():
                    if lastbestI < (i-stepiter):
                        lr = lr/5
                    # reinitialize optimizer
                    optvec = bestvec.clone().requires_grad_(True)
                    optimizer = torch.optim.AdamW([optvec], lr=lr)
                    
                optimizer.zero_grad()
                pulse,g = self.splitVecSpins(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
                err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)
                
                if err < bestErr: 
                    # new all-time best
                    bestvec = optvec.clone().detach()
                    bestErr = err.detach().clone()
                
                err.backward(retain_graph=True)
                optimizer.step()
                if (i) % limits['dispiter'] == 0:
                    print(f'{i:6}: {err[0]:14.6f} - best: {bestErr[0]:10.6f} - lr: {lr}')
                    pulse,g = self.splitVecSpins(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
                    #err = calcErr(pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)                
                if bestErr < lastBest and (i) % limits['plotiter'] == 0:
                    # plot best vec
                    pulse,g = self.splitVecSpins(self,bestvec, dat_v['nTx'], limits=limits,stepdur=stepdur)
                    if do_plot:
                        self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=True,i=i,VOPs=VOPs,TR=TR)
                    self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
                    #print(optvec[:5].real)
                    lastBest = bestErr
                    lastbestI = i
                #if doExit:
                #    break
                i+=1
            except KeyboardInterrupt:
                if i == niter:
                    break
                i = niter
                #break
        #bestvec = bestvec.clone().detach()
        self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=False,i=i,VOPs=VOPs,TR=TR)
        pulse = pulse.detach()
        g = g.detach()
        return pulse,g    
    
    
    @classmethod
    def optimize_kT_points(self,dat_v,target,niter=100000,nsteps=5,pulse_dur=100,pulse_start=None,
                         g_start=None,stepdur=10e-6,VOPs=None,TR=20e-6,limits=None,do_plot=True):
        
        if limits is None:
            limits = self.getLimits()
        dev = dat_v['s'].device
        dt = dat_v['dist'].dtype
        dtc = dat_v['s'].dtype
        lr=limits['lr']
        
        # gradient blips between rect pulses
        grad_blip_dur = torch.ones(nsteps,dtype=dtc,device=dev) * 6
        # duration of RF subpulses [steps]
        subpulse_dur = torch.ones(nsteps,dtype=dtc,device=dev)
        subpulse_dur = subpulse_dur * ((pulse_dur - grad_blip_dur.sum())/nsteps).abs().floor();
        grad = torch.ones(nsteps*3,dtype=dtc,device=dev) * 0.0001
        rf   = torch.ones(nsteps * dat_v['nTx'],dtype=dtc,device=dev) * 20
        
        optvec_template = torch.cat((grad_blip_dur,subpulse_dur,grad,rf))
        optvec = optvec_template.clone().requires_grad_(True)
        
        # setup optimizer
        optimizer = torch.optim.AdamW([optvec], lr=lr/1e0)
        
        target_v = target.masked_select(dat_v['mask3D'])
        
        # calc and plot initial status
        pulse,g = self.splitVecKT(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=pulse_dur)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,VOPs=VOPs,TR=TR)
        err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        
        bestErr = 1e32
        lastBest = 1e32
        lastbestI = 1
        
        # disp is slow, plotting is slower -> don't plot/disp too often
        #dispiter = 10
        #plotiter = 100
        # after how many steps do we re-initialize the optimizer?
        stepiter = limits['stepiter']
        # keep track of best ever result
        bestvec = optvec.clone().detach()
        
        # catch CTRL+C:
        #signal.signal(signal.SIGINT, signal.default_int_handler)
        #doExit = False
        i = 0
        while (i < niter+1):
            try:
                if i % stepiter == 0 or err.isnan():
                    if lastbestI < (i-stepiter):
                        lr = lr/5
                    # reinitialize optimizer
                    optvec = bestvec.clone().requires_grad_(True)
                    optimizer = torch.optim.AdamW([optvec], lr=lr)
                    
                optimizer.zero_grad()
                pulse,g = self.splitVecKT(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=pulse_dur)
                err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)
                
                if err < bestErr: 
                    # new all-time best
                    bestvec = optvec.clone().detach()
                    bestErr = err.detach().clone()
                
                err.backward(retain_graph=True)
                optimizer.step()
                if (i) % limits['dispiter'] == 0:
                    print(f'{i:6}: {err[0]:14.6f} - best: {bestErr[0]:10.6f} - lr: {lr}')
                    pulse,g = self.splitVecKT(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=pulse_dur)
                    #err = calcErr(pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)                
                if bestErr < lastBest and (i) % limits['plotiter'] == 0:
                    # plot best vec
                    pulse,g = self.splitVecKT(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=pulse_dur)
                    if do_plot:
                        self.plotVec(pulse.detach(),g.detach(),dat_v,stepdur,limits=limits,save=True,i=i,VOPs=VOPs,TR=TR)
                    self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
                    lastBest = bestErr
                    lastbestI = i
                #if doExit:
                #    break
                i+=1
            except KeyboardInterrupt:
                if i == niter:
                    break
                i = niter
                #break
        #bestvec = bestvec.clone().detach()
        self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=False,i=i,VOPs=VOPs,TR=TR)
        pulse = pulse.detach()
        g = g.detach()
        return pulse,g   
    
    @classmethod
    def optimize_spokes(self,dat_v,target,niter=100000,nsteps=100,nspokes=3,pulse_start=None,
                         g_start=None,stepdur=10e-6,VOPs=None,TR=20e-6,limits=None,do_plot=True,rotx=0,thick=0.035):
        
        if limits is None:
            limits = self.getLimits()
        dev = dat_v['s'].device
        dt = dat_v['dist'].dtype
        dtc = dat_v['s'].dtype
        lr=limits['lr']
        
        if g_start is None:
            grad = torch.ones(nspokes*3,dtype=dtc,device=dev) * 0.003
        else:
            grad = g_start.to(dev).to(dtc).T.reshape(-1,1)
        if pulse_start is None:
            rf = torch.ones(nspokes * dat_v['nTx'],dtype=dtc,device=dev) * 20
        else:
            rf = pulse_start.to(dev).to(dtc).reshape(-1,1)
        
        optvec_template = torch.cat((grad,rf))
        optvec = optvec_template.clone().requires_grad_(True)
        
        # setup optimizer
        optimizer = torch.optim.AdamW([optvec], lr=lr/1e0)
        
        target_v = target.masked_select(dat_v['mask3D'])
        
        # calc and plot initial status
        pulse,g = self.splitVecSpokes(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=nsteps,rotx=rotx,thick=thick)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,VOPs=VOPs,TR=TR)
        err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        
        bestErr = 1e32
        lastBest = 1e32
        lastbestI = 1
        
        # disp is slow, plotting is slower -> don't plot/disp too often
        #dispiter = 10
        #plotiter = 100
        # after how many steps do we re-initialize the optimizer?
        stepiter = limits['stepiter']
        # keep track of best ever result
        bestvec = optvec.clone().detach()
        
        # catch CTRL+C:
        #signal.signal(signal.SIGINT, signal.default_int_handler)
        #doExit = False
        i = 0
        while (i < niter+1):
            try:
                if i % stepiter == 0 or err.isnan():
                    if lastbestI < (i-stepiter):
                        lr = lr/5
                    # reinitialize optimizer
                    optvec = bestvec.clone().requires_grad_(True)
                    optimizer = torch.optim.AdamW([optvec], lr=lr)
                    
                optimizer.zero_grad()
                pulse,g = self.splitVecSpokes(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=nsteps,rotx=rotx,thick=thick)
                err = self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)
                
                if err < bestErr: 
                    # new all-time best
                    bestvec = optvec.clone().detach()
                    bestErr = err.detach().clone()
                
                err.backward(retain_graph=True)
                optimizer.step()
                if (i) % limits['dispiter'] == 0:
                    print(f'{i:6}: {err[0]:14.6f} - best: {bestErr[0]:10.6f} - lr: {lr}')
                    pulse,g = self.splitVecSpokes(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=nsteps,rotx=rotx,thick=thick)
                    #err = calcErr(pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,VOPs=VOPs,TR=TR)                
                if bestErr < lastBest and (i) % limits['plotiter'] == 0:
                    # plot best vec
                    pulse,g = self.splitVecSpokes(self,optvec, dat_v['nTx'], limits=limits,stepdur=stepdur,pulse_dur=nsteps,rotx=rotx,thick=thick)
                    if do_plot:
                        self.plotVec(pulse.detach(),g.detach(),dat_v,stepdur,limits=limits,save=True,i=i,VOPs=VOPs,TR=TR)
                    self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
                    lastBest = bestErr
                    lastbestI = i
                #if doExit:
                #    break
                i+=1
            except KeyboardInterrupt:
                if i == niter:
                    break
                i = niter
                #break
        #bestvec = bestvec.clone().detach()
        self.calcErr(self,pulse,g,dat_v,target_v,stepdur=stepdur,limits=limits,verbose=True,VOPs=VOPs,TR=TR)
        if do_plot:
            self.plotVec(pulse,g,dat_v,stepdur,limits=limits,save=False,i=i,VOPs=VOPs,TR=TR)
        pulse = pulse.detach()
        g = g.detach()
        return pulse,g   
    
    
    def calcErr(self,pulse,g,dat_v,target,limits,stepdur=10e-6,verbose=False,VOPs=None,TR=20e-3):
        maxGedge = limits['maxG']/1e3
        maxRFedge = limits['maxRF']
        
        g = g.clone()
        pulse = pulse.clone()
        g[:,-1] = 0
        if False: # add 1% of jitter to RF and grads
            pulse = pulse * (0.99 + torch.rand(pulse.shape).to(pulse.device)*0.02)
            g = g * (0.99 + torch.rand(g.shape).to(g.device)*0.02)
        
        #k = gtokTX(g,stepdur)
        #print(pulse)
        FA = self.calcFA(dat_v,pulse,g,stepdur,t1=limits['t1'],t2=limits['t2'])
        if False:
            FA = FA + torch.rand(FA.shape).to(FA.device)*0.01
        zero = torch.tensor([0],device=pulse.device,dtype=pulse.real.dtype)
        # FA magnitude error
        penMag = ((FA.abs()-target.abs())**2).mean().sqrt() 
        if limits['phaseOpt']:
            penPhs = ((FA.angle()-target.angle())**2).mean() * 10
        else:
            penPhs = 0.0
        if limits['sliceProfWeight'] > 0.0:
            mask = (target.abs() <= 0.1);
            #print(target.abs().min())
            #plt.plot(target.abs().cpu())
            plt.show()
            
            if mask.sum() > 0:
                penSlice = ((FA.masked_select(mask).abs()*limits['sliceProfWeight'])**2).mean()
            else:
                penSlice = 0.0
        else:
            penSlice = 0.0
        # pulse voltage limit
        errVolt = torch.max(zero,pulse.abs().max()-limits['maxRF'])**2 * 1e6
        # gradient power limit
        errGrad = (torch.max(zero,g.abs().max()-limits['maxG']) * 1e3)**2 * 1e9
        # slew rate limit
        #zg = torch.zeros(3,1).to(g.device).to((g.dtype))
        #gSL = torch.cat((g,zg),dim=1)
        slew = torch.diff(g,dim=1) / 10e-6#stepdur
        errSlew = torch.max(zero,slew.abs().max()-limits['maxSlew'])**2 * 1e6
        # add slight slew rate penalty
        penSlew = (slew.abs()**2).mean() * 1e-6
        
        # edge condition: first and last step of RF and Grad have to be zero
        #pulseOut = pulse.reshape(-1,dat_v['nTx'])
        #errEdge = ((pulseOut[0,:].abs().max()*1e3)**2 + (pulseOut[-1,:].abs().max()*1e3)**2 \
        #          + (g[:,0].abs().max()*1e6)**2 + (g[:,-1].abs().max()*1e6)**2) * 1e0
        #errEdge = (torch.max(pulseOut[0,:].abs(),pulseOut[-1,:].abs(),g[:,0].abs(),g[:,-1].max())**2) * 1e3
        errEdge =           (torch.max(zero,g[:, 0].abs().max()-maxGedge) * 1e3)**2
        errEdge = errEdge + (torch.max(zero,g[:,-1].abs().max()-maxGedge) * 1e3)**2
        
        if limits['gradMomZero']:
            errEdge = errEdge + (g.sum(1).abs().sum()*1e2) ** 2
        
        if limits['RF_mask'] is not None:
            pen_mask = ~limits['RF_mask']
            errEdge += (pulse.abs().sum(1) * pen_mask).sum() ** 2
            #print(f'RFmaskErr: {(pulse.abs().sum(1) * pen_mask).sum()}')
        
        errEdge = errEdge * 10
        # pulse power penalty
        penPow = ((pulse.abs()/1e2)**2).mean()
        
        
        mask = (target.abs() > 0.1);
        maskedFA = FA.masked_select(mask)
        a = torch.sin(maskedFA.abs()) * torch.exp(1j * maskedFA.angle())
        #sumSig = a.mean().abs()
        #penSum = 1/sumSig * limits['sumSig']
        sumSig = a.mean().abs() / np.sin(limits['targetFA']/180*np.pi)    
        if limits['sumSig'] > 0:
            #print(f'a: {a.mean().abs()}; tar: {np.sin(limits["targetFA"])}')
            penSum = torch.max(zero,limits['sumSig'] - sumSig)**2 * 1e6
        else:
            penSum = zero.clone()
            
        
        if VOPs is None:
            errSAR = 0
            localSAR=0
        else:
            pulse = pulse.real.nan_to_num(0.0) + 1j * pulse.imag.nan_to_num(0.0)
            localSAR=self.calcPulseSAR(pulse,VOPs,TR=TR,stepdur=stepdur)
            a = torch.max(zero,localSAR-limits['maxSAR'])
            # if a.isinf().sum() > 0:
            #     print('Warning:INF')
            #a = a.nan_to_num(9999)
            b = (a**2)*100
            errSAR = b#.abs()
            #errSAR = errSAR.nan_to_num(999.0)
            #errSAR = torch.abs(torch.max(zero,localSAR-limits['maxSAR'])**2*100)
        # total error:
        err = penMag + penSlice + penPhs + errVolt + errGrad + errSlew + errEdge + penPow + penSlew + errSAR + penSum
        if verbose:
            print( '========================')
            print(f'errVolt={errVolt[0]:9.6f}')
            print(f'errGrad={errGrad[0]:9.6f}')
            print(f'errSlew={errSlew[0]:9.6f}')
            print(f'errEdge={errEdge[0]:9.6f}')
            limsar = limits['maxSAR']
            print(f'errSAR ={errSAR[0]:9.6f} ({localSAR:6.3f}/{limsar:6.3f})')
            print( '------------------------')
            print(f'penMag ={penMag:9.6f} (RMSE)')
            print(f'penPhs ={penPhs:9.6f}')
            print(f'penSlc ={penSlice:9.6f}')
            print(f'penPow ={penPow:9.6f}')
            print(f'penSlew={penSlew:9.6f}')
            print(f'penSum ={penSum[0]:9.6f} (sumSig: {sumSig:6.3f})')
            print( '------------------------')
            print(f'err:    {err[0]:9.6f}')
            print( '========================')
        return err
    
    @classmethod
    def plotVec(self,pulse,g,dat_v,stepdur,limits,save=False,i=0,VOPs=None,TR=None):
        #if g is not None:
        #    vec = torch.cat((g.flatten()/limits['maxG'],optvec),0)
        #else:
        #    vec = optvec
        #pulse,g,k = splitVec(vec,dat_v['nTx'],limits=limits,stepdur=stepdur)
        rep = int(stepdur*1e6)
        pulse = pulse.detach()
        pulsePLT = pulse.repeat_interleave(rep,0)
        a = torch.nn.Upsample(scale_factor=rep,mode='linear',align_corners=False)
        gPLT = a(g.unsqueeze(0)).squeeze()
        #slew = torch.diff(gPLT,dim=1) / 1e-6
        slew = torch.diff(g,dim=1) / 10e-6#stepdur
        slewPLT = a(slew.unsqueeze(0)).squeeze()
        nsteps = pulse.shape[0] * rep
        FA = self.calcFA(dat_v,pulse,g,stepdur,t1=limits['t1'],t2=limits['t2'])
        dtc = dat_v['s'].dtype
        dev = dat_v['s'].device
        FA3D = torch.zeros(dat_v['mask3D'].shape,dtype=dtc,device=dev)
        FA3D[dat_v['mask3D']] = (FA + 1j * 0)
        FA3D = FA3D.detach().cpu()
        
        k = gtokTX(g,stepdur) 
        k_ = k.cpu().detach().numpy()
        
        maxk = np.abs(k_).max()
        if save:
            maxk = 150
        elif maxk==0:
            maxk = 1
        
        clim=[0,limits['plotscale']]
        pos = [int(x/2) for x in FA3D.shape]
        #pos[2] = pos[2] + 15
        pos=[32,32,41]
        cmap='turbo'
        
        f = plt.figure(figsize=[15,15])
        plt.subplot(441)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        #plt.title(title)
        if FA3D.shape[0] == 4*FA3D.shape[1]:
            ind = FA3D.shape[0] // 2
            im1 = FA3D[:ind,...]
            im2 = FA3D[ind:,...]
            FA3D = torch.cat((im1,im2),1)
            pos[0] = pos[0]//4
            pos[1] = pos[1]*1
        elif FA3D.shape[0] == 6*FA3D.shape[1]:
            ind = FA3D.shape[0] // 3
            im1 = FA3D[:ind,...]
            im2 = FA3D[ind:2*ind,...]
            im3 = FA3D[2*ind:,...]
            FA3D = torch.cat((im1,im2,im3),1)
            pos[0] = pos[0]//6
            pos[1] = pos[1]*1
        elif FA3D.shape[0] == 9*FA3D.shape[1]:
            ind = FA3D.shape[0] // 3
            im1 = FA3D[:ind,...]
            im2 = FA3D[ind:2*ind,...]
            im3 = FA3D[2*ind:,...]
            FA3D = torch.cat((im1,im2,im3),1)
            pos[0] = pos[0]//9
            pos[1] = pos[1]*1
        plt.imshow(FA3D[:,:,pos[2]].abs().T, cmap=cmap, origin='lower',clim=clim)
        plt.title('FA [deg]')
        plt.axis('off')
        plt.subplot(442)
        plt.imshow(FA3D[:,pos[1],:].abs().T, cmap=cmap, origin='lower',clim=clim)
        plt.axis('off')
        plt.subplot(445)
        plt.imshow(FA3D[pos[0],:,:].abs().T, cmap=cmap, origin='lower',clim=clim)
        plt.axis('off')
        plt.colorbar()
        plt.subplot(446)
        plt.imshow(FA3D[:,pos[1],:].angle().T, cmap=cmap, origin='lower',clim=[-torch.pi,torch.pi])
        plt.title('phase')
        plt.axis('off')
        plt.colorbar()
        ax = plt.subplot2grid((4,4), (2,2),projection='3d')
        #ax = fig.add_subplot(1, 2, 2, projection='3d')
        #ax = plt.subplot(449,projection= '3d')
        ax.plot(k_[0,:],k_[1,:], 'gray', zdir='z', zs=-maxk)
        ax.plot(k_[1,:],k_[2,:], 'gray', zdir='x', zs=-maxk)
        ax.plot(k_[0,:],k_[2,:], 'gray', zdir='y', zs=maxk)
        ax.plot3D(k_[0,:],k_[1,:],k_[2,:],'b')
        plt.xlim([-maxk,maxk])
        plt.ylim([-maxk,maxk])
        ax.set_zlim([-maxk,maxk])
        plt.title('k-space [1/m]')
        
        ax = plt.subplot2grid((4,4), (0,2))
        # plt.plot(pulsePLT[:,:].cpu().abs())
        # plt.title('RF mag [V]')
        # if save:
        #     plt.ylim([0,limits['maxRF']])
        # plt.xlim(0,nsteps)
        plt.imshow(pulsePLT.abs().cpu().T,cmap='turbo',origin='lower',aspect='auto',interpolation='none')
        plt.colorbar(location="top",label="Pulse magnitude [V]")
        #plt.title('pulse voltage [V]')
        plt.xlabel('time [us]')
        plt.ylabel('TX channel')
        ax.yaxis.set_ticks(np.arange(0, 16, 2))
        if save:
            plt.clim([0,limits['maxRF']])
        else:
            plt.clim([0,pulse.abs().max()])
        
        ax = plt.subplot2grid((4,4), (1,2))
        # plt.plot(pulsePLT[:,:].cpu().angle()*180/np.pi)
        # plt.ylim([-180,180])
        # plt.xlim(0,nsteps)
        # plt.title('RF phs [deg]')
        plt.imshow(pulsePLT.angle().cpu().T*180/np.pi,cmap='hsv',origin='lower',aspect='auto',interpolation='none')
        plt.colorbar(location="top",label="Pulse phase [deg]")
        #plt.title('pulse voltage [V]')
        plt.xlabel('time [us]')
        plt.ylabel('TX channel')
        ax.yaxis.set_ticks(np.arange(0, 16, 2))
        plt.clim([-180,180])
        
        plt.subplot2grid((4,4), (2,0))
        plt.plot(gPLT.cpu().detach().transpose(1,0)*1e3)
        if save:
            plt.ylim(-limits['maxG']*1e3,limits['maxG']*1e3)
        plt.xlim(0,nsteps)
        plt.title('gradients [mT/m]')
        plt.legend(['x','y','z'])
        plt.subplot2grid((4,4), (2,1))
        plt.plot(slewPLT.cpu().detach().transpose(1,0))
        if save:
            plt.ylim(-limits['maxSlew'],limits['maxSlew'])
        plt.xlim(0,nsteps)
        plt.title('slew [T/m/s]')
        plt.legend(['x','y','z'])
        
        # SAR info
        if VOPs is not None:
            ax = plt.subplot2grid((4,4), (0,3))
            SAR = self.calcPulseSAR(pulse, VOPs,TR,stepdur).detach().cpu()
            plt.bar(0,(SAR/limits['maxSAR']*100))
            plt.ylim([0,120])
            plt.title('SAR [%]')
            ax.get_xaxis().set_visible(False)
            ax.set_box_aspect(3)
        
        # FA magnitude error
        if limits['targetFA'] is not None:
            targ = limits['targetFA']
            ax = plt.subplot2grid((4,4), (1,3))
            penMag = ((FA.abs()-targ)**2).mean().sqrt().detach().cpu() 
            #penMag = ((FA.abs()-target.abs())**2).mean().sqrt().detach().cpu() 
            plt.bar(0,(penMag))
            plt.ylim([0,targ/5])
            plt.title('RMSE')
            ax.get_xaxis().set_visible(False)
            ax.set_box_aspect(3)
        
        # text info
        ax = plt.subplot2grid((4,4), (2,3))
        plt.axis('off')
        if VOPs is not None:
            t1 = f"SAR: {SAR:5.2f} W/kg"
            plt.text(0,1,t1,fontsize=14)
        if limits['targetFA'] is not None:
            t1 = f"RMSE: {penMag:5.2f}"
            plt.text(0,0.8,t1,fontsize=14)
        if save:
            t1 = f"Iter: {i:6}"
            plt.text(0,0.6,t1,fontsize=14)
            
        plt.subplots_adjust(hspace=0.4)
        plt.subplots_adjust(wspace=0.4)
        
        f.suptitle(self.figtitle)
        
        # plot rectangle around FA-maps
        rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.00, 0.48), 0.50, 0.48, fill=False, color="k", lw=2, zorder=1000, transform=f.transFigure, figure=f)
        f.patches.extend([rect])
        # plot rectangle around pulse
        rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.50, 0.48), 0.27, 0.48, fill=False, color="k", lw=2, zorder=1000, transform=f.transFigure, figure=f)
        f.patches.extend([rect])
        # plot rectangle around k-space info
        rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.00, 0.25), 0.77, 0.23, fill=False, color="k", lw=2, zorder=1000, transform=f.transFigure, figure=f)
        f.patches.extend([rect])
        # plot rectangle around SAR/RMSE info
        rect = plt.Rectangle(
        # (lower-left corner), width, height
        (0.77, 0.25), 0.22, 0.71, fill=False, color="k", lw=2, zorder=1000, transform=f.transFigure, figure=f)
        f.patches.extend([rect])
        
        f.text(0.01,0.935,'a)',fontsize=24)
        f.text(0.51,0.935,'b)',fontsize=24)
        f.text(0.01,0.455,'c)',fontsize=24)
        f.text(0.775,0.935,'d)',fontsize=24)
        if save:
            filename = f'img/{self.figtitle}_{i:08}.png'
            f.savefig(filename)
            filename = f'img/{self.figtitle}_{i:08}.mat'
            mat = dict()
            mat['pulse'] = pulse.cpu().numpy()
            mat['g'] = g.cpu().numpy()
            mat['k'] = k.cpu().numpy()
            mat['stepdur'] = stepdur
            savemat(filename,mat)
        plt.show()
        return f
    
    def splitVec(self,optvec,nTx,limits,stepdur=10e-6):
        ntimes = int(optvec.shape[0] / (nTx+3))
        g = optvec[:3*ntimes].reshape([3,-1]).real * limits['maxG']
        if limits['maxG'] < 1e-6:
            g = g * 0
        #k = gtokTX(g,stepdur)
        #g = optvec[:3*ntimes].reshape([3,-1])*gnorm
        pulseS = optvec[3*ntimes:(nTx+3)*ntimes]
        #pulseS = optvec[3*ntimes:(nTx*2+3)*ntimes]
        nps = int(pulseS.shape[0]/2)
        pulse = pulseS.reshape(ntimes,nTx) * limits['maxRF']
        return pulse,g
    
    def splitVecSpins(self,optvec,nTx,limits,stepdur=10e-6):
        ntimes = int((optvec.shape[0]-5) / (nTx))
        k = spins(ntimes,kmax=optvec[0].real,alpha=optvec[1].real,beta=optvec[2].real,u=optvec[3].real,v=optvec[4].real,dev=optvec.device)
        g = ktogTX(k,stepdur)
        if limits['maxG'] < 1e-6:
            g = g * 0
        pulseS = optvec[5:(nTx)*ntimes+5]
        nps = int(pulseS.shape[0]/2)
        pulse = pulseS.reshape(ntimes,nTx) * limits['maxRF']
        return pulse,g
    
    def splitVecKT(self,optvec,nTx,limits,stepdur=10e-6,pulse_dur=1000,):
        ntimes = int((optvec.shape[0]) / (nTx+5))
        grad_blip_dur = optvec[0:ntimes].abs()
        subpulse_dur = optvec[ntimes:2*ntimes].abs()
        grad_mags = optvec[2*ntimes:5*ntimes].reshape((3,-1)).real
        pulse_mags = optvec[5*ntimes:].reshape((ntimes,nTx))
        # start with empty vectors
        g = torch.zeros(3,pulse_dur,device=optvec.device,dtype=optvec.abs().dtype)
        pulse = torch.zeros(pulse_dur,nTx,device=optvec.device,dtype=optvec.dtype)
        # now fill the vectors
        ts = 0 # current time step
        for n in range(ntimes):
            # do grads
            end = ts+grad_blip_dur[n].int()
            if end > pulse_dur:
                end = pulse_dur
            for gx in range(3):
                g[gx,ts:end] = self.blip(grad_blip_dur[n],grad_mags[gx,n])
            ts = end
            
            # do RF
            end = ts+subpulse_dur[n].int()
            if end > pulse_dur:
                end = pulse_dur
            for k in range(nTx):
                pulse[ts:end,k] = pulse_mags[n,k]
            ts = end
            pass
            
        return pulse,g
    
    
    def splitVecSpokes(self,optvec,nTx,limits,stepdur=10e-6,pulse_dur=1000,nspokes=3,rotx=0.0,thick=0.035):
        dev = optvec.device
        dt = optvec.abs().dtype
        dtc = optvec.dtype
        ntimes = int((optvec.shape[0]) / (nTx+3))
        grad_blip_dur = 10
        subpulse_dur = int((pulse_dur - (grad_blip_dur * (nspokes-1))) // (nspokes+0.5))
        rew_dur = pulse_dur - (subpulse_dur*nspokes + grad_blip_dur * (nspokes-1))-1
        grad_mags = optvec[:3*ntimes].reshape((3,-1)).real * limits['maxG'] / grad_blip_dur
        pulse_mags = optvec[3*ntimes:].reshape((ntimes,nTx))
        # start with empty vectors
        gx = torch.zeros(1,device=optvec.device,dtype=optvec.abs().dtype).requires_grad_(True)
        gy = torch.zeros(1,device=optvec.device,dtype=optvec.abs().dtype).requires_grad_(True)
        gz = torch.zeros(1,device=optvec.device,dtype=optvec.abs().dtype).requires_grad_(True)
        pulse = torch.zeros(1,nTx,device=optvec.device,dtype=optvec.dtype).requires_grad_(True)
        # now fill the vectors
        ts = 0 # current time step
        
        bwt = 6
        nramp = 6 # for sli-sel grad
        empirical_factor = 0.75 # 
        # RF basic shape
        sinc_dur = subpulse_dur - 2*nramp
        inp = torch.linspace(-bwt/2,bwt/2,sinc_dur,device=dev,dtype=dtc)
        sinc = torch.sinc(inp)
        filt = torch.hamming_window(sinc_dur).to(dev);
        fsinc = (sinc * filt).requires_grad_(True)
        # now assemble all subpulses
        for n in range(ntimes):
            # grad blips
            if n>0:
                gx = torch.cat((gx,self.blip(grad_blip_dur,grad_mags[0,n])))
                gy = torch.cat((gy,self.blip(grad_blip_dur,grad_mags[1,n])))
                gz = torch.cat((gz,self.blip(grad_blip_dur,grad_mags[2,n])*0))
                pulse = torch.cat((pulse,torch.zeros(grad_blip_dur,nTx,device=optvec.device,dtype=optvec.dtype)))
            
            # do sli-sel grad
            bw = bwt / (subpulse_dur * stepdur * empirical_factor)
            gradamp = bw / smallTipAngle.gamma / thick * 2 * torch.pi
            gradamp = torch.tensor([gradamp],device=dev,dtype=dt)
            # every 2nd spoke goes "backwards"
            if n%2 > 0:
                gradamp = -gradamp
                        
            gzs = torch.zeros(0,device=dev,dtype=dt).requires_grad_(True)
            for i in range(nramp):
                gzs = torch.cat((gzs,gradamp/(nramp+1)*(i+1)))
            for i in range(subpulse_dur-(2*nramp)-1):
                gzs = torch.cat((gzs,gradamp))
            for i in range(nramp):
                gzs = torch.cat((gzs,gradamp-gradamp/(nramp+1)*(i+1)))
            # rotate sli-sel grad            
            gxs = gzs * 0
            gys =  - gzs * self.sind(-rotx)
            gzs = - gzs * self.cosd(-rotx)
            # append to grads
            gx = torch.cat((gx,gxs))
            gy = torch.cat((gy,gys))
            gz = torch.cat((gz,gzs))
            gx = torch.cat((gx,torch.zeros(1,device=dev,dtype=dt)))
            gy = torch.cat((gy,torch.zeros(1,device=dev,dtype=dt)))
            gz = torch.cat((gz,torch.zeros(1,device=dev,dtype=dt)))
            # do RF
            pulse = torch.cat((pulse,torch.zeros(nramp,nTx,device=dev,dtype=dtc)))
            thisPulse = pulse_mags[n,:] * limits['maxRF'] * fsinc.unsqueeze(1)
            pulse = torch.cat((pulse,thisPulse))
            pulse = torch.cat((pulse,torch.zeros(nramp,nTx,device=dev,dtype=dtc)))
        
        # slice rewinder
        #rew_dur = 15
        gradamp = -gradamp
        gzs = torch.zeros(0,device=dev,dtype=dt).requires_grad_(True)
        for i in range(nramp):
            gzs = torch.cat((gzs,gradamp/(nramp+1)*(i+1)))
        for i in range(rew_dur-(nramp*2)-1):
            gzs = torch.cat((gzs,gradamp))
        for i in range(nramp):
            gzs = torch.cat((gzs,gradamp-gradamp/(nramp+1)*(i+1)))
        # rotate rewinder           
        gxs = gzs * 0
        gys =  - gzs * self.sind(-rotx)
        gzs = - gzs * self.cosd(-rotx)
        gx = torch.cat((gx,gxs))
        gy = torch.cat((gy,gys))
        gz = torch.cat((gz,gzs))
        gx = torch.cat((gx,torch.zeros(1,device=optvec.device,dtype=optvec.abs().dtype)))
        gy = torch.cat((gy,torch.zeros(1,device=optvec.device,dtype=optvec.abs().dtype)))
        gz = torch.cat((gz,torch.zeros(1,device=optvec.device,dtype=optvec.abs().dtype)))
        pulse = torch.cat((pulse,torch.zeros(rew_dur,nTx,device=optvec.device,dtype=optvec.dtype)))
        
        grad = torch.stack([gx,gy,gz],0)
        #print(grad.shape)
        #print(pulse.shape)
        #plt.plot(grad.cpu().detach().numpy()[:,:].T)
        #print(pulse.shape)
        #print(grad.shape)
        mat = dict()
        mat['g'] = grad.cpu().detach().numpy()
        mat['p'] = pulse.abs().cpu().detach().numpy()
        savemat('mat.mat',mat)
        print(grad[2,:].detach().cpu().numpy())
        print(pulse[:,0].detach().abs().cpu().numpy())
        return pulse,grad
    
    
    def splitPulse(self,optvec,nTx,limits,stepdur=10e-6):
        ntimes = int(optvec.shape[0] / (nTx*2))
        pulseS = optvec
        nps = int(pulseS.shape[0]/2)
        pulse = (pulseS[0:nps] + 1j * pulseS[nps:]).reshape(ntimes,nTx) * limits['maxRF']
        return pulse
    
    def blip(steps,mag):
        dev = mag.device
        out = torch.zeros(steps,device=dev).requires_grad_(True)
        
        if steps%2 == 0:
            end1 = int(steps/2)
            out1 = torch.linspace(0,1,end1,device=dev) * mag
            out2 = torch.linspace(1,0,end1,device=dev) * mag
            out = torch.cat((out1,out2))
        else:
            end1 = int(steps/2)+1
            out1 = torch.linspace(0,1,end1,device=dev) * mag
            mag2 = out1[-2]
            out2 = torch.linspace(1,0,end1-1,device=dev) * mag2
            out = torch.cat((out1,out2))
        return out
        
    def cosd(x):
        return np.cos(np.deg2rad(x))    
    def sind(x):
        return np.sin(np.deg2rad(x))
