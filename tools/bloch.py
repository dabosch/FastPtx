#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:57:23 2022

@author: gadgetron
"""
from tools.smallTipAngle import smallTipAngle
import torch
import numpy as np
import time

class bloch (smallTipAngle):
    
    figtitle = "bloch"
    stepiter = 300
    
    def my_arccos(self,inTensor):
        outTensor = inTensor.clone()
        mask = (inTensor > 1.0)
        mask2 = (inTensor < -1.0)
        mask3 = (inTensor >= -1.0) + (inTensor <= 1.0)
        outTensor.masked_fill(mask,1.0)
        outTensor.masked_fill(mask2,-1.0)
        outTensor = torch.acos(outTensor)
        return outTensor
    
    @classmethod
    def calcFA(self,dat_v,pulse,g,stepdur=10e-6,t1=None,t2=None):
        eps=1e-7
        #t = time.time()
        mag = self.bloch(self,dat_v,pulse,g,stepdur=stepdur,t1=t1,t2=t2)
        #elapsed = time.time() - t
        #print(f'elapsed time bloch: {elapsed}')
        #mag = mag.clamp(-1.0+eps,1.0-eps)
        #mag = mag.to(torch.float64)
        # We introduce this inaccuracy so that the gradients never become NaN.
        
        r = (mag**2).sum(1).sqrt()
        if((r==0).any()):
            print('warning: r was zero in calcFA')
            r = r + eps
        cos_theta = (mag[:,2]/r).to(torch.float64).clamp(min=-1+eps,max=1-eps)
        #cos_theta.clamp(-1,1)
        theta = torch.arccos(cos_theta).to(torch.float32)#.clamp(min=0+eps,max=torch.pi-eps)
        #theta = cos_theta
        phi = torch.arctan(mag[:,1]/mag[:,0]) + (mag[:,0]/mag[:,0].abs()+1)*torch.pi/2 + torch.pi
        alpha = theta * torch.exp(1j * phi)
        if theta.isnan().sum() > 0:
            #raise Exception('nan nan nan nan batmaaaanan')
            print('Warning: nan nan nan nan batmaaaaaanan')
        FA = alpha / torch.pi * 180
        return FA#.to(torch.float32)
    
    @classmethod
    def calcFA3D(self,dat_v,pulse,k=None,stepdur=10e-6,t1=None,t2=None):
        FA = self.calcFA(dat_v,pulse,k,stepdur,t1=t1,t2=t2)
        dtc = dat_v['s'].dtype
        dev = dat_v['s'].device
        FA3D = torch.zeros(dat_v['mask3D'].shape,dtype=dtc,device=dev)
        FA3D[dat_v['mask3D']] = FA
        #FA3D = FA3D.detach().cpu()
        return FA3D

    def bloch(self,dat_v,pulse,g,stepdur=10e-6,t1=None,t2=None):
        # sanitize input values
        if pulse.isnan().any():
            pulse = pulse.real.nan_to_num(0.0) + 1j * pulse.imag.nan_to_num(0.0)
        if g.isnan().any():
            g = g.nan_to_num(0.0)
        #e1 = stepdur/t1
        #e2 = stepdur/t2
        #eff_b1 = torch.einsum('tk,vk->tv',pulse,dat_v['s'])
        eff_b1 = pulse@dat_v['s'].T
        rotx = eff_b1.real * self.gamma * stepdur
        roty = eff_b1.imag * self.gamma * stepdur
        # off-resonance components in rad/s (caused by gradients and B0)
        offres = -( (g.T@dat_v['dist']) * self.gamma - dat_v['b0'])
        rotz = offres * stepdur

        mag = torch.zeros(dat_v['s'].shape[0],3,device=dat_v['s'].device,dtype=g.dtype)
        mag[:,2] = 1
        mag = mag.requires_grad_(True)
        
            
        if t2 is None:
            e2 = 1
        else:
            e2 = np.exp(-stepdur/t2)
        if t1 is None:
            e1 = 1
        else:
            e1 = np.exp(-stepdur/t1)
        
        
        for t in range(g.shape[1]):
            mag1 = self._precess(self,rotx[t,:], roty[t,:], rotz[t,:], mag.clone());
            if mag1.isnan().any():
                # undo this step
                mag1 = mag.clone()
                print('Warning: nan during _precess!')
                
            mag = self._relax(self,e1,e2,mag1.clone())
            if mag.isnan().any():
                # undo this step
                mag = mag1.clone()
                print('Warning: nan during _relax!')
                print(f'e1: {e1}, e2: {e2}')
        
        return mag
    
    def _relax(self,e1,e2,mag):
        #magO = torch.zeros(mag.shape,device=mag.device,dtype=mag.dtype)
        magx = mag[:,0] * e2
        magy = mag[:,1] * e2
        magz = (mag[:,2] - 1.0)* e1 + 1.0
        magO = torch.stack([magx,magy,magz],1)
        return magO#.clamp(-1.0,1.0)
        #a = [int(i) for i in range(magO.shape[0])]
        #indices = torch.tensor([a,2])
        #torch.index_put(mag0,(1,2),(3,4))
        #[torch.tensor([0, 0], device=DEVICE)], torch.tensor(1., device=DEVICE), accumulate=True
        #mag0.index_put(indices,(1)) #(mag[:,2] - 1.0)* e1 + 1.0)
        #magO[:,2] = (mag[:,2] - 1.0)* e1 + 1.0
        
        #mx = mag[:,0] * e2
        #my = mag[:,1] * e2
        #mz = 1.0 + e1 * (mag[:,2] - 1.0)* e2
        #mag = torch.stack((mx,my,mz),1)
        #return magO


    def _precess(self,bx, by, bz, mag):

        #double b, c, k, s, nx, ny, nz;
        b = (bx**2 + by**2 + bz**2).sqrt()
        # division is slow, so do only once
        rb = 1/b
        # if absolute value of the b-fields is zero, division will result in nan.
        # We'll convert that to an integer so we don't have nans in the future
        rb.nan_to_num(1.0)
        #b.masked_fill(b==0,1)
        # b = betrag des b1=vectors
        # bx,by,bz wird normiert,
        # nx,ny,nz ist der magnetisierungsvektor
        bx = bx * rb
        by = by * rb
        bz = bz * rb
        
        #bx = bx.nan_to_num(0.0)
        #by = by.nan_to_num(0.0)
        #bz = bz.nan_to_num(0.0)
        #b = b.nan_to_num(0.0)
        
        c = (torch.sin(0.5 * b)**2)*2
        s = torch.sin(b)
        nx = mag[:,0]
        ny = mag[:,1]
        nz = mag[:,2]
        
        #k = mag-vector o b-vector (skalarprodukt)
        k = nx * bx + ny * by + nz * bz
        
        add = torch.zeros(mag.shape[0],3,device=mag.device,dtype=mag.dtype)
        add[:,0] = (bx*k-nx)*c + (ny*bz-nz*by)*s
        add[:,1] = (by*k-ny)*c + (nz*bx-nx*bz)*s
        add[:,2] = (bz*k-nz)*c + (nx*by-ny*bx)*s
        
        if add.isnan().any():
            print('Warning: nan in add (_precess l173!)')
        
        #addX = (bx*k-nx)*c + (ny*bz-nz*by)*s
        #addY = (by*k-ny)*c + (nz*bx-nx*bz)*s
        #addZ = (bz*k-nz)*c + (nx*by-ny*bx)*s
        #add = torch.stack((addX,addY,addZ),1)
        
        #nxN = mag[:,0] + (bx*k-nx)*c + (ny*bz-nz*by)*s
        #nyN = mag[:,1] + (by*k-ny)*c + (nz*bx-nx*bz)*s
        #nzN = mag[:,2] + (bz*k-nz)*c + (nx*by-ny*bx)*s

        #magO = torch.stack((nxN,nyN,nzN),1)
        magO = (mag + add)
        
        return magO.clamp(-1.0,1.0)
    