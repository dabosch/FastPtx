#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 18:46:47 2022

@author: dario
"""
import torch
import math
import numpy as np

__all__ = ['gtokTX', 'ktogTX', 'spins', 'calc_slew','plotTrajectory']

gamma = 267522189.9851 # in rad/s/T

def gtokTX(g,ts=10e-6):
    # number of k-space locations
    nK = g.shape[1];
    k = torch.zeros(g.shape,device=g.device,dtype=g.dtype)
    for i in range(nK):
        k[:,i] = ts * g[:,i+1:].sum(1) + ts/2 * g[:,i]
    k = k*(-gamma);
    return k

def ktogTX(k,ts=10e-6):
    # number of k-space locations
    nK = k.shape[1];
    g = torch.zeros(k.shape,device=k.device,dtype=k.dtype)
    for i in range(nK-2,0,-1):
        g[:,i+1] = (k[:,i] - ts * g[:,i+1:].sum(1) )  / (ts)/2
    g = g/(-gamma);
    return g

def calc_slew(g,ts=10e-6):
    slew = torch.diff(g,dim=1) / ts
    return slew

def spins(nT, alpha=10, kmax=50, beta=0.5,u=8,v=5,dev=torch.device('cpu'), dt=torch.float32):
    from torch import sin,cos
    
    kr = torch.zeros(3,nT,device=dev,dtype=dt)
    k = torch.zeros(3,nT,device=dev,dtype=dt)
    
    u_ = u * 2*torch.pi / 1e3
    v_ = v * 2*torch.pi / 1e3
    
    #for t in range(nT):
    #    k[0,t] = kmax / (1 + torch.exp(alpha *  ((t/nT) - beta) ))
    t = torch.tensor([int(x) for x in range(nT)],device=dev,dtype=dt)
    
    kr[0,:] = torch.real(kmax / (1 + torch.exp(alpha *  ((t/nT) - beta) )))
    # interpolate last radius-steps towards 0
    n_interp = int(nT/10)
    kr[0,-n_interp:] = torch.linspace(n_interp-1,0,n_interp,device=dev)/n_interp * kr[0,-n_interp]
    
    kr[1,:] = torch.real( torch.linspace(0,nT,nT,device=dev) * (u_) )
    kr[2,:] = torch.real( torch.linspace(0,nT,nT,device=dev) * (v_) )
    
    k[0,:] = sin( kr[1,:] ) * cos ( kr[2,:] ) * kr[0,:]
    k[1,:] = sin( kr[2,:] ) * cos ( kr[1,:] ) * kr[0,:]
    k[2,:] = cos( kr[1,:] ) * kr[0,:]

    # smoothen beginning
    r = kmax/10
    for i in range(int(r)):
        k[:,i+1] = k[:,i] + (k[:,i+1]-k[:,i]) * i/r
    
    
    return k

def spiral(nT, alpha=10,kmax=50,n=5,dev=torch.device('cpu'), dt=torch.float32):
    k = torch.zeros(3,nT,device=dev,dtype=dt)
    for t_ in range(nT):
        t = 1-t_/(nT)
        k[0,t_] = np.real(kmax * t**alpha * np.exp(1j * 2*torch.pi*n * t))
        k[1,t_] = np.imag(kmax * t**alpha * np.exp(1j * 2*torch.pi*n * t))
    return k

def plotTrajectory(k):
    import matplotlib.pyplot as plt 

    plt.subplot(221)
    plt.plot(k.T)  
    plt.legend(['kx','ky','kz'])
    plt.subplot(222)
    maxk = k.abs().max()
    ax = plt.axes(projection='3d')
    ax.plot3D(k[0,:],k[1,:],k[2,:]);
    plt.xlim([-maxk,maxk])
    plt.ylim([-maxk,maxk])
    ax.set_zlim([-maxk,maxk])
    plt.subplot(223)
    g = ktogTX(k)
    plt.plot(g.T)
    plt.legend(['gx','gy','gz'])
    slew = calc_slew(g);
    plt.subplot(224)
    plt.plot(slew.T)
    plt.legend(['x','y','z'])
    plt.show()
    
    
def test():
    k = spins(600,kmax=140,alpha=10,beta=0.5,u=8,v=5)
    print(k)
    
    
