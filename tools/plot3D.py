#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 17:27:37 2022

@author: dario
"""
import matplotlib.pyplot as plt

def plot3D(img,title=None,cmap='turbo',clim=None,pos = None):
    img = img.numpy()
    if pos is None:
        pos = [int(x/2) for x in img.shape[0:3]]
    if clim is None:
        clim = [img.min(),img.max()]
    f = plt.figure()
    plt.subplot(221)
    plt.title(title)
    plt.imshow(img[:,:,pos[2]], cmap=cmap, origin='lower',clim=clim)
    plt.subplot(222)
    plt.title(title)
    plt.imshow(img[:,pos[1],:], cmap=cmap, origin='lower',clim=clim)
    plt.subplot(223)
    plt.title(title)
    plt.imshow(img[pos[0],:,:], cmap=cmap, origin='lower',clim=clim)
    plt.colorbar()
    plt.show()
    return f


