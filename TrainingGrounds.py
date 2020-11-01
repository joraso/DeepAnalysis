# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:00:27 2020

Testing and developing space.

@author: Joe Raso
"""

from Models import *
from Priors import *
from Layers import *
from Maps import InvMap

import matplotlib.pyplot as plt


if __name__ == '__main__':
    
#    model = DoubleWell()
#    r1 = model.randos(2000, loc=[-1.7, 0], scale=0.2)
#    r2 = model.randos(2000, loc=[1.7, 0], scale=0.2)
#    train = np.concatenate([r1, r2], axis=0)
#    plt.scatter(r1[:,0],r1[:,1],c='b')
#    plt.scatter(r2[:,0],r2[:,1],c='r')
    
#    model = MullerBrown()
#    X1, X2, X3 = model.MCsample(1500,100)
#    train = np.concatenate([X1, X2, X3], axis=0)
    
    model = ParticleDimer()
#    train = model.MCsample(4000,100)
    train = np.loadtxt("PDMCsamples.csv", delimiter=',')
    
    gauss = Normal(model.dim)
    
    def createRealNVP(dim):
        if dim == 2:
            part = Partition(2, split_indices=[[0], [1]])
        else:
            part = Partition(dim)
        S1 = AffineLayer(len(part.split_indices[1]))
        T1 = AffineLayer(len(part.split_indices[1]))
        S2 = AffineLayer(len(part.split_indices[0]))
        T2 = AffineLayer(len(part.split_indices[0]))
        return RealNVP([S1, T1, S2, T2], part)
        
    real1 = createRealNVP(model.dim)
    real2 = createRealNVP(model.dim)
    real3 = createRealNVP(model.dim)
    
    maple = InvMap([real1, real2, real3], model, gauss)
    maple.connect_all()

    maple.MLCompile()
    maple.MLTrain(train, 100)
#    plt.plot(maple.MLhistory)
    
#    maple.KLCompile()
#    maple.KLTrain(200)
#    plt.plot(maple.KLhistory)    
    
    
#    nsample = 1000
#    x, E, w = maple.Sample(2000)
#    model.FE_1D(x, w, nbins=100)
#    model.potential_surface()
#    plt.scatter(x[:,0],x[:,1])
