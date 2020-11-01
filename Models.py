# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:25:03 2020
Contains sample 'Model' objects

model - archetypal object

utilities:
MetropolisSample() - to provide MC sampling of simple models.

Toy Models:
DoubleWell
MullerBrown
PaticleDimer

@author: Joe Raso
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class model(object):
    def __init__(self, dim, params):
        """The Archetypal model object."""
        self.dim = dim
        self.__dict__.update(params)
    def energy(self, x):
        """every model object must have an energy function that returns
        the reduced energy for a given configuration x"""
        return 0

# Utilities ===================================================================

def MetropolisSample(model, seed, nsample, nsteps,
                     temp=1.0, sigma=0.1):
    """For generating samples from simple models using the Metropolis
    algorithm. Samples are generated from parallel runs of nsteps."""
    beta = 1/temp
    X = seed.copy()*np.ones((nsample, model.dim))
    E = model.energy(X)
    Xnew = np.zeros((nsample,model.dim))
    for i in range(nsteps):
        Move = np.random.normal(size=(nsample, model.dim), scale=sigma)
        Xnew = X + Move
        Enew = model.energy(Xnew)
        dE = Enew - E
        P = np.exp(-dE*beta)
        test = np.random.rand(nsample)
        accept = np.tile(np.expand_dims((test < P),axis=1), model.dim)*1
        X += accept*Move
        E += accept[:,0]*dE
    return X
    

# Toy Models ==================================================================
        
class DoubleWell(model):
    def __init__(self, params = {'a4':1.0, 'a2':6.0, 'a1':1.0, 'b':1.0}):
        super().__init__(2, params)
    def energy(self, x):
        E = self.b*(x[:,1]**2)
        E += self.a1 - self.a2*(x[:,0]**2) + self.a4*(x[:,0]**4)
        return E
    def randos(self, n, loc=0.0, scale=1.0):
        """a testing functionality, draws n random points."""
        return np.random.normal(loc=loc, scale=scale, size=(n,self.dim))
    def potential_surface(self, window=[-3,3,-5,5]):
        """Copied (mostly) from Noe. Plots the 2D potential."""
        xgrid = np.linspace(window[0], window[1], 100)
        ygrid = np.linspace(window[2], window[3], 100)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        X = np.vstack([Xgrid.flatten(), Ygrid.flatten()]).T
        E = self.energy(X)
        E = E.reshape((100, 100))
        plt.contourf(Xgrid, Ygrid, E, 50, cmap='jet', vmax=4)
        plt.xticks(window[:2])
        plt.yticks(window[2:])
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{2}$')
    def FE_1D(self, xsamples, wsamples, nbins=100):
        """Uses samples from a map along with their weights to show the
        predicted energy surface along the x-axis."""
        # Procedure for calculating the free energy
        counts, bins =np.histogram(xsamples[:,0], bins=nbins, weights=wsamples)
        p = counts / np.sum(counts)   # the approximation of probability, px(x)
        centers = bins[:-1] + 0.5*(bins[1]-bins[0])
        FE = -np.log(p)
        FE -= np.min(FE)
        plt.scatter(centers,FE, c='g')
        xa = np.linspace(-3,3,100)
        fa = self.energy(np.array([xa,np.zeros(len(xa))]).transpose())
        fa -= np.log(np.sqrt(np.pi))
        fa -= np.min(fa)
        plt.plot(xa,fa)
        plt.xlim([-4,4])
        plt.ylim([-1,20])
        plt.ylabel(r'$\Delta F$')
        plt.xlabel(r'$x_{1}$')
        
class MullerBrown(model):
    def __init__(self, params = {'alpha':0.1, 'A':[-200, -100, -170, 15],
                'a':[-1, -1, -6.5, 0.7], 'b':[0, 0, 11, 0.6],
                'c':[-10, -10, -6.5, 0.7], 'x_j':[1, 0, -0.5, -1],
                'y_j':[0, 0.5, 1.5, 1]}):
        super().__init__(2, params)
    def energy(self, coords):
        """Copied nearly verbatim from Lenny/Wei Tzu's project paper. Some
        (very) minor modifications were made to make it compatible with the
        rest of my code base. Namely, the pytorch implementation was replaced
        with a tensorflow implementation."""
        x = coords[:,0]; y = coords[:,1]
        if type(coords) == np.ndarray:
            E = np.zeros(x.shape)
            for A, a, b, c, xj, yj in zip(self.A, self.a, self.b,
                                          self.c, self.x_j, self.y_j):
                E += A*np.exp(a*(x-xj)**2 + b*(x-xj)*(y-yj) + c*(y-yj)**2)
            return self.alpha * E
        elif type(coords) == tf.Tensor:
            E = tf.zeros(tf.shape(x))
            for A, a, b, c, xj, yj in zip(self.A, self.a, self.b,
                                          self.c, self.x_j, self.y_j):
                E += A*tf.math.exp(a*(x-xj)**2 + b*(x-xj)*(y-yj) + c*(y-yj)**2)
            return self.alpha * E
    def potential_surface(self, window=[-1.5,1.25,-0.5,2]):
        """Copied (mostly) from Lenny/WeiTsu. Plots the 2D potential."""
        xgrid = np.linspace(window[0], window[1], 100)
        ygrid = np.linspace(window[2], window[3], 100)
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        X = np.vstack([Xgrid.flatten(), Ygrid.flatten()]).T
        E = self.energy(X)
        E = E.reshape((100, 100))
        plt.contourf(Xgrid, Ygrid, E, 500, cmap = "jet", vmin = -10, vmax = -3)
        plt.xticks(window[:2])
        plt.yticks(window[2:])
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{2}$')
    def xproj(self, x, vector=[1,-1]):
        """returns the scalar projection of the 2D points along the given
        vector."""
        return np.dot(x, vector) / np.sqrt(2)
    def cutaway(self, m=0.5, xrange=(-1.2,0.8), zero=None):
        """Plots a cross-section of the potential energy along y = -x + m"""
        x1 = np.linspace(xrange[0], xrange[1], num=100)
        x2 = -x1 + m
        x = np.vstack([x1,x2]).transpose()
        E = self.energy(x)
        if zero != None:
            E = E - zero
        xproj = self.xproj(x)
        plt.plot(xproj, E)
    def FE_1D(self, xsamples, wsamples, nbins=100):
        """Uses samples from a map along with their weights to show the
        predicted energy surface along the x-axis."""
        # Take the projection.
        xproj = self.xproj(xsamples)
        # Procedure for calculating the free energy
        counts, bins = np.histogram(xproj, bins=nbins, weights=wsamples)
        p = counts / np.sum(counts)   # the approximation of probability, px(x)
        centers = bins[:-1] + 0.5*(bins[1]-bins[0])
        FE = -np.log(p)
        E0 = np.min(FE)
        FE -= E0
        self.cutaway(zero=E0)
        plt.scatter(centers,FE, c='g')
        plt.ylabel(r'$\Delta F$')
        plt.xlabel(r'$x_{1}$')
    def MCsample(self, nsamples, nsteps):
        """Uses MC sampling to generate nsamples around each of the three wells
        in the potential. (returns the three as seperate trajectories, in case
        you want to do some different kinds of map testing)"""
        x0_1 = np.array([-0.5,1.5])
        x0_2 = np.array([-0.1, 0.5])
        x0_3 = np.array([1.0, 0.0])
        X1 = MetropolisSample(self, x0_1, nsamples, nsteps)
        X2 = MetropolisSample(self, x0_2, nsamples, nsteps)
        X3 = MetropolisSample(self, x0_3, nsamples, nsteps)
        return X1, X2, X3

# Imports specifically for the particle dimer model
from matplotlib.patches import Rectangle, Circle
from scipy.optimize import linear_sum_assignment

class ParticleDimer(model):
    def __init__(self, params={'nsolvent':36, # Number of solvent particles
                               'epsilon':1.0, 'sigma':1.1, # LJ parameters
                               'k_d':20, # dimer force constant
                               'd0':1.5, # transition state distance
                               'a':25, 'b':10, 'c':-0.5, # dimer energy params
                               'l_box':3, 'k_box':100, # box parameters
                               'k_restrain':0.0}):
        """Modeled after Lenny/WeiTsu's and Noe's work."""
        dim = params['nsolvent']*2 + 4
        super().__init__(dim, params)
        self.positionmask = np.ones((self.nsolvent + 2,
                                     self.nsolvent + 2), dtype=np.float32)
        for i in range(self.nsolvent + 2):
            self.positionmask[i, i] = 0.0
        # Make the initial reference state a dumb lattice state,
        # Can (and should) be reassigned later.
        self.reference = self.initial_positions(self.d0)
    def bond_distance_np(self, x):
        """Computes the distance in between dimer components."""
        d2 = (x[:, 0]-x[:, 2])**2 + (x[:, 1]-x[:, 3])**2
        return np.sqrt(d2)
    def bond_energy_np(self, x):
        """Energy between the dimer particles."""
        d2 = (x[:, 0]-x[:, 2])**2 + (x[:, 1]-x[:, 3])**2
        d4 = d2**2
        d = np.sqrt(d2) # Here is the thorn
        return 1/4*self.a*d4 - 1/2*self.b*d2 + self.c*d
    def LJ_energy_np(self, x):
        """Potential energy due to the solvent interactions. Taken from my last
        implementation of the lennard-jones"""
        xloc = x[:, 0::2]; yloc = x[:, 1::2]
        xloc = np.tile(np.expand_dims(xloc, 2), [1, 1, np.shape(xloc)[1]])
        yloc = np.tile(np.expand_dims(yloc, 2), [1, 1, np.shape(yloc)[1]])
        xdist = xloc - np.transpose(xloc, [0,2,1])
        ydist = yloc - np.transpose(yloc, [0,2,1])
        mask = np.tile(np.expand_dims(self.positionmask,0), (np.shape(x)[0],1,1))
        D2 = xdist**2 + ydist**2
        D2 = D2 - (1-mask) # set diag to 1 to avoid div by 0
        D2inv = self.sigma**2 / D2
        D2inv = D2inv*mask # remove eroneous diagonal elements
        return 0.5*self.epsilon*np.sum(D2inv**6, axis=(1, 2))
    def box_energy_np(self, x):
        """Harmonic boundary condition energy. Based on Noe's implementation."""
        xloc = x[:, 0::2]; yloc = x[:, 1::2]
        E = 0
        d_left = -(xloc + self.l_box)
        E += np.sum((np.sign(d_left) + 1) * self.k_box * d_left**2, axis=1)
        d_right = (xloc - self.l_box)
        E += np.sum((np.sign(d_right) + 1) * self.k_box * d_right**2, axis=1)
        d_down = -(yloc + self.l_box)
        E += np.sum((np.sign(d_down) + 1) * self.k_box * d_down**2, axis=1)
        d_up = (yloc - self.l_box)
        E += np.sum((np.sign(d_up) + 1) * self.k_box * d_up**2, axis=1)
        return E
    def harmonic_constraint(self, x):
        """Applies a harmonic restraint."""
        d2 = (x - self.reference)**2
        return np.sum(self.k_restrain * (self.sigma**2 * d2) ** 6, axis=1)
    def energy(self, x):
        """Returns the energy (numpy implementation)"""
        return self.bond_energy_np(x) + self.LJ_energy_np(x) + \
                self.box_energy_np(x) + self.harmonic_constraint(x)
    
    def initial_positions(self, ddist):
        """Inititalized a simulations with the dimer at ddist. copied straight
        from Noe."""
        # dimer
        pos = []
        pos.append(np.array([-0.5*ddist, 0]))
        pos.append(np.array([0.5*ddist, 0]))
        # solvent particles
        sqrtn = int(np.sqrt(self.nsolvent))
        locs = np.linspace(-self.l_box-1, self.l_box+1, sqrtn+2)[1:-1]
        for i in range(0, sqrtn):
            for j in range(0, sqrtn):
                pos.append(np.array([locs[i], locs[j]]))
        pos = np.array(pos).reshape((1, 2*(self.nsolvent+2)))
        return pos
        
    def draw_config(self, x, dimercolor='blue', alpha=0.7):
        """Draws the cinfiguration. copied straight from Noe."""
        # prepare data
        X = x.reshape(((self.nsolvent+2), 2))
        # set up figure
        plt.figure(figsize=(5, 5)); axis = plt.gca()
        d = self.l_box; axis.set_xlim((-d, d)); axis.set_ylim((-d, d))
        # draw box
        axis.add_patch(Rectangle((-d-self.sigma, -d-self.sigma),
            2*d+2*self.sigma, 0.5*self.sigma, color='lightgrey', linewidth=0))
        axis.add_patch(Rectangle((-d-self.sigma, d+0.5*self.sigma),
            2*d+2*self.sigma, 0.5*self.sigma, color='lightgrey', linewidth=0))
        axis.add_patch(Rectangle((-d-self.sigma, -d-self.sigma),
            0.5*self.sigma, 2*d+2*self.sigma, color='lightgrey', linewidth=0))
        axis.add_patch(Rectangle((d+0.5*self.sigma, -d-self.sigma),
            0.5*self.sigma, 2*d+2*self.sigma, color='lightgrey', linewidth=0))
        # draw solvent
        circles = []
        for x in X[2:]:
            circles.append(axis.add_patch(Circle(x, radius=0.5*self.sigma,
                linewidth=2, edgecolor='black', facecolor='grey', alpha=alpha)))
        # draw dimer
        circles.append(axis.add_patch(Circle(X[0], radius=0.5*self.sigma,
            linewidth=2, edgecolor='black', facecolor=dimercolor, alpha=alpha)))
        circles.append(axis.add_patch(Circle(X[1], radius=0.5*self.sigma,
            linewidth=2, edgecolor='black', facecolor=dimercolor, alpha=alpha)))
        return circles
        
    def hungarian_permute(self, X):
        """Applies the Hungarian algorithm to permute particle indices to
        minimize the distance to the reference state. Mostly copied from Noe's
        code, except I did make a few adaptations"""
        xref = np.expand_dims(self.reference, axis=0)
        indices = np.arange(4, self.dim)
        xloc = X[:, 4::2]; yloc = X[:, 5::2]
        x0 = (xref*np.ones(X.shape))[:,4::2]
        y0 = (xref*np.ones(X.shape))[:,5::2]
        x0 = np.tile(np.expand_dims(x0, 2), [1, 1, np.shape(xloc)[1]])
        y0 = np.tile(np.expand_dims(y0, 2), [1, 1, np.shape(xloc)[1]])
        xloc = np.tile(np.expand_dims(xloc, 2), [1, 1, np.shape(xloc)[1]])
        yloc = np.tile(np.expand_dims(yloc, 2), [1, 1, np.shape(yloc)[1]])
        xdist = xloc - np.transpose(x0, [0,2,1])
        ydist = yloc - np.transpose(y0, [0,2,1])
        D = np.sqrt(xdist**2 + ydist**2)
        Y = X.copy()
        for i in range(D.shape[0]):
            _, columns = linear_sum_assignment(D[i])
            assignments = [2*columns+i for i in range(2)]
            assignments = np.vstack(assignments).T.flatten()
            Y[i, indices] = X[i, indices[assignments]]
        return Y
            
    def MCsample(self, nsamples, nsteps, nfirststeps=1000):
        """Procedure for generating training samples"""
        # Get a new reference configuration
        X = MetropolisSample(self, self.reference, 1, nfirststeps)
        self.reference = X[0]
        # Sample from the new reference
        X = MetropolisSample(self, self.reference, nsamples, nsteps)
        return self.hungarian_permute(X)

if __name__ == '__main__':
    pw = ParticleDimer()
    X = pw.MCsample(100,1000)
