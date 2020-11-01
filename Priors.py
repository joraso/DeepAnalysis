# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:02:21 2020
Contains sample 'Prior' objects

prior - archetypal object

Useful priors:
Normal

@author: Joe Raso
"""

import numpy as np
import tensorflow as tf

class prior(object):
    def __init__(self, dim, params):
        """The Archetypal prior object."""
        self.dim = dim
        self.__dict__.update(params)
    def energy(self, z):
        """every prior object must have an energy function that returns
        the reduced energy for a given configuration z"""
        return 0
    def sample(self, n):
        """every prior object must also have an sample function that returns
        n sample configurations"""
        return np.zeros((n, self.dim))
        
# Sample Priors ===============================================================
        
class Normal(prior):
    def __init__(self, dim, params = {'std':1.0, 'loc':0.0}):
        """A Gaussian prior with std=1, centered at the origin"""
        super().__init__(dim, params)
    def energy(self, z):
        return tf.reduce_sum((z - self.loc)**2, axis=1) / (2 * self.std**2)
    def sample(self, n):
        return np.random.normal(loc=self.loc, scale=self.std, size=(n,self.dim))
