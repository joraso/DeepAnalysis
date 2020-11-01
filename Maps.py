# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:58:57 2020
Contains 'InvMap' objects and functions

InvMap - The invertible map object

@author: Joe Raso
"""

import sys
import keras
import tensorflow as tf
import numpy as np

class InvMap(object):
    def __init__(self, blocks, model, prior, regulator=None, optimizer=None,
                 params={}):
        """The invertible map object.
        
        Inputs
        ----------
        blocks : list of InvBlock objects
            list of invertible layers, listed from x -> z.
        model : model object
            energy model of the real (x) domain.
        prior: prior object
            energy model of the latent (z) domain.
        regulator : function
            energy regulation function. Default is identity.
        optimizer : keras.optimizers.Optimizer object
            optimizing algorithm to be used in training. Default is to Adam.
        params : dictionary
            dictionary containing the other numerical parameters related to the
            map and it's training.
        """
        # Set component objects
        self.blocks = blocks
        self.model = model
        self.prior = prior
        # Set regulator to identity if none is provided.
        if regulator == None:
            def identity(x):
                return x
            regulator = identity
        self.regulator = regulator
        # Set optimizer to adam if none is provided
        if optimizer == None:
            optimizer = keras.optimizers.adam(lr=0.001)
        self.optimizer = optimizer
        # Add other parameters
        default_params = {'temperature':1.0, 'explore':1.0, 'batch_size':512}
        self.__dict__.update(default_params)
        self.__dict__.update(params)
        # Other associated (not user provided) containers
        self.KLhistory = []; self.MLhistory = []
    
    # Connectors ==========================================================
    def connect_forward(self, x):
        """Creates forward (x -> z) connections."""
        for i in range(len(self.blocks)):
            self.blocks[i].connect_xz(x)
            x = self.blocks[i].z_out
        return x
    def connect_backwards(self, z):
        """Creates backward (z -> x) connections."""
        for i in range(len(self.blocks)-1, -1, -1):
            self.blocks[i].connect_zx(z)
            z = self.blocks[i].x_out
        return z
    def connect_LogDetJ(self):
        """Compiles the net LogDetJ for the forward and reverse map."""
        LogDetJxzs = []; LogDetJzxs = []
        for i in range(len(self.blocks)):
            LogDetJxzs.append(self.blocks[i].LogDetJxz)
            LogDetJzxs.append(self.blocks[i].LogDetJzx)
        # create tensors for the loss functions to use
        self.Rxz = tf.reduce_sum(LogDetJxzs, axis=0, keepdims=False)
        self.Rzx = tf.reduce_sum(LogDetJzxs, axis=0, keepdims=False)
        # create layers for evaluating
        self.Rxz_layer = keras.layers.Add()(LogDetJxzs)
        self.Rzx_layer = keras.layers.Add()(LogDetJzxs)
    def connect_all(self):
        """Connects and builds the forward and backwards maps into a pair of
        keras models """
        # connect and compile layers
        self.x_in = keras.layers.Input(shape=(self.model.dim,))
        self.z_out = self.connect_forward(self.x_in)
        self.z_in = keras.layers.Input(shape=(self.prior.dim,))
        self.x_out = self.connect_backwards(self.z_in)
        # connect to the Jacobian
        self.connect_LogDetJ()
        # build networks
        self.Txz = keras.models.Model(inputs=self.x_in, outputs=self.z_out)
        self.Tzx = keras.models.Model(inputs=self.z_in, outputs=self.x_out)
        # Have to build this in order to evaluate the jacobian after training
        self.LogDetJxz = keras.models.Model(inputs=self.x_in,
            outputs=self.Rxz_layer)
        self.LogDetJzx = keras.models.Model(inputs=self.z_in,
            outputs=self.Rzx_layer)

    # KL Training =========================================================
    def KLDivergence(self):
        """Creates a loss function that computes the KL divergence for
        KL, or by-energy, training."""
        def loss(y_true, y_pred):
            # Here is where we might need a type conversion
            Ereg = self.regulator(self.model.energy(self.x_out) /
                self.temperature)
            return Ereg - self.explore * self.Rzx
        return loss
    def KLCompile(self):
        """Compiles the backward map in preperation for KL training."""
        KLloss = self.KLDivergence()
        self.Tzx.compile(self.optimizer, loss=KLloss)
    def KLStep(self):
        """Takes a single training step in the KL regime."""
        z = self.prior.sample(self.batch_size)
        dummy = np.zeros((self.batch_size, self.model.dim))
        batch_loss = self.Tzx.train_on_batch(z, dummy)
        self.KLhistory.append(batch_loss)
    def KLTrain(self, epochs, verbose=True):
        """Performs KL training for 'epochs'."""
        for e in range(epochs):
            self.KLStep()
            if verbose:
                print("Epoch: {}, KL Loss: {}".format(e, self.KLhistory[-1]))
                sys.stdout.flush()
        
    # ML Training =========================================================
    def MLDivergence(self):
        """Creates a loss function that computes the maximum likeslyhood loss
        function for ML, or by-example training."""
        def loss(y_true, y_pred):
            Ereg = self.regulator(self.prior.energy(self.z_out))
            return Ereg - self.Rxz
        return loss
    def MLCompile(self):
        """Compiles the backward map in preperation for ML training."""
        MLloss = self.MLDivergence()
        self.Txz.compile(self.optimizer, loss=MLloss)
    def MLStep(self, training_set):
        """Takes a single training step in the ML regime."""
        sample = np.random.choice(
            np.arange(training_set.shape[0]), self.batch_size)
        x = training_set[sample,:]
        dummy = np.zeros((self.batch_size, self.prior.dim))
        batch_loss = self.Txz.train_on_batch(x, dummy)
        self.MLhistory.append(batch_loss)
    def MLTrain(self, training_set, epochs, verbose=True):
        """Performs ML training for 'epochs'."""
        for e in range(epochs):
            self.MLStep(training_set)
            if verbose:
                print("Epoch: {}, ML Loss: {}".format(e, self.MLhistory[-1]))
                sys.stdout.flush()
                
    # Sampling ============================================================
    def Sample(self, n):
        """Returns a set of n real space samples, along with their statistical
        weights and energies"""
        sess = tf.Session()
        z = self.prior.sample(n)
        Enz = self.prior.energy(z).eval(session=sess)
        x = self.Tzx.predict(z)
        Enx = self.model.energy(x)
        logdetJ = self.LogDetJzx.predict(z)
        logw = -Enx + Enz + logdetJ[:,0] # Have to slice it down
        return x, Enx, np.exp(logw)    
