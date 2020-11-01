# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:44:56 2020
Contains 'layer' objects and functions
Invertable transformation layers.

Utilities:
connect() - utility function for connecting keras layers
AffineLayer() - Function for creating nonlinear 'hidden' or 'artificial'
                networks
Partition - object for partitioning indices.
IndexLayer - a keras layer to sort by indices.
                
Invertible Layers:
InvBlock - Parent object
RealNVP - invertible real-valued non-volume-preserving composite layer (block)

@author: Joe Raso
"""

import keras
import numpy as np
import tensorflow as tf

def connect(input_layer, layers):
    """From Noe: Connects a given sequence of layers and returns output layer

    Inputs
    ----------
    input_layer : keras layer
        Input layer
    layers : list of keras layers
        Layers to be connected sequentially

    Returns
    -------
    output_layer : kears layer
        Output Layer

    """
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer

def AffineLayer(output_size, init_outputs=None,
                nlayers=3, nhidden=100, activation='relu'):
    """From Noe: Generic dense trainable nonlinear transform. Returns the
    layers of a dense feedforward network with nlayers-1 hidden layers with
    nhidden neurons and the specified activation functions. The last layer is
    linear in order to access the full real number range and has output_size
    output neurons.

    Inputs
    ----------
    output_size : int
        number of output neurons
    nlayers : int
        number of layers
    nhidden : int
        number of neurons in each hidden layer
    activation : str
        nonlinear activation function in hidden layers
    init_outputs : None or float or array
    """
    M = [keras.layers.Dense(nhidden, activation=activation)
            for i in range(nlayers-1)]
    if init_outputs is None:
        final_layer = keras.layers.Dense(output_size, activation='linear')
    else:
        final_layer = keras.layers.Dense(output_size, activation='linear',
                    kernel_initializer=keras.initializers.Zeros(),
                    bias_initializer=keras.initializers.Constant(init_outputs))
    M += [final_layer]
    return M

class IndexLayer(keras.engine.Layer):
    def __init__(self, indices, **kwargs):
        """From Noe: Returns a keras layer that sorts the input by
         index, returning [:, indices]"""
        super().__init__(**kwargs)
        self.indices = indices
        self.trainable=False
    def call(self, x):
        return tf.gather(x, self.indices, axis=1)
    def compute_output_shape(self, input_shape):
        return input_shape[0], len(self.indices)

class Partition(object):
    def __init__(self, dim, split_indices=None):
        """Object that handles splitting and merging of data into channels.
        
        Inputs
        ----------
        dim : int
            dimentionality of the data to be split/merged
        split_indices : list of lists
            specifying the channels to be split. each member of the list will
            be a channel, containing the indices in that entry
        """
        self.dim = dim
        # Create index key for splitting if not provided.
        if split_indices==None:
            self.autogenerate()
        else:
            self.split_indices = split_indices
        self.channels = len(self.split_indices)
        # Create index key for merging.
        split_order = []; self.merge_indices = []
        for i in range(self.channels):
            split_order += self.split_indices[i]
        for i in range(self.dim):
            self.merge_indices.append(split_order.index(i))
    def autogenerate(self, nchannels=2):
        """randomly creates nchannels (not neccessarily of the same dimension)"""
        self.split_indices = []
        for i in range(nchannels):
            # Distribute at minimum one index into each channel
            self.split_indices.append([i])
        for j in range(i+1,self.dim):
            c = np.random.randint(0, high=nchannels)
            self.split_indices[c].append(j)
    def splitchannel(self):
        split = []
        for i in range(self.channels):
            split.append(IndexLayer(self.split_indices[i]))
        return split
    def mergechannel(self):
        return IndexLayer(self.merge_indices)

class InvBlock(object):
    def __init__(self, transforms, params):
        """The archetypal invertable block object. the inputs are a list of
        keras layers with the nonlinear transforms used in the layer.
        Must have the following properties and functions:"""
        self.transforms = transforms
        self.__dict__.update(params)
        self.LogDetJxz = 0; self.LogDetJzx = 0
        self.x_out = 0; self.x_in = 0; self.z_out = 0; self.z_in = 0;
    def connect_xz(self, x):
        pass # returns
    def connect_zx(self, z):
        pass

class RealNVP(InvBlock):
    def __init__(self, transforms, partition, params={}):
        """A real-valued non-volume preserving transformation. Input is a
        list of 4 affine layers (of type keras layer).
        
        Inputs
        ----------
        transforms : list of list of keras layers
            list of 4 affine layers (each an unconnected list of keras layers).
            [S1, T1, S2, T2]
        partition : partition object
            The partition object defining how the data is to be subdivided.
        params: dict.
            dictionary of parameters.
        """
        super().__init__(transforms, params)
        self.partition = partition
        
        self.S1 = self.transforms[0]
        self.T1 = self.transforms[1]
        self.S2 = self.transforms[2]
        self.T2 = self.transforms[3]
        
    def connect_xz(self, x):
        """Creates forward (x -> z) connection."""
        def lambda_exp(x):
            return keras.backend.exp(x)
        def lambda_sum(x):
            return keras.backend.sum(x[0], axis=1, keepdims=True) + \
                   keras.backend.sum(x[1], axis=1, keepdims=True)
        # partition the channels
        self.x_in = x
        [channel1, channel2] = self.partition.splitchannel()
        x1 = channel1(self.x_in); x2 = channel2(self.x_in)
        self.x1_in = x1; self.x2_in = x2
        # First sublayer x -> y
        y1 = x1
        self.Sxy = connect(x1, self.S1)
        self.Txy = connect(x1, self.T1)
        prodx = keras.layers.Multiply()(
            [x2, keras.layers.Lambda(lambda_exp)(self.Sxy)])
        y2 = keras.layers.Add()([prodx, self.Txy])
        # Second sublayer y -> z
        self.z2_out = y2
        self.Syz = connect(y2, self.S2)
        self.Tyz = connect(y2, self.T2)
        prody = keras.layers.Multiply()(
            [y1, keras.layers.Lambda(lambda_exp)(self.Syz)])
        self.z1_out = keras.layers.Add()([prody, self.Tyz])
        # recombine channels
        z_out = keras.layers.Concatenate()([self.z1_out, self.z2_out])
        self.z_out = self.partition.mergechannel()(z_out)
        # Jacobian - log det(dz/dx)
        self.LogDetJxz = keras.layers.Lambda(lambda_sum)([self.Sxy, self.Syz])
        
    def connect_zx(self, z):
        """Creates backward (z -> x) connection."""
        def lambda_negexp(x):
            return keras.backend.exp(-x)
        def lambda_negsum(x):
            return keras.backend.sum(-x[0], axis=1, keepdims=True) + \
                   keras.backend.sum(-x[1], axis=1, keepdims=True)
        # partition the channels
        self.z_in = z
        [channel1, channel2] = self.partition.splitchannel()
        z1 = channel1(self.z_in); z2 = channel2(self.z_in)  
        self.z1_in = z1; self.z2_in = z2
        # Inverse of second sublayer z -> y
        y2 = z2
        self.Szy = connect(z2, self.S2)
        self.Tzy = connect(z2, self.T2)
        diffz = keras.layers.Subtract()([z1, self.Tzy])
        y1 = keras.layers.Multiply()(
            [diffz, keras.layers.Lambda(lambda_negexp)(self.Szy)])
        # Inverse of second sublayer y -> x
        self.x1_out = y1
        self.Syx = connect(y1, self.S1)
        self.Tyx = connect(y1, self.T1)
        diffy = keras.layers.Subtract()([y2, self.Tyx])
        self.x2_out = keras.layers.Multiply()(
            [diffy, keras.layers.Lambda(lambda_negexp)(self.Syx)])
        # recombine channels
        x_out = keras.layers.Concatenate()([self.x1_out, self.x2_out])
        self.x_out = self.partition.mergechannel()(x_out)
        # Inverse Jacobian - log det(dx/dz)
        self.LogDetJzx = keras.layers.Lambda(lambda_negsum)([self.Szy, self.Syx])
