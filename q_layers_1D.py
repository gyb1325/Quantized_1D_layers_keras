#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 21:36:52 2018

@author: guyo
"""

# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K

from keras.layers import InputSpec, Layer, Dense, Conv2D, Conv1D
from keras import constraints
from keras import initializers

from layers.quantized_ops import quantize, clip_through
from layers import conv_utils


class Clip(constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        #todo: switch for clip through?
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"name": self.__call__.__name__,
                "min_value": self.min_value,
                "max_value": self.max_value}
        
class QuantizedConv1D(Conv1D):
    def __init__(self, filters,kernel_regularizer=None,activity_regularizer=None,
         H=1., nb=16,  **kwargs):
        super(QuantizedConv1D, self).__init__(filters, **kwargs)
        self.H = H
        self.nb = nb
        self.activity_regularizer =activity_regularizer
        self.kernel_regularizer = kernel_regularizer
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1 
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)    # To be resolved 


        self.kernel_constraint = Clip(-self.H, self.H)
#        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        # TO resolve why it does not work when I enable this initializer
#        self.bias_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=kernel_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.filters,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        else:
            self.bias = None   
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True 
    def call(self,inputs):
        quantized_kernel = quantize(self.kernel,nb= self.nb)
        outputs = K.conv1d(
            inputs,
            quantized_kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])       
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)
    def get_config(self):
        config = {'H': self.H
                  }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



QuantizedConvolution1D = QuantizedConv1D




class QuantizedDense(Dense):
    ''' Binarized Dense layer
    References: 
    "QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1" [http://arxiv.org/abs/1602.02830]
    '''
    def __init__(self, units, H=1., nb=16,  **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.H = H
        self.nb = nb
        #super(QuantizedDense, self).__init__(units, **kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot H: {}'.format(self.H))
#        if self.kernel_lr_multiplier == 'Glorot':
#            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (input_dim + self.units)))
            #print('Glorot learning rate multiplier: {}'.format(self.kernel_lr_multiplier))
            
        self.kernel_constraint = Clip(-self.H, self.H)
#        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='kernel',
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

        if self.use_bias:
#            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        else:
#            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):
        quantized_kernel = quantize(self.kernel, nb=self.nb)
        output = K.dot(inputs, quantized_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


        
        
    def get_config(self):
        config = {'H': self.H}
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




