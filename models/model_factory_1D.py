from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import Reshape, Activation, Dropout, Input, MaxPooling1D, BatchNormalization, Flatten, Dense, Lambda, concatenate, Conv1D, GlobalAveragePooling1D
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.regularizers import l2
import numpy as np

from layers.q_layers_1D import QuantizedConv1D,QuantizedDense
from layers.quantized_ops import quantized_relu as quantize_op
from layers.b_layer_1D import BinaryConv1D,BinaryDense
from layers.binary_ops import binary_tanh as binary_tanh_op
import sys

from keras.utils import plot_model

def build_model(WINDOW_SIZE):
    abits = 16
    wbits = 16
    kernel_lr_multiplier  = 10 

    def quantized_relu(x):
        return quantize_op(x,nb=abits)

    def binary_tanh(x):
        return binary_tanh_op(x)
#    network_type = 'float'
    #network_type ='qnn'
#    network_type = 'full-qnn'
    network_type ='bnn'
    # network_type = 'full-bnn'   
    H = 1.
    if network_type =='float':
        Conv_ = lambda f, s, c, n: Conv1D(kernel_size=s, filters=f, padding='same', activation='linear',
                                   input_shape = (c,1), name = n)
        Conv = lambda  f, s, n: Conv1D(kernel_size= s, filters=f,  padding='same', activation='linear', name = n)
        
        Dense_ = lambda f, n: Dense(units = f, kernel_initializer='normal', activation='relu', name = n)
        Act = lambda: ReLU()
    elif network_type=='qnn':
       # sys.exit(0)
        Conv_ = lambda f, s,  c, n: QuantizedConv1D(kernel_size= s, H=1, nb=wbits, filters=f, strides=1,
                                            padding='same', activation='linear',
                                            input_shape = (c,1),name = n)
        Conv = lambda f, s, n: QuantizedConv1D(kernel_size=s, H=1, nb=wbits, filters=f, strides= 1,
                                            padding='same', activation='linear',
                                            name = n)
        Act = lambda: ReLU()
        
        Dense_ = lambda f, n: QuantizedDense(units = f, nb = wbits, name = n)
        
    elif network_type=='full-qnn':
        #sys.exit(0)
        Conv_ = lambda f, s,  c, n: QuantizedConv1D(kernel_size= s, H=1, nb=wbits, filters=f, strides=1,
                                            padding='same', activation='linear',
                                            input_shape = (c,1),name = n)
        Conv = lambda f, s, n: QuantizedConv1D(kernel_size=s, H=1, nb=wbits, filters=f, strides= 1,
                                            padding='same', activation='linear',
                                            name = n)
        Act = lambda: Activation(quantized_relu)
        Dense_ = lambda f, n: QuantizedDense(units = f, nb = wbits, name = n)
    elif network_type=='bnn':
       # sys.exit(0)
        Conv_ = lambda f,s,c,n: BinaryConv1D(kernel_size= s, H=1, filters=f, strides=1, padding='same',
                                         activation='linear',
                                         input_shape = (c,1),
					 name = n)
        Conv = lambda f,s,n: BinaryConv1D(kernel_size=s, H=1, filters=f, strides=1, padding='same',
                                         activation='linear', 
                                         name = n )
        Dense_ = lambda f, n: BinaryDense(units = f, name = n)
        Act = lambda: ReLU()
    elif network_type=='full-bnn':
        #sys.exit(0)
        Conv_ = lambda f,s,c,n: BinaryConv1D(kernel_size= s, H=1, filters=f, strides=1, padding='same',
                                         activation='linear',
                                         input_shape = (c,1),
					 name = n)
        Conv = lambda f,s,n: BinaryConv1D(kernel_size=s, H=1, filters=f, strides=1, padding='same',
                                         activation='linear', 
                                         name = n    )
        Act = lambda: Activation(binary_tanh)
    else:
        #sys.exit(0)
        print('wrong network type, the supported network types in this repo are float, qnn, full-qnn, bnn and full-bnn') 


    model = Sequential()      
    OUTPUT_CLASS = 4    # output classes
    #model = Sequential()
    model.add(Conv_(64, 55, WINDOW_SIZE,  'conv1')  )  
    model.add(Act())
    model.add(MaxPooling1D(10))
    model.add(Dropout(0.5))
    model.add(Conv(64, 25,  'conv2' ))
    model.add(Act())
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.5))
    model.add(Conv(64, 10,  'conv3'))
    model.add(Act())
    model.add(GlobalAveragePooling1D())
    model.add(Dense_(256,  'den6'))
    model.add(Dropout(0.5))
    model.add(Dense_(128,  'den7'))
    model.add(Dropout(0.5))		
    model.add(Dense_(64, 'den8'))
    model.add(Dropout(0.5))	
    model.add(Dense(OUTPUT_CLASS, kernel_initializer='normal', activation='softmax', name = 'den9'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])        
        
        
    print(model.summary())
    plot_model(model, to_file='my_model.png')
    return model
