# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:41:10 2024

@author: DDZ
"""

# The code is written with tensorflow, multi-gpus

import tensorflow as tf
import h5py
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import  Bidirectional, Dense, Multiply, GRU, BatchNormalization, Add
import numpy as np
import pandas as pd 
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import  HeNormal

from tensorflow.keras.layers import  GlobalAveragePooling1D  
start_time = time.time() 


num_classes = 5 # The number of output classes
batch_size = 32 # The size of the mini-batch
epochs = 40 # The number of training epochs
learning_rate = 0.001 # The learning rate for the optimizer

out_1x1 = 32                  # Number of output channels for 1x1 Convolution Branch
red_3x3 = 48                  # Number of output channels for 1x1 Convolution in 3x3 Branch
out_3x3 = 64                  # Number of output channels for 3x3 Convolution Branch
red_5x5 = 64                   # Number of output channels for 1x1 Convolution in 5x5 Branch
out_5x5 = 96                   # Number of output channels for 5x5 Convolution Branch
red_7x7 = 16                   # Number of output channels for 1x1 Convolution in 5x5 Branch
out_7x7 = 48                   # Number of output channels for 5x5 Convolution Branch
out_pool = 32                  # Number of output channels for Pooling Branch


# save the model of best performance based on the accuracy 
filepath='.../model_name.h5'  
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')

# The shape is (num_samples, sequence_length, num_channels,  load the trainging data and label 
x_train = h5py.File('.../train_data.h5', 'r')
x_train = x_train.get('X') 
x_train = np.asarray(x_train, dtype=np.float16) # To avoid the case of "out of GPU ram" 
x_train = np.transpose(x_train, (0, 2, 1))
x_train = x_train[:, :, 3]
x_train = np.expand_dims(x_train, axis = 2)


y_train = pd.read_csv('.../train_label.csv')
y_train = np.array(y_train, dtype=np.float16)   

print('the shape of train samples', x_train.shape)
print('the shape of train label:', y_train.shape)


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)


# Define the number of the EEG channel input
n_channels = 1  # single channel 
input_shape = (x_train.shape[1], n_channels)


def se_block(input_tensor):
    num_channels = input_tensor.shape[-1]

    # Squeeze
    squeeze = GlobalAveragePooling1D()(input_tensor)
    
    # Excitation
    excitation = Dense(num_channels // 16, activation='relu')(squeeze)
    excitation = Dense(num_channels, activation='sigmoid')(excitation)
    
 
    output_tensor = Multiply()([input_tensor, excitation])
    
    return output_tensor



def FEMModule(x, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, red_7x7, out_7x7, out_pool):
    # 1x1 Convolution Branch
    branch1x1 = layers.Conv1D(out_1x1, 1, activation='relu')(x)

    # small size
    branch3x3 = layers.Conv1D(red_3x3, 1, activation='relu')(x)
    branch3x3 = layers.Conv1D(out_3x3, 3, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01))(branch3x3)
    branch3x1 = se_block(branch3x3)
    branch3x3 = Add()([branch3x3, branch3x1])

    # Middle size
    branch5x5 = layers.Conv1D(red_5x5, 1, activation='relu')(x)
    branch5x5 = layers.Conv1D(out_5x5, 16, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01))(branch5x5)
    branch5x1 = se_block(branch5x5)
    branch5x5 = Add()([branch5x5, branch5x1])
    
    # large size
    branch7x7 = layers.Conv1D(red_7x7, 1, activation='relu')(x)
    branch7x7 = layers.Conv1D(out_7x7, 64, padding='same', activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01))(branch7x7)
    branch7x1 = se_block(branch7x7)
    branch7x7 = Add()([branch7x7, branch7x1])
    
    # 1D Max Pooling followed by 1x1 Convolution Branch
    branch_pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
    branch_pool = layers.Conv1D(out_pool, 1, activation='relu')(branch_pool)

    # Concatenate the outputs along the channel dimension
    FEM_module = layers.concatenate([branch1x1, branch3x3, branch5x5, branch7x7, branch_pool], axis=-1)

    return FEM_module



strategy = tf.distribute.MultiWorkerMirroredStrategy()

print("The number of working GPUS:", strategy.extended.worker_devices)



with strategy.scope(): 
# Define the model architecture
    inputs = layers.Input(shape= input_shape)
    x = BatchNormalization()(inputs)
    x = FEMModule(x, n_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, red_7x7, out_7x7, out_pool)
    x = BatchNormalization()(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Bi-GRU
    x1 = Bidirectional(GRU(64, return_sequences=True))(x)
    x1 = layers.Conv1D(x.shape[-1], 1, activation='relu')(x1)
    x = Add()([x, x1])

    x = BatchNormalization()(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Global average pooling layer
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(num_classes)(x)
    x = layers.Softmax()(x)
    model = models.Model(inputs=inputs, outputs=x)
    
    # Compile the model with an optimizer, a loss function and a metric
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Print a summary of the model
# model.summary()
if __name__=='__main__':

    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', patience = 2,mode='auto',  verbose = 1, factor=0.5, min_lr = 0.0000001)
    
    # Train the model with 2 GPUs using MirroredStrategy
    
    model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_val, y_val),
                  callbacks=[learning_rate_reduction,checkpoint],
                  shuffle=True)

    
    end_time = time.time()    
    run_time = end_time - start_time   
    print('The time is:',run_time)
