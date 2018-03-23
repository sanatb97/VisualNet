# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:55:06 2017

@author: sanat
"""
from __future__ import division, print_function, absolute_import


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_hdf5_image_dataset
import tflearn.datasets.oxflower17 as oxflower17
import h5py
import math
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
"""train_dataset_file = '/home/sanat/Desktop/CDSAML/imagepaths(train).txt' 
val_dataset_file = '/home/sanat/Desktop/CDSAML/imagepaths(val).txt'
build_hdf5_image_dataset(train_dataset_file, image_shape=(128, 128), mode='file', output_path='dataset.h5', categorical_labels=True, normalize=True)

h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y'] 

X_val, Y_val = image_preloader(val_dataset_file, image_shape=(227, 227),
                       mode='file', categorical_labels=True,
                       normalize=True)"""

# Building 'Alexnet 2.0'

network = input_data(shape=[None, 227,227, 3])
"""layer 1:conv+max+norm"""
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
"""layer 2:conv+max+norm"""
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
"""layer 3:conv+max+norm"""
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)

"""layer 4:conv"""
network = conv_2d(network, 384, 3, activation='relu')
"""layer 5:conv"""
network = conv_2d(network, 384, 3, activation='relu')
"""layer 6:conv"""
network = conv_2d(network, 384, 3, activation='relu')


"""layer 7:fully_connected+dropout"""
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
"""layer 8:fc+dropout"""
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
"""layer 9:fc+dropout"""
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
"""layer 10:fc"""
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',learning_rate=0.001)
                     
"""network = fully_connected(network, 9, activation='softmax')             #change the output from 17 to 9 classes
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)                    
"""
# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=300, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='VisualNet')

