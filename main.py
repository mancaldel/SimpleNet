import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import skimage.io as io

import tensorflow as tf
from tensorflow.python.framework import ops

import time
# import sys, os

import simplenet


#   #########################################################################


#   START THE CODE    #######################################################
print('  --- START ---  ')


# Load data
X_train = np.load('./data/X_train.npy')
Y_train = np.load('./data/Y_train.npy')

X_dev = np.load('./data/X_dev.npy')
Y_dev = np.load('./data/Y_dev.npy')

print('%i examples loaded in the training set' % X_train.shape[0])
print('    X_train: ', X_train.shape)
print('    Y_train: ', Y_train.shape)

print('%i examples loaded in the training set' % X_dev.shape[0])
print('    X_dev: ', X_dev.shape)
print('    Y_dev: ', Y_dev.shape)

# Train the system
print('\nStarting the training')
print('-----------------------')
clk_train_start = time.time()
_, _, parameters = simplenet.simple_model(X_train, Y_train, X_dev, Y_dev,
                                          learning_rate=0.0009, num_epochs=5)#,
                                          # restore='./tmp/model.ckpt')
print(time.time() - clk_train_start, "seconds elapsed")

#   #########################################################################


# prediction = simplenet.predict(X_dev, parameters)
# print(prediction)
