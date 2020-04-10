## import packages

import os
import sys
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
import keras.backend as K


## define a mini RNN class
class MiniRNN:
    @staticmethod
    def build(height, width, depth, classes):
        
        
