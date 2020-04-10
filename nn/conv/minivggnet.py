## import packages

import os
import sys
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
import keras.backend as K


## define MiniVGGNet
class MiniVGGNet:
    @staticmethod
    def build(height, width, depth, classes):
        
        # initiate the model
        model = Sequential()
        
        # use channelDim to indicate index of channels
        channelDim = -1
        
        # image channel reordering
        if K.image_data_format == "channel_first":
            inputShape = (depth, height, width) # input volume = (batch, chan, H, W)
            channelDim = 1
        else:
            inputShape = (height, width, depth) # input volume = (batch, H, W, chan)

        """
        1st CONV => ReLu => CONV => ReLu => POOL block
        32 filters, kernel size = 3x3
        """
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        # between 2 CONV, add a dropout layer to kill overfitting
        model.add(Dropout(0.25))
       
        """
        2nd CONV => ReLu => CONV => ReLu => POOL block
        Increase to 64 filters, kernel size = 3x3
        """
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        # between Conv & FC, add a dropout layer to kill overfitting
        model.add(Dropout(0.25))

        """
        1st FC layer
        Increase to 512 outputs
        """
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        # between FC & FC, add a dropout layer to kill overfitting
        model.add(Dropout(0.5))
        
        """
        2nd FC layer connecting to softmax classifier
        Decrease back to # of classes outputs
        """
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        #return model
        return model

