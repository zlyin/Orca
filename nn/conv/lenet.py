## import packages
import os
import sys

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


## define class
class LeNet:
    @staticmethod
    def build(height, width, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        
        # change if needed
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)

        """
        Building 1st set of CONV => ReLu => Pool 
        20 outputs, 5x5 kernel, use same replicate padding method to maintain
        inputsize=outputsizel
        """
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        """
        Building 2nd set of CONV => ReLu => Pool 
        20 outputs, 5x5 kernel, use same replicate padding method to maintain
        inputsize=outputsizel
        """
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
 

        """
        Flatten input volume & build 2 FC layers
        500 outputs
        """
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        """
        Softmax classifier
        """
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model         
