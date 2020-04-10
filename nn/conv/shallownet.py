## import packages
import os
import sys

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as Keras_backend


## define class & build NN
class ShallowNet:

    # define a static method, build, to build NN
    @staticmethod
    def build(height, width, depth, classes):
        # initiate a layers Sequential() & image shape
        model = Sequential()
        
        inputShape = (height, width, depth)
        if Keras_backend.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define NN
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # flatten before send to FC layer
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model




