## import packages
import os
import sys

import cv2
import random
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.core import Lambda
from keras.engine.input_layer import Input
from keras import backend as K

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_addons as tfa

# enable_eager_execution so that the following tf functions can work
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True

## define method for layers
def rotate(images, rotation_range, scale_range):
    
    # consider K.image_data_format
    Cindex = 3
    if K.image_data_format == "channels_first":
        # rearrange to "channel_last" so that following tf functions can work
        images = tfk.preprocessing.image.img_to_array(images, \
                data_format="channels_last")
    B, H, W, C = images.shape
    
    angle = random.randint(rotation_range[0], rotation_range[1])
    scale = random.uniform(scale_range[0], scale_range[1])

    # mirror/flip image
    images = tf.image.flip_left_right(images) 
    
    # scale 
    newH, newW = int(H * scale), int(W * scale)
    images = tf.image.resize(images, (newH, newW), preserve_aspect_ratio=True)

    # rotate with a random angle & move center
    # convert tensor to numpy at first!
    # rotateImages = images.numpy() 
    rotateImages = tfa.image.rotate(images, angle)

    #print("before rotation", images.shape)

    # crop to image to 42x42
    images = tf.image.resize_with_crop_or_pad(rotateImages, 42, 42) 

    # convert EagerTensor to numpy array
    #return images.numpy()
    #return images
    return images


def rotata_outputShape(input_shape):
    if K.image_data_format == "channel_first":
        return (inputShape[0], 42, 42)
    ## bug is here!
    return (42, 42, input_shape[-1])


## define class
class KFER_LeNet:
    @staticmethod
    def build(height, width, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # change if needed
        if K.image_data_format == "channel_first":
            inputShape = (depth, height, width)

        #tf.compat.v1.enable_eager_execution()
        #print("in lenet1", tf.executing_eagerly())
        """
        Building Rotation & Scale layer;
        Noting that the first layer must define inputShape!
        """
        #x = Input(shape=inputShape)
        model.add(Lambda(rotate, input_shape=inputShape, \
                output_shape=rotata_outputShape, \
                arguments={"rotation_range" : [-45, 45], \
                "scale_range" : [0.8, 1.2]}))
        # outputShape=(42, 42, 3)
        
        """
        Building Conv(32, 5x5) => ReLu => MaxPool => (21, 21, 32)
        """
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        """
        Conv(32, 4x4) => ReLu => AvgPool => (10, 10, 32)
        """
        model.add(Conv2D(32, (4, 4), padding="same"))
        model.add(Activation("relu"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        """
        Conv(64, 5x5) => ReLu => AvgPool => (5, 5, 64)
        """
        model.add(Conv2D(64, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        """
        Flatten => FC(3072) => ReLu => FC(7) => softmax
        """
        model.add(Flatten())
        model.add(Dense(3072))
        model.add(Activation("relu"))
    
        model.add(Dense(classes))
        model.add(Activation("softmax"))
    
        return model
        







