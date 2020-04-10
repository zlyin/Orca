## import packages
from keras.models import Sequential, Model

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten

from keras.regularizers import l2
from keras import backend as K


## define AlexNet class
class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # define inputShape according to "channel_last" or "channel_first"
        inputShape = (height, width, depth)     # (227, 227, 3)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = 1     # will add an axis of batch!

        # initiate a Sequential model
        model = Sequential()
        
        # 1st block == Conv => Act => BN => Maxpool => Dropout
        model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=inputShape, \
                kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))
 
        # 2nd block == Conv => Act => BN => Maxpool => Dropout
        model.add(Conv2D(256, (5, 5), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd block == Conv => Act => BN 
        model.add(Conv2D(384, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # 4th block == Conv => Act => BN 
        model.add(Conv2D(384, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # 5th block == Conv => Act => BN => Maxpool => Dropout
        model.add(Conv2D(256, (3, 3), padding="same",
            kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # 6th block == FC => Act => BN => Dropout
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # 7th block == FC => Act => BN => Dropout
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # 8th block == FC => Softmax
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        # return NN
        return model

