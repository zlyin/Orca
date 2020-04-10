# import packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate    # concatenate layers
from keras.regularizers import l2
from keras import backend as K


"""
Define Deeper GoogLeNet
"""
# define MiniGooLeNet class
class DeeperGoogLeNet:

	@staticmethod
	def conv_module(x, K, kx, ky, strides, chanDim, padding="same", reg=5e-4, name=None):

            # initialize layer names
            convName, bnName, actName = (name, name, name)

            # append layer name to module name
            if name is not None:
                convName += "_conv"
                bnName += "_bn"
                actName += "_act"

	    # CONV => BN => RELU
            x = Conv2D(K, (kx, ky), strides=strides, padding=padding,
                    kernel_regularizer=l2(reg), name=convName)(x)
            #x = BatchNormalization(axis=chanDim, name=bnName)(x)
            x = Activation("relu", name=actName)(x)
            x = BatchNormalization(axis=chanDim, name=bnName)(x)

            # return the block
            return x

	@staticmethod
	def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce,
            num5x5, num1x1Proj, chanDim, stagename, reg=5e-4):

            # define the first branch
            first = DeeperGoogLeNet.conv_module(x, num1x1, 1, 1,
                    (1, 1), chanDim, reg=reg, name=stagename + "_first")

            # define the second branch
            second = DeeperGoogLeNet.conv_module(x, num3x3Reduce, 1, 1,
                    (1, 1), chanDim, reg=reg, name=stagename + "_second1")
            second = DeeperGoogLeNet.conv_module(second, num3x3, 3, 3,
                    (1, 1), chanDim, reg=reg, name=stagename + "_second2")

            # define the third branch
            third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, 1, 1,
                    (1, 1), chanDim, reg=reg, name=stagename + "_third1")
            third = DeeperGoogLeNet.conv_module(third, num5x5, 5, 5,
                    (1, 1), chanDim, reg=reg, name=stagename + "_third2")

            # define the fourth branch
            fourth = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                    padding="same", name=stagename + "_pool")(x)
            fourth = DeeperGoogLeNet.conv_module(fourth, num1x1Proj, 1, 1,
                    (1, 1), chanDim, reg=reg, name=stagename + "_fourth")

            # concatenate all fms in channel
            x = concatenate([first, second, third, fourth], axis=chanDim,
                    name=stagename + "_concat")

            # return the block
            return x

	@staticmethod
	def build(width, height, depth, classes, reg):
            # initialize the input shape to be "channels last" and the
            # channels dimension itself
            inputShape = (height, width, depth)
            chanDim = -1

            # if we are using "channels first", update the input shape
            # and channels dimension
            if K.image_data_format() == "channels_first":
                    inputShape = (depth, height, width)
                    chanDim = 1

            # define the model input shape and a sequence of conv_modules
            inputs = Input(shape=inputShape)
            x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), chanDim, reg=reg, name="block1")
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool1")(x)

            x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chanDim, reg=reg, name="block2")
            x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), chanDim, reg=reg, name="block3")
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool2")(x)

            # define two inception modules followed by max POOL 
            x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim, "3a", reg=reg)
            x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim, "3b", reg=reg)
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool3")(x)

            # define five inception modules followed by max POOL 
            x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chanDim, "4a", reg=reg)
            x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, chanDim, "4b", reg=reg)
            x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, chanDim, "4c", reg=reg)
            x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, chanDim, "4d", reg=reg)
            x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim, "4e", reg=reg)
            x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool4")(x)

            # apply avgPool followed by dropout
            x = AveragePooling2D((4, 4), name="pool5")(x)
            x = Dropout(0.4, name="dropout")(x)

            # softmax classifier
            x = Flatten(name="flatten")(x)
            x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
            x = Activation("softmax", name="softmax")(x)

            # create the Model, use inputs as inputs, 
            model = Model(inputs=inputs, outputs=x, name="googlenet")

            # return the constructed network architecture
            return model
