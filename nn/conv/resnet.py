# import packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D 

from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras.layers import ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate, add 
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


"""
Define ResNet 50/101/152
"""
# define MiniGooLeNet class
class ResNet:

    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=1e-4,
            bnEps=2e-5, bnMom=0.9):
        # shortcut
        shortcut = data

        # 1st block of Residual module = Conv1x1, kernel # = 1/4 of output_dim
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K / 4), kernel_size=(1, 1), use_bias=False,
                kernel_regularizer=l2(reg))(act1)

        # 2nd block of Residual module = Conv3x3, kernel # = 1/4 of output_dim
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K / 4), kernel_size=(3, 3), strides=stride, padding="same", 
                use_bias=False, kernel_regularizer=l2(reg))(act2)

        # 3rd block of Residual module = Conv3x3, kernel # = output_dim
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, kernel_size=(1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if we need to reduce size on the shortcut branch
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False,
                    kernel_regularizer=l2(reg))(act1)
        
        # add shortcut with conv output
        x = add([shortcut, conv3])
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=1e-4,
            bnEps=2e-5, bnMom=0.9, dataset=None):
        # adjust input shape as backend setting
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # create Input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        # add different layers w.r.t different dataset
        if dataset == "cifar":
            x = Conv2D(filters[0], kernel_size=(3, 3), use_bias=False,
            padding="same", kernel_regularizer=l2(reg))(x) 
        elif dataset == "tiny-imagenet":
            # Conv5x5 => Act => BN => MaxPool
            x = Conv2D(filters[0], kernel_size=(5, 5), use_bias=False,\
                    padding="same", kernel_regularizer=l2(reg))(x)
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides=2)(x)
        else:
            # Conv5x5 (kernel=64) => Act => BN => MaxPool
            x = Conv2D(filters[0], kernel_size=(7, 7), use_bias=False, \
                    padding="same", kernel_regularizer=l2(reg))(x)  # not reduce fm
            x = Activation("relu")(x)
            x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
            x = MaxPooling2D((3, 3), strides=2)(x)
    
        # loop over stages to add
        for i in range(0, len(stages)):
            # initialize strides for each layer in the stage
            if i == 0:
                stage_strides = [1] * stages[i]
            else:
                stage_strides = [2] + [1] * (stages[i] - 1)
            stage_reds = [True] + [False] * (stages[i] - 1)
            
            # loop over layers in the stage
            for j in range(stages[i]):
                x = ResNet.residual_module(x, filters[i + 1], stage_strides[j], \
                        chanDim, red=stage_reds[j], reg=reg, bnEps=bnEps, bnMom=bnMom)
            pass
       
        # add BN => Act => AveragePool
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x) 
        x = Activation("relu")(x)
        #print("before avgpooling, x.shape = ", x.shape)
        x = GlobalAveragePooling2D()(x) 
        # replace AveragePooling2D to become adaptive, remove Flatten as well

        # FC & softmax
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
      
        # return the built model
        model = Model(inputs=[inputs], outputs=[x], name="resnet")
        return model


"""
- define different version of ResNet
"""
def ResNet50(width, height, depth, classes, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset=None):
    return ResNet.build(width, height, depth, classes, [3, 4, 6, 3], \
            [64, 256, 512, 1024, 2048], reg=reg, bnEps=bnEps, bnMom=bnMom, \
            dataset=dataset)

def ResNet101(width, height, depth, classes, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset=None):
    return ResNet.build(width, height, depth, classes, [3, 4, 23, 3], \
            [64, 256, 512, 1024, 2048], reg=reg, bnEps=bnEps, bnMom=bnMom, \
            dataset=dataset)

def ResNet152(width, height, depth, classes, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset=None):
    return ResNet.build(width, height, depth, classes, [3, 4, 36, 3], \
            [64, 256, 512, 1024, 2048], reg=reg, bnEps=bnEps, bnMom=bnMom, \
            dataset=dataset)

