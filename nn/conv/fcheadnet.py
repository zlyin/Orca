## import packages
import os 
import sys
from keras.layers.core import Flatten, Dropout, Dense

## define FCHeadNet class
class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, N):
        """
        initialize a head model to be placed on top of baseModel;
        baseModel = backbone of CNN;
        classes = # of classes;
        N = # of output neurons;
        """
        # get output from backbone
        headModel = baseModel.output
        # flatten output volume for FC
        headModel = Flatten(name="flatten")(headModel)
        # add FC with ReLu activation
        headModel = Dense(N, activation="relu")(headModel)
        # add Dropout 
        headModel = Dropout(0.5)(headModel)
        # add Dense with softmax
        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel



