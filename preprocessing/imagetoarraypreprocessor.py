## import packages
import os
import sys

from keras.preprocessing.image import img_to_array


## define class
class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat
        pass

    def preprocess(self, image):
        # Use Keras utility function rearrange the order of (H, W, Channesl)
        return img_to_array(image, data_format=self.dataFormat)








