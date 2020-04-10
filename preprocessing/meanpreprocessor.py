## import packages
import os
import sys
import cv2
import numpy as np


## define mean substraction processor class
class MeanPreprocessor:

    def __init__(self, rMean, gMean, bMean):
        """
        - input = means of r, g, b channels stored in a json file while loading
          training set
        """
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        pass

    def preprocess(self, image):
        """
        - preform mean substraction to an input image
        - split image into R/G/B channels, substrating corresponding mean value
          & return merged image(in BGR channel order)
        - a type of image normalization which normalize pixels around zero mean
        """
        # split image into 3 slices
        (B, G, R) = cv2.split(image.astype("float32"))

        B -= self.bMean
        G -= self.gMean
        R -= self.rMean

        # return merged image
        return cv2.merge([B, G, R])





