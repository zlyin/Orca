## import packages
import os
import sys
import cv2
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d


## define patch preprocess class
class PatchPreprocessor:
    
    def __init__(self, width, height):
        """
        - return patches in shape of (height, width)
        """
        self.width = width
        self.height = height
        pass

    def preprocess(self, image):
        """
        - extract random crop from the input image & return crops in the target
          shape
        - to be applied when raw images dimensions are larger than input shape
          of NN during TRAINING process, say 256 > 227 (AlexNet)
        - equivalent to another layer of data augmentation, which is reasonable
          considered a method of killing overfitting/regularization
        """
        return extract_patches_2d(image, (self.height, self.width), \
                max_patches=10)[0]
