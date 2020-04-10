## import package
import os
import sys
import imutils
import cv2
import numpy as np


## define AspectAwarePreprocessor class
class AspectAwarePreprocessor:

    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # initiate vars with target image dimensions
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        h, w = image.shape[:2]
        dH, dW = 0, 0
        
        """
        2 steps to maintain AR:
        resize along shorter dimension & crop along longer dimension;
        call cv2.resize(image, (target_W, tatget_H)) to ensure dimension
        """
        if h < w:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)
        else:
            # h > w
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        
        # use dH & dW to crop the central area of Resized image
        newH, newW = image.shape[:2]
        newImage = image[dH : newH - dH, dW : newW - dW]

        # resize once more to ensure output dimension
        newImage = cv2.resize(newImage, (self.width, self.height), \
                interpolation=self.inter)

        return newImage
    
