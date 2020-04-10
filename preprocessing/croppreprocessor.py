## import packages
import os
import sys
import cv2
import numpy as np


## define a crop preprocess class
class CropPreprocessor:

    def __init__(self, width, height, hflip=True, interpolate=cv2.INTER_AREA):
        """
        - define targeted dimension of image crops, whether to apply horizontal flip &
          interpolation method
        """
        self.width = width
        self.height = height
        self.hflip = hflip
        self.interpolate = interpolate
        pass

    def preprocess(self, image):
        """
        - extract image crops in the defined shape at 4 corners & center of 
        the input image. 
        - augment to 10 images by applying horizontal flip if required
        - a TTA/over-sampling method to be applied during EVALUATING preprocess
        - generally improve cls accuracy by 1~2%
        """
        # get necessary dimensions & coords
        H, W = image.shape[:2]
        crops = []
        # (startX, startY, endX, endY) of 4 corners
        coords = [
                [0, 0, self.width, self.height],
                [W - self.width, 0, W, self.height],
                [W - self.width, H - self.height, W, H], 
                [0, H - self.height, self.width, H],
                ]
        # center coord
        dH = int(0.5 * (H - self.height))
        dW = int(0.5 * (W - self.width))
        coords.append([dW, dH, W - dW, H - dH])

        # create 5 crops
        for startX, startY, endX, endY in coords:
            crop = image[startY : endY, startX : endX, :] if len(image.shape) \
                    == 3 else image[startY : endY, startX : endX]
            # MUST resize to ensure dimension are exactly same!
            crop = cv2.resize(crop, (self.width, self.height))
            crops.append(crop)
        
        # apply horizontal flip if required
        if self.hflip:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        # return 10 crops
        return np.array(crops)


        
