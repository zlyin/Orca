## import packages
import cv2


## define a simple preprocessor class
class SimplePreprocessor:

    def __init__(self, width, height, interpolate=cv2.INTER_AREA):
        """
        - define targeted width & height of output image & interpolation method
        """
        # initiate
        self.width = width
        self.height = height
        self.interpolate = interpolate


    def preprocess(self, image):
        """
        - simply resize the image to a targeted size
        - IGNORE the original aspect ratio
        """
        return cv2.resize(image, (self.width, self.height), \
                interpolation=self.interpolate)

