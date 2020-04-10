
# import packages
import os
import sys
import cv2
import numpy as np

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None, mode=None):
        # take in a list of preprocessing ops
        if preprocessors is None:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors
        self.mode = mode
        pass

    def load(self, imagePaths, verbose=-1):
        # list to hold images & labels
        data, labels = [], []
        
        # loop over imagePaths
        for i, imagePath in enumerate(imagePaths):
            # load the image and extract its label, assuming our path is in the formate as:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            # image_name = "dataset/path/category/cat_00985.jpg"
            if self.mode == "test":
                # get test image name as labels
                label = imagePath.split(os.path.sep)[-1]
            else:
                label = imagePath.split(os.path.sep)[-2]

            # apply each of preprocessors to the image
            if self.preprocessors:
                for p in self.preprocessors:
                    image = p.preprocess(image)
                pass

            # treat processed image as a "feature vector"
            data.append(image)
            labels.append(label)
            
            # print out status
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed %d/%d" % (i + 1, len(imagePaths)))
            pass

        # return loaded images & labels
        return (np.array(data), np.array(labels))




    


