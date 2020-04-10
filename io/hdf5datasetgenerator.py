## import packages
import os
import sys
import numpy as np
import h5py
from keras.utils import np_utils


## define a HDF5DatasetGenerator class
class HDF5DatasetGenerator:

    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None,
            binarize=True, classes=2):
        """
        - dbPath = path to a HDF5 db
        - batchSize = batch size of data
        - preprocessors = [image preprocessors] to apply
        - aug = Keras ImageDataGenerator to apply data augmentation directly;
          defaulting to None
        - binarize = whether to binarize the stored class labels(ints) into
          one-hot encoding vectors; defaulting to True
        - classes = # of classes
        """
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["labels"].shape[0]

        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        pass

    def generator(self, passes=np.inf):
        """
        - passes = maximum number of epochs
        - return batches of images & labels until training epochs reaches passes
          limit
        """
        # initialize epoch counter
        epoch = 0
        
        # keep looping until it has reached the desired epochs number
        while epoch < passes:
            # loop over batches
            for i in range(0, self.numImages, self.batchSize):
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]
                
                # check if need to binarize to one-hot encoding
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                # check if need to apply preprocessers to image batch
                if self.preprocessors:
                    procImages = []
                    # loop over images & processors to process 
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    # save preprocessed images as new images 
                    images = np.array(procImages)
                
                # check if need to apply data augmentation
                if self.aug:
                    # "next()" return the next object in a generator
                    images, labels = next(self.aug.flow(images, labels, \
                            batch_size=self.batchSize))
                
                # yield processed images & labels
                yield (images, labels)
            
            # update epoch
            epoch += 1
            pass

    def close(self):
        # close the db
        self.db.close()















