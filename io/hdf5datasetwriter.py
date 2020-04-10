#!/usr/bin/python3.6

## import packages
import os
import sys
import h5py


## define HDF5DatasetWriter
class HDF5DatasetWriter:
    def __init__(self, outputPath, dataDims, dataKey="images", bufSize=1000,
            labelDtype="int", labelOneHot=0):

        # check if outputPath exists or not
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'outputPath = %s' already exists!"
                    "Use another path or manually delete the file to continue."
                    % outputPath)

        # open a HDF5 database & create datasets to store images/features & labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dataDims, dtype="float")

        if labelDtype == "string":
            dt = h5py.special_dtype(vlen=str)
        else:
            dt = "int" 
        # dim
        if labelOneHot != 0:
            labelDims = (dataDims[0], labelOneHot)
        else:
            labelDims = (dataDims[0], )
        self.labels = self.db.create_dataset("labels", labelDims, dtype=dt)

        # store the buffer size & initialize the buffer itself along with
        # the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data" : [], "labels" : []}
        self.idx = 0
        pass

    
    def add(self, rows, labels):
        # add the list of rows & labels to dataset
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        
        # flush into db if buffer is full
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
        pass

    def flush(self):
        # write files in the buffer to disk and reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx : i] = self.buffer["data"]
        self.labels[self.idx : i] = self.buffer["labels"]

        # update for next flush
        self.idx = i 
        self.buffer = {"data" : [], "labels" : []}
        pass

    def storeClassLabels(self, classLabels):
        # create a dataset to store actual class label names; then store class labels
        #dt = h5py.special_dtype(vlen=unicode)
        dt = h5py.string_dtype(encoding="utf-8")
        
        labelSet = self.db.create_dataset("label_names", (len(classLabels), ), \
                dtype=dt)
        labelSet[:] = classLabels
        pass

    
    def close(self):
        # check if there are any data entries left in the buffer, flush into
        # disk and close
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the db file
        self.db.close()
        pass





