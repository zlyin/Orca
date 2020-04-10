## import packages
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import BaseLogger


## define training monitor class
class TrainingMonitor(BaseLogger):
    
    def __init__(self, figPath, jsonPath=None, startAt=0):

        """
        define initiate method, inherate parent class initiate method.
        note that 'super(cls, instance)' == return next class of cls, in the MRO of instance
        MRO = Method Resolution Order, get from left -> right
        """
        super(TrainingMonitor, self).__init__() # in MRO of self/TM, find TM's next class
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt
        pass

    def on_train_begin(self, log={}):
        """
        load the training history into self.H if self.jsonPath exists;
        update self.H up unitil the self.startAt epoch, since that's where we \
                resume training from;
        """
        self.H = {}

        if self.jsonPath and os.path.exists(self.jsonPath):
            # load in serialized json file
            self.H = json.loads(open(self.jsonPath).read())

            # loop over entries in the json file, trim list entries that past
            # self.startAt epoch
            if self.startAt > 0:
                for k in self.H.keys():
                    self.H[k] = self.H[k][:self.startAt]
            pass
        #print("debug", self.H.keys()) 

    def on_epoch_end(self, epoch, logs={}):
        """
        most_important method that is automatically employed by Keras;
        update params in self.H when each epoch finishes;
        args = logs stores metrics/info for the current epoch;
        """  
        for key, val in logs.items():
            l = self.H.get(key, [])
            l.append(float(val)) # np.float32 can't be serialized!
            self.H[key] = l

        # check if need to serialize the training history to file
        #print(self.H)

        if self.jsonPath:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # update learning curves in a NEW plot when each epoch ends
        if len(self.H["loss"]) > 1:
            x = np.arange(0, len(self.H["loss"]))
            
            super_large_var = False
            plt.style.use("ggplot")
            plt.figure()
            # loop & plot the rest curves
            for k in self.H.keys():
                # debug
                #print("key=", k, self.H[k])
                # avoid learning rate decay
                if k == "lr": 
                    continue
                # enforce ylim
                if max(self.H[k]) > 20:
                    super_large_var = True
                plt.plot(x, self.H[k], label=k)

            plt.xlabel("Epoch #")
            plt.ylabel("Loss / Metrics")
            plt.title("Learning Curves at Epoch=%d" % len(self.H[k]))
            plt.legend()
            if super_large_var == True:
                plt.ylim((0, 3))
            # save fig to output path & close figure
            plt.savefig(self.figPath)
            plt.close() 


