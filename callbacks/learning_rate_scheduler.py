# import packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


## define LearningRateDecay class
class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        """
        - present learning rate decay process;
        - epochs = # of epochs;
        - title = title of plot"
        - no return
        """
        lrs = [self(i) for i in epochs]

        # plot the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.close()


## define StepDecay class
class StepDecay(LearningRateDecay):

    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        """
        - initiate StepDecay class with initial lr, decay factor & decay period
        """
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
        pass

    def __call__(self, epoch):
        """
        - compute the learning rate for the current epoch using step-based decay
        function
        - accept current epoch number
        - return alpha, float type
        """
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = initAlpha * (self.factor ** exp)
        
        # return alpha 
        return float(alpha)


## define linear & polynomial learning rate decay
class PolynomialDecay(LearningRateDecay):

    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        """
        - initialize polynomial learning rate decay schedule with 3 args"
        - maxEpochs = # of training epochs;
        - initAlpha = initial learning rate;
        - power = power/exponent of the polynomial; linear if power=1.0,
          quadratic if power=2.0, etc.
        """
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power
        pass

    def __call__(self, epoch):

        """
        - compute the learning rate for the current epoch based polynomical
          decay
        function
        - accept current epoch number
        - return alpha, float type
        """
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        
        # return alpha
        return float(alpha)
