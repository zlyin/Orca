#!/usr/local/bin/python3.6

## import pacakges
import os
import sys
import numpy as np


## define class of Preceptron
class Perceptron:
    def __init__(self, N, alpha):
        # initialize weight matrix & learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha
        pass

    def step(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # bias tricks
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over epoches & data points
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):   # loop each row
                # one layer preceptron
                pred = self.step(np.dot(x, self.W))

                # calculate loss & update weights
                if pred != target:
                    error = pred - target
                    self.W += -self.alpha * error * x
              
    def predict(self, X, addBias=True):
        # ensure input is 2D format
        X = np.atleast_2d(X)

        # check if add bias
        if addBias:
            X = np.c_[X, np.ones(X.shape[0])]
        
        # predict
        return self.step(np.dot(X, self.W))


