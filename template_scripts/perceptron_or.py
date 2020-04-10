#!/usr/local/bin/python3.6

## import package 
import os
import sys
import numpy as np
sys.path.append("./nn")
from perceptron import Perceptron


## create a OR dataset
X = np.array([[0, 0], [0, 1],[1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

## initialize preceptron & train it
print("[INFO] training preceptron..")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

print("[INFO] testing preceptron..")
for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, gt={}, pred={}".format(x, target[0], pred))






