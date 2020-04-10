#!/usr/local/bin/python3.6

## import packages
import os
import sys
sys.path.append("./nn")
from nn.neuralnetwork import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


# create XOR datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

Epochs = 20000
nn = NeuralNetwork([2,2,1], alpha=0.8)
displayLoss = nn.fit(X, y, epochs=Epochs)

# predict X
print("[INFO] Predicting on XOR...")

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0] # because p is 2d array
    
    # encode in to 1 or 0
    pred_label = 1 if pred > 0.5 else 0
    print("[INFO] data={}, gt={}, pred={}, pred_label={}".format(x, target, pred, pred_label))


# plot learning curve
plt.figure()
plt.plot(np.arange(0, Epochs + 100, 100), displayLoss)
plt.title("larning curve of NN {} on XOR".format([2,2,1]))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()


## To demonstrate one hidden layer is required to separate nonlinear datasets
nn1 = NeuralNetwork([2,1], alpha=0.8)
displayLoss1 = nn1.fit(X, y, epochs=Epochs)

print("[INFO] Predicting on XOR...")
for (x, target) in zip (X,y):
    pred = nn1.predict(x)[0][0]
    pred_label = 1 if pred > 0.5 else 0
    print("[INFO] data={}, gt={}, pred={}, pred_label={}".format(x, target, pred, pred_label))

plt.figure()
plt.plot(np.arange(0, Epochs + 100, 100), displayLoss)
plt.title("larning curve of NN {} on XOR".format([2,1]))
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()















