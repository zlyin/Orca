#!/usr/local/bin/python3.6

## import packages
import os
import sys
sys.path.append("./nn")
from nn.neuralnetwork import NeuralNetwork

from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

## Load in mnist dataset
print("[INFO] Loading MNIST samples...")
digits = datasets.load_digits()
data = digits.data.astype("float")

# min-max normalization
data = (data - data.min()) / (data.max() - data.min())
print("[INFO] samples = {}, dim = {}".format(data.shape[0], data.shape[1]))

## split datasets 75%-25%
trainX, testX, trainY, testY = train_test_split(data, digits.target, test_size=0.25)

# One-hot encoding targets 
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

## train the model
print("[INFO] training network....")
nn = NeuralNetwork([trainX.shape[1], 32, 16, 16, 10], alpha=0.5)
print("[INFO] {}".format(nn))
displayLoss = nn.fit(trainX, trainY, epochs=1000)

## test
print("[INFO] evaluating...")
pred_probs = nn.predict(testX)
pred_labels = pred_probs.argmax(axis=1)
print(classification_report(pred_labels, testY.argmax(axis=1)))


# plot learning curve
plt.figure()
plt.plot(np.arange(0, 1100, 100), displayLoss)
plt.title("loss of on MNIST samples".format(nn))
plt.xlabel("epoch #")
plt.ylabel("loss")
plt.show()

