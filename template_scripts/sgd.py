#!/usr/local/bin/python3.6

## import packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs


## define methods
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    preds = sigmoid(X.dot(W))
    # converts scores to labels
    preds[preds > 0.5] = 1
    preds[preds < 1] = 0
    return preds

def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i +  batch_size])


## parse argument
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs")
parser.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch size for SGD")
args = vars(parser.parse_args())


## generate normal distribution random data
X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
# add 1-col into X
X = np.c_[X, np.ones((X.shape[0], 1))]
# reshape y
y = y.reshape((y.shape[0], 1))
# split the dataset
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5, random_state=42)


## initialize weights W & do training
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    epochLoss = []

    # loo over batches
    for batch_X, batch_Y in next_batch(trainX, trainY, args["batch_size"]):
        # compute batch_preds
        batch_preds = sigmoid(batch_X.dot(W))
        batch_error = batch_preds - batch_Y
        epochLoss.append(np.sum(batch_error ** 2))
        
        # BP
        Wgradient = batch_X.T.dot(batch_error)
        W += -args["alpha"] * Wgradient
        pass
    
    # in each epoch, record total loss
    loss = np.average(epochLoss)
    losses.append(loss)

    # print out training status

## evaluate the model
print("[INFO] evaluating....")
test_preds = predict(testX, W)
print(classification_report(test_preds, testY))


## plot learning curves
plt.style.use("ggplot")
plt.figure()

ax1 = plt.subplot(121)
ax1.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY.flatten(), s=30)
ax1.set_title("Data")

ax2 = plt.subplot(122)
ax2.plot(np.arange(0, args["epochs"]), losses)
ax2.set_title("Training Loss")
ax2.set_xlabel("epochs")
ax2.set_ylabel("loss")

plt.show()
plt.key_wait(0)
