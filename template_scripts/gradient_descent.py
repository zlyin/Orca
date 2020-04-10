#!/usr/local/bin/python3.6

# import pacakges
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse


# define sigmoid activation function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# define label predictions method
def predict(X, W):
    preds = sigmoid(X.dot(W))

    # apply a threshold to output binary labels
    preds[preds > 0.5] = 1
    preds[preds < 1] = 0
    return preds


# argument parse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs")
parser.add_argument("-a", "--alpha", type=float, default=0.01, help="learing rate")
args = vars(parser.parse_args())


## generate a random 2D datasets from blobs
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
# insert a col of 1s into X, "bias trick"
X = np.c_[X, np.ones((X.shape[0], 1))] # shape=(1000, 3)
# partition the dataset into 50-50%
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.5, random_state=42)


## randomly initialize weights
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    # get preds & compute errors
    preds = sigmoid(trainX.dot(W))
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # gradient descent & update weights
    Wgradient = trainX.T.dot(error)
    W += -args["alpha"] * Wgradient

    # print out status update
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch = %d, loss = %.7f" % (epoch, loss))
    pass


## evaluate on testX & testY
print("[INFO] evaulating....")
test_preds = predict(testX, W)
print(classification_report(testY, test_preds))


## plot dataset & learning curve
plt.style.use("ggplot")
plt.figure()

ax1 = plt.subplot(121)
ax1.set_title("Data")
ax1.scatter(testX[:,0], testX[:, 1], marker="o",c=testY.flatten(), s=30)

ax2 = plt.subplot(122)
ax2.plot(np.arange(0, args["epochs"]), losses)
ax2.set_title("training loss")
ax2.set_xlabel("epochs")
ax2.set_ylabel("loss")
plt.show()
plt.keywait(0)

    





