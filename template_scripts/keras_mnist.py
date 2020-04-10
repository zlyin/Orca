#!/usr/local/bin/python3.6

## import packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt
import argparse

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

## construct argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, help="path to the output of loss/acc plot")
args = vars(parser.parse_args())


## grab full MNIST datasets
print("[INFO] download full MNIST dataset...")
mnist = datasets.fetch_mldata("MNIST Original")

data = mnist.data.astype("float") / 255
trainX, testX, trainY, testY = train_test_split(data, mnist.target, test_size=0.25)

# label encode Y
lb = LabelBinarizer()
trainY, testY = lb.fit_transform(trainY), lb.transform(testY)


## define NN architecture 784(i.e. 32*32*3)-256-128-10
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))


## train the model with SGD
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
# get history of training process
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)


## evaluate the NN
print("[INFO] evaluating NN...")
pred_probs = model.predict(testX, batch_size=128)
pred_labels = pred_probs.argmax(axis=1)
print(classification_report(pred_labels, testY.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))


## plot learning curve
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss & Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss & Acc")
plt.legend()
plt.show()
plt.savefig(args["output"])



