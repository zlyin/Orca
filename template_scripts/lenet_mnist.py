#!/usr/bin/python3.6

## import packages

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn")
from nn.conv.lenet import LeNet

from keras.optimizers import SGD
from keras import backend as K
from keras.datasets import mnist
from keras.datasets import fashion_mnist

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np


## grab MNIST datasets 
print("[INFO] Fetch MNIST...")
#(trainX, trainY), (testX, testY) = mnist.load_data()
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# encode targets
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

## preprocessing MNIST dataset - channel ordering; MNIST = greyscale, 28x28 images
if K.image_data_format() == "channel_first":
    trainX = trainX.reshape(trainX.shape[0], 1, 28, 28)
    testX = testX.reshape(testX.shape[0], 1, 28, 28)
else:
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0], 28, 28, 1)

## initiate model & train & evaluate
print("[INFO] compling model...")
sgd = SGD(lr=0.01)
model = LeNet.build(height=28, width=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])


print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, \
        epochs=20, verbose=1)


print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),\
        target_names=[str(x) for x in lb.classes_]))


## plot learning curve
plt.style.use("ggplot")
plt.figure()
x = np.arange(0, 20)
plt.plot(x, H.history["loss"], label="train_loss")
plt.plot(x, H.history["val_loss"], label="val_loss")
plt.plot(x, H.history["accuracy"], label="train_acc")
plt.plot(x, H.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("LeNet on MNIST")
plt.xlabel("Epoch #")
plt.ylabel("Loass/Accuracy")
plt.show()










