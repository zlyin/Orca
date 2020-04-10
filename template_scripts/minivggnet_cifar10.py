#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn/conv")
from minivggnet import MiniVGGNet

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras.datasets import cifar10
from keras import backend as K
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils


"""
# USAGE
- End-to-End training process, using MiniVGGNet to train cifar10
- Use Keras default time-based learning decay schedule
"""


## Build arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, \
        help="path to the output loss/accuracy plot")
parser.add_argument("-e", "--epochs", type=int, default=40, \
        help="# of epochs")
args = vars(parser.parse_args())


## Fetch dataset & preprocessing
print("[INFO] Fetch CIFAR-10 ....")
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]

## Initialize model & train & evaluate
print("[INFO] compiling model...")
sgd = SGD(lr=0.01, decay=0.01 / args["epochs"], momentum=0.9, nesterov=True)
model = MiniVGGNet.build(height=32, width=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=sgd, \
        metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, \
        epochs=args["epochs"], verbose=1)

print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), \
        target_names=labelNames))

## plot learning curve & save
plt.style.use("ggplot")
plt.figure()
x = np.arange(0, args["epochs"])
plt.plot(x, H.history["loss"], label="train_loss")
plt.plot(x, H.history["val_loss"], label="val_loss")
plt.plot(x, H.history["accuracy"], label="train_acc")
plt.plot(x, H.history["val_accuracy"], label="val_acc")
plt.legend()
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.title("MiniVGGNet on CIFAR-10")
plotpath = os.path.join(args["output"], "MiniVGGNet_on_CIFAR10_Epoch40.png")
plt.savefig(plotpath)


