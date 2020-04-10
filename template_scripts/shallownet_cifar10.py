#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn")
from nn.conv.shallownet import ShallowNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import argparse

# import tf config to enable GPU
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
session = tf.compat.v1.Session(config=config)

from keras import backend as K
from keras.datasets import cifar10
from keras.optimizers import SGD

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch_size of images")
parser.add_argument("-e", "--epochs", default=100, type=int, help="number of training epochs")
args = vars(parser.parse_args())

## load training & testing data
print("[INFO] loading CIFAR-10 dataset...")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# record label names of CIFAR10
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]


## initialize models
print("[INFO] compiling models...")
sgd = SGD(lr=0.01)
model = ShallowNet.build(height=32, width=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
metrics=["accuracy"])

print("[INFO] training...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32,
        epochs=args["epochs"], verbose=1)


print("[INFO] testing...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
    target_names=labelNames))

## plot the learning curve
plt.style.use("ggplot")
plt.figure()
x = np.arange(0, args["epochs"])
plt.plot(x, H.history["loss"], label="train_loss")
plt.plot(x, H.history["val_loss"], label="val_loss")
plt.plot(x, H.history["accuracy"], label="train_acc")
plt.plot(x, H.history["val_accuracy"], label="val_acc")
plt.title("Learning curves")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()



