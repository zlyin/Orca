#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import sys
sys.path.append("./callbacks")
sys.path.append("./nn/conv")
from trainingmonitor import TrainingMonitor 
from minivggnet import MiniVGGNet

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from keras import backend as K
from keras.optimizers import SGD
from keras.datasets import cifar10

from sklearn.preprocessing import LabelBinarizer

import numpy as np
import matplotlib
matplotlib.use("Agg")
import argparse


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, \
        help="path to output learning curves plot")
parser.add_argument("-e", "--epochs", type=int, default=100, \
        help="epochs # to train")

args = vars(parser.parse_args())


## load & preprocess training data
print("[INFO] loading data...")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]

## compiling the model
print("[INFO] compling model...")
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True) # not use lr decay on purpose
model = MiniVGGNet.build(height=32, width=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=sgd, \
        metrics=["accuracy"])

## contruct TrainMonitor callback & train the network
figPath = os.path.join(args["output"], "{}.png".format(os.getpid()))
jsonPath = os.path.join(args["output"], "{}.json".format(os.getpid()))

callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, \
        epochs=args["epochs"], callbacks=callbacks, verbose=20)























