#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIABLE_DEVICES"] = "0"

import sys
sys.path.append("./nn/conv")
from minivggnet import MiniVGGNet

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import keras.backend as K
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


"""
# USAGE
- End-to-End training process, using MiniVGGNet to train cifar10
- Implement a lr step_decay method & feed into LearningRateScheduler
"""


## define learning rate step_decay method
def step_decay(epoch):  
    """
    Noting that LearningRateScheduler only reads in arg=epoch!
    Have to define params within the method!
    """
    initAlpha = 0.01
    factor = 0.75
    dropEvery = 5
    # use default params to compute lr_decay
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    
    return float(alpha)
    

## build up argument
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, \
        help="path to output loss/acc plot")
parser.add_argument("-lrStepDecay", "--learning_rate_step_decay", type=bool, \
        default=True, help="use step-decay to control learning rate")
parser.add_argument("-e", "--epochs", type=int, default=40, \
        help="path to output loss/acc plot")
args = vars(parser.parse_args())


## load & preprocess the data
print("[INFO] loading CIFAR-10....")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]


## train & evaluate the model
print("[INFO] compiling the model...")
model = MiniVGGNet.build(height=32, width=32, depth=3, classes=10)

# time-based lr decay
if args["learning_rate_step_decay"] == False:
    sgd = SGD(lr=0.01, decay=0.01 / args["epochs"], momentum=0.9, nesterov=True)
else:
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, \
        metrics=["accuracy"])

print("[INFO] training the model...")

# step-based lr decay
if args["learning_rate_step_decay"]:
    print("[INFO] using step-decay of learning rate...")

    """
    schedule: a function that takes an epoch index as input 
    (integer, indexed from 0) and current learning rate 
    and returns a new learning rate as output (float).
    """
    callbacks = [LearningRateScheduler(step_decay)]

    H = model.fit(trainX, trainY, validation_data=(testX, testY), \
            batch_size=64, epochs=args["epochs"], \
            callbacks=callbacks, verbose=100)
else:
    H = model.fit(trainX, trainY, validation_data=(testX, testY), \
            batch_size=64, epochs=args["epochs"], verbose=100)

print("[INFO] eval_argumentuating the model...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), \
        target_names=labelNames))


## plot the learning curve
plt.style.use("ggplot")
plt.figure()
x = np.arange(0, args["epochs"])
plt.plot(x, H.history["loss"], label="training_loss")
plt.plot(x, H.history["accuracy"], label="training_acc")
plt.plot(x, H.history["val_loss"], label="val_loss")
plt.plot(x, H.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Learning curves")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Acccuracy")
#plt.show()
plt.savefig(args["output"])



