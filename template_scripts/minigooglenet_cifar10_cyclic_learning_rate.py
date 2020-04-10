#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIABLE_DEVICES"] = "0"

import sys
sys.path.append("./nn/conv/")
from minigooglenet import MiniGoogLeNet
sys.path.append("./callbacks/")
from cyclic_learning_rate import CyclicLR
sys.path.append("./configs/")
import cyclic_lr_config as config

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tfconfig)

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


"""
# USAGE
- End-to-End training process, using MiniGoogLeNet on cifar-10
- Use CyclicLR as a callback to demo 3 types of lr decay, linear/polynomial/exp
- Use config.py to record all command arguments instead of argparse.ArgumentParser()
"""


## load & preprocess data
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# apply mean subtraction to the data
#mean = np.mean(trainX, axis=0)
#trainX -= mean
#testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# construct for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, \
        horizontal_flip=True, fill_mode="nearest")


## initialize the optimizer and model
# initialize the cyclical learning rate callback
for clr_method in config.CLR_METHODS:

    # Note that using min_lr to initiate an SGD optimizer & compile model
    print("[INFO] compiling model...")
    opt = SGD(lr=config.MIN_LR, momentum=0.9)
    model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, \
            metrics=["accuracy"])

    print("\n[INFO] using '{}' method".format(clr_method))
    clr = CyclicLR(
            mode=clr_method, \
            base_lr=config.MIN_LR, \
            max_lr=config.MAX_LR, \
            step_size= config.STEP_SIZE * (trainX.shape[0] // config.BATCH_SIZE))

    ## train the network
    print("[INFO] training network...")
    H = model.fit_generator(
            aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), \
            validation_data=(testX, testY), \
            steps_per_epoch=trainX.shape[0] // config.BATCH_SIZE, \
            epochs=config.NUM_EPOCHS, \
            callbacks=[clr], \
            verbose=1)

    ## evaluate the network and show a classification report
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
    print(classification_report(testY.argmax(axis=1), \
            predictions.argmax(axis=1), \
            target_names=config.CLASSNAMES))


    ## construct a plot that plots and saves the training history
    x = np.arange(0, config.NUM_EPOCHS)
    plt.style.use("ggplot")

    plt.figure()
    plt.plot(x, H.history["loss"], label="train_loss")
    plt.plot(x, H.history["val_loss"], label="val_loss")
    plt.plot(x, H.history["accuracy"], label="train_acc")
    plt.plot(x, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy - %s" % clr_method)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    TRAINING_PLOT_PATH = os.path.sep.join([config.OUTPUT, \
            "%s_training_plot.png" % clr_method])
    plt.savefig(TRAINING_PLOT_PATH)

    ## plot the learning rate history
    y = np.arange(0, len(clr.history["lr"]))
    plt.figure()
    plt.plot(y, clr.history["lr"])
    plt.title("Cyclical Learning Rate (CLR) - %s" % clr_method)
    plt.xlabel("Training Iterations")
    plt.ylabel("Learning Rate")

    CLR_PLOT_PATH = os.path.sep.join([config.OUTPUT, \
            "%s_clr_plot.png" % clr_method])
    plt.savefig(CLR_PLOT_PATH)

    pass
