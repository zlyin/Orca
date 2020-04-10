#!/usr/bin/python3.6

# import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn/conv/")
from minivggnet import MiniVGGNet

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
session = tf.compat.v1.Session(config=config)

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import argparse
import h5py
import pickle   # serialization tool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True, \
        help="path to output directory")

parser.add_argument("-e", "--epochs", type=int, default=40, \
        help="# of epochs")
parser.add_argument("-n", "--num_models", type=int, default=5, \
        help="# of models to ensemble")
args = vars(parser.parse_args())


## load CIFAR-10 and preprocess data
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize label names of CIFAR-10
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]

## initiate data agumentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, \
        height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")


## for loop args["num_models"] to train weak classifiers
for i in range(args["num_models"]):
    # compile model 
    print("[INFO] training base classifier %d / %d..." % (i + 1,
    args["num_models"]))

    sgd = SGD(lr=0.01, decay=0.01 / args["epochs"], momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, \
            metrics=["accuracy"])
    
    # train the model
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64), \
            validation_data=(testX, testY), epochs=args["epochs"], \
            steps_per_epoch=len(trainX) // 64, verbose=1)

    # save base classifier to disk
    modelpath = [args["output"], "base_model_%d.model" % (i + 1)] 
    model.save(os.path.sep.join(modelpath))

    # evaluate & write classification_report to disk as well
    predictions = model.predict(testX, batch_size=64)
    report = classification_report(predictions.argmax(axis=1), \
            testY.argmax(axis=1), target_names=labelNames)
    reportpath = [args["output"], "report_%d.txt" % (i + 1)]
    with open(os.path.sep.join(reportpath), "w") as f:
        f.write(report)
    f.close()

    # plot learning curves
    plt.figure()
    plt.style.use("ggplot")

    x = np.arange(0, args["epochs"])
    plt.plot(x, H.history["loss"], label="train_loss") 
    plt.plot(x, H.history["val_loss"], label="val_loss") 
    plt.plot(x, H.history["accuracy"], label="train_accuracy") 
    plt.plot(x, H.history["val_accuracy"], label="val_accuracy") 
    plt.title("Learning curves of model %d / %d" % (i + 1, args["num_models"]))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    
    plotpath = [args["output"], "learning_curves_%d.png" % (i + 1)]
    plt.savefig(os.path.sep.join(plotpath))

    # otherwise will plot all curves on the same plot
    plt.close() 
    pass


