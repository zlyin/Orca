#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VAISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn/conv")
from minivggnet import MiniVGGNet

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weights", required=True, \
        help="path to weights directory")
parser.add_argument("-e", "--epochs", type=int, default=40, help="epochs #")
parser.add_argument("-wow", "--weights_overwrite", type=bool, default=False,
        help="overwrite saved weights file")
args = vars(parser.parse_args())


## load & preprocess dataset
print("[INFO] loading cifar 10...")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


## compile the model
print("[INFO] compile the model...")
sgd = SGD(lr=0.01, decay=0.01 / args["epochs"], momentum=0.9, nesterov=True)
model = MiniVGGNet.build(height=32, width=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=sgd, \
        metrics=["accuracy"])

## initiate ModelCheckpoint into callbacks
if not args["weights_overwrite"]:
    # use a format string to create fname, "{var:format}" = template
    fname = os.path.join(args["weights"], "weights-{epoch:03d}-{val_loss:0.4f}.hdf5")
else:
    # use a simple string to create fname; will overwrite again and again
    fname = args["weights"]

checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", \
        save_best_only=True, verbose=1)
callbacks = [checkpoint]

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, \
        epochs=args["epochs"], callbacks=callbacks, verbose=2)




