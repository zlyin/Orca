#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import sys
sys.path.append("./nn/conv/")
sys.path.append("./preprocessing/")
sys.path.append("./datasets/")
from aspectawarepreprocessor import AspectAwarePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simpledatasetloader import SimpleDatasetLoader
from minivggnet import MiniVGGNet

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from imutils import paths
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scipy.io

## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to dataset")

parser.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs")
parser.add_argument("-aug", "--augment_data", type=bool, default=False, \
        help="if need to augment data")
args = vars(parser.parse_args())


## load in dataset images
print("[INFO] loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [path.split(os.path.sep)[-2] for path in imagePaths]
classNames = list(set(classNames))

# initialize image preprocessors
aap = AspectAwarePreprocessor(64, 64)
itap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, itap])

data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0     # normalize into [0, 1]

# split dataset
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, \
        random_state=42)

# convert data & labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


## compiling model
print("[INFO] compiling model...")
sgd = SGD(lr=0.05)
#sgd = SGD(lr=0.05, decay= 0.05 / args["epochs"])

model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", \
        optimizer=sgd, metrics=["accuracy"])


## train with / without data augmentation
if args["augment_data"]:
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
            height_shift_range=0.1, shear_range=0.1, zoom_range=0.2, \
            horizontal_flip=True, fill_mode="nearest")
    
    # need to augment trainX & trainY simultaneously!
    augImages = aug.flow(trainX, trainY, batch_size=32)

    print("[INFO] training models WITH Data Augmentation....")
    H = model.fit_generator(augImages, validation_data=(testX, testY), \
            steps_per_epoch=len(trainX) // 32, epochs=args["epochs"], verbose=1)

else:
    print("[INFO] training models WITHOUT data augmentation....")
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32,\
            epochs=args["epochs"], verbose=1)


## evaluating 
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), 
    target_names=lb.classes_))


## plot learning curves
plt.style.use("ggplot")
plt.figure()
x = np.arange(0, args["epochs"])
plt.plot(x, H.history["loss"], label="train_loss")
plt.plot(x, H.history["val_loss"], label="val_loss")
plt.plot(x, H.history["accuracy"], label="train_acc")
plt.plot(x, H.history["val_accuracy"], label="val_acc")
title = "learning curves with Data Augmentation" if args["augment_data"] else \
        "learning curves"
plt.title(title)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()



"""
conclusion:
    WITHOUT lr_decay & data_augmentation ==> acc=0.61, Highly Overfitting;
    WITH only data_augmentation ==> acc=0.75, Slightly Overfitting;
    WITH lr_decay & data_augmentation ==> acc=0.81, Slightly Overfitting;
"""
