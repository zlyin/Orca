#!/usr/bin/python3.6

## import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from imutils import paths
import matplotlib.pyplot as plt

# import tf config to enable GPU
from keras import backend as K
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.compat.v1.Session(config=config)


import numpy as np
import argparse

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.append("./preprocessing")
sys.path.append("./datasets")
sys.path.append("./nn")
from simplepreprocessor import SimplePreprocessor
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simpledatasetloader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet

from keras.optimizers import SGD


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to the input dataset")
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch_size of images")
parser.add_argument("-e", "--epochs", default=100, type=int, help="number of training epochs")
args = vars(parser.parse_args())

## grab images & preprocess
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.astype("float") / 255.0

## split the dataset & encode categorical labels
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# print(trainX.shape, testX.shape)
# print(trainY.shape, testY.shape)


## compile model & train
print("[INFO] compiling models....")
sgd = SGD(lr=0.05)
model = ShallowNet.build(height=32, width=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# train the network
print("[INFO] train the network....")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=args["batch_size"], epochs=args["epochs"], verbose=3)
# H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=3)

# evaluate
print("[INFO] evaluate the NN...")
predictions = model.predict(testX, batch_size=args["batch_size"])
# convert probs => labels
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))


## plot learning curves
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


