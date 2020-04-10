#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn")
sys.path.append("./preprocessing")
sys.path.append("./datasets")
from nn.conv.shallownet import ShallowNet
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=1.0
sess = tf.compat.v1.Session(config=config)
from keras.optimizers import SGD

import numpy as np
from imutils import paths
import argparse
import matplotlib.pyplot as plt

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to input datasetdataset")
parser.add_argument("-m", "--model", default=None, help="name of output model")
parser.add_argument("-o", "--output_dir", default=None, help="name of output model")
parser.add_argument("-e", "--epochs", type=int, default=40, help="path to input datasetdataset")
args = vars(parser.parse_args())

dataset_name = args["dataset"].split("/")[1]

## load datasets
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(imagePaths, verbose=1000)
data = data.astype("float") / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

## initialize the model & train
print("[INFO] compiling model...")
model = ShallowNet.build(height=32, width=32, depth=3, classes=3)
sgd = SGD(lr=0.005)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=args["epochs"], verbose=1)


## evaluate
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))


## plot learning curves and save model file
print("[INFO] serializing the NN...")
if args["output_dir"] and args["model"]:
    model_file = args["model"] + "_on_%s.hdf5" % dataset_name
    output_model_path = os.path.join(args["output_dir"], model_file)
    model.save(output_model_path)


plt.style.use("ggplot")
plt.figure()
x = np.arange(0, args["epochs"])
plt.plot(x, H.history["loss"], label="train_loss")
plt.plot(x, H.history["val_loss"], label="val_loss")
plt.plot(x, H.history["accuracy"], label="train_acc")
plt.plot(x, H.history["val_accuracy"], label="val_acc")
plt.title("Learning curves of %s on %s" % (args["model"], dataset_name))
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
if args["output_dir"] and args["model"]:
    picture_name = args["model"] + "_on_%s_learning_curves.png" % dataset_name
    output_plot_path = os.path.join(args["output_dir"], picture_name)
    plt.savefig(output_plot_path)
plt.show()


 












