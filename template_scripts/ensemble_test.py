#!/usr/bin/python3.6

## import packages
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

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10

import numpy as np
import glob # module to list all files
import argparse


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", required=True, \
        help="path to saved models directory")

args = vars(parser.parse_args())


## load in testset
_, (testX, testY) = cifar10.load_data()
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
testY = lb.fit_transform(testY)

labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]


## load in models & predict 
modelFiles = os.path.sep.join([args["models"], "*.model"])
modelPaths = list(glob.glob(modelFiles))
print("[INFO] will ensemble the following models:\n", modelPaths)

# initiate placeholder to store predictions
predictions = []

for i, path in enumerate(modelPaths):
    print("[INFO] loading model %d / %d" % (i + 1, len(modelPaths)))
    # load in base model
    model = load_model(path)

    # predict on test set
    prediction = model.predict(testX, batch_size=64)
    predictions.append(prediction)
    pass

# ensemble predictions from base models
print("[INFO] ensemble model scores...")
avg_predictions = np.average(predictions, axis=0)   # along "models" direction
report = classification_report(avg_predictions.argmax(axis=1), \
        testY.argmax(axis=1), target_names=labelNames)
print(report)

reportPath = os.path.sep.join([args["models"], "ensemble_report.txt"])
with open(reportPath, "w") as f:
    f.write(report)
f.close()





