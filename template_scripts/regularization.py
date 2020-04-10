#!/usr/local/bin/python3.6

## import packages
import os
import sys
import argparse
from imutils import paths

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# add path to import functions
sys.path.append("./preprocessing")
sys.path.append("./datasets")
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader


## construct argument parser, add args
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to dataset")
args = vars(parser.parse_args())

## grab image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

## load in image & preprocess image
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
# reshape images into vectors
data = data.reshape((data.shape[0], 32 * 32 * 3))

## encoder labels & partition dataset
lc = LabelEncoder()
labels = lc.fit_transform(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=5)

## try 3 types of regulizers
for reg in [None, "l1", "l2"]:
    print()
    print("[INFO] add %s regularization to loss function" % reg)

    # define model
    model = SGDClassifier(loss="log", penalty=reg, max_iter=100, learning_rate="constant", eta0=0.001, random_state=42)
    model.fit(trainX, trainY)

    # evaluate on testset
    acc = model.score(testX, testY)
    print("[INFO] %s regularization with accuracy %.2f %%" % (reg, acc * 100))








