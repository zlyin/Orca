#!/usr/local/bin/python3.6

## import packages
import os
import sys
import argparse

# import sklearn
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# import relevant unitilies; need to add folders to os.sys.path!
sys.path.append("./preprocessing")
sys.path.append("./datasets")

from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from imutils import paths


## construct argument parser, add args
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", required=True, help="path to input dataset")
parser.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
parser.add_argument("-j", "--jobs", type=int, default=-1, help="# of cores kNN classifier (-1 uses all cores)")

"""
vars takes an object as a parameter; 
e.g{'dataset': '../animal_image_dog_cat_and_panda/', 'neighbors': 5, 'jobs': -1}
"""
args = vars(parser.parse_args())


## load images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
#print(imagePaths[:10])
#e.g.= '../animal_image_dog_cat_and_panda/panda/panda_00528.jpg'

# initiate the image preprocessor, set fixed_image size
simpro = SimplePreprocessor(32, 32)
simloader = SimpleDatasetLoader(preprocessors=[simpro])
data, labels = simloader.load(imagePaths, verbose=500)
data = data.reshape(data.shape[0], 32*32*3)

# show information about memory consumption
print("[INFO] features matrix consumes %.1f MB" % (data.nbytes /(1024 * 1000.0)))


## encoder & split dataset
le = LabelEncoder()
labels = le.fit_transform(labels)

# split dataset into train & test
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

## train & evaluate kNN
print("[INFO] training kNN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))


