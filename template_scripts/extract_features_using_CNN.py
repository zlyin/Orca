#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn/conv/")
sys.path.append("./io/")
from hdf5datasetwriter import HDF5DatasetWriter

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from keras.applications import VGG16, VGG19, ResNet50, InceptionV3
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img

from sklearn.preprocessing import LabelEncoder

from imutils import paths
import progressbar
import argparse
import random
import numpy as np


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, \
        help="path to input datasetdataset")
parser.add_argument("-o", "--output", required=True, \
        help="path to output HDF5 file")

parser.add_argument("-m", "--model", type=str, default="vgg16", \
        help="which model to extract features")
parser.add_argument("-b", "--batch_size", type=int, default=32, \
        help="batch size of images")
parser.add_argument("-bf", "--buffer_size", type=int, default=1000, \
        help="size of feature extraction buffer")
args = vars(parser.parse_args())


# exp tip, cache the var
bs = args["batch_size"]
modelBanks = {
        "vgg16" : VGG16(weights="imagenet", include_top=False),
        "vgg19" : VGG19(weights="imagenet", include_top=False),
        "resnet50" : ResNet50(weights="imagenet", include_top=False),
        "inceptionv3" : InceptionV3(weights="imagenet", include_top=False),
        }
inputShapeBanks = {
        "vgg16" : (224, 224),
        "vgg19" : (224, 224),
        "resnet50" : (229, 229),
        "inceptionv3" : (229, 229),
        }
outputShapeBanks = {
        "vgg16" : (7, 7, 512),
        "vgg19" : (7, 7, 512),
        "resnet50" : (8, 8, 2048),
        "inceptionv3" : (8, 8, 2048),
        }


"""
Grab the list of iamges used to extract features;
Shuffle them in order to get easy split of training - test dataset by array sliceing
"""
print("[INFO] loading images...") 
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

## extract class labels & encode into numbers
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)


## load the NN model
print("[INFO] loading network...")
# Do NOT include top softmax layer!
model = modelBanks[args["model"]]

# initiate a HDF5 dataset writer, store the class label names in the dataset
# 512 * 7 * 7 = output volume shape of the penultimate layer in VGG16
optshape = 1
for num in outputShapeBanks[args["model"]]:
    optshape *= num

dataset = HDF5DatasetWriter(args["output"], (len(imagePaths), optshape), \
        dataKey="features", bufSize=args["buffer_size"])

dataset.storeClassLabels(le.classes_)


"""
Loop over imagePaths to load in images in batch;
"""
# create a progress bar
widgets = ["Extracting Features :", progressbar.Percentage(), " ", \
        progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over imagePaths
for i in np.arange(0, len(imagePaths), bs):
    # get imagePaths & labels of current batch
    batchPaths = imagePaths[i : i + bs]
    batchLabels = labels[i : i + bs]
    batchImages = []

    # load images of current batch
    for j, imgPath in enumerate(batchPaths):
        # VGG16 is trained on 224 * 224 images
        image = load_img(imgPath, target_size=inputShapeBanks[args["model"]])
        image = img_to_array(image)

        # add one more dim
        image = np.expand_dims(image, axis=0)
        
        # Special for ImageNet dataset => substracting mean RGB pixel intensity
        image = imagenet_utils.preprocess_input(image)
        
        batchImages.append(image)
        pass

    # stack vertically the images in the batch
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs) # scores before softmax

    # flatten features of each image w.r.t to output volume shape
    features = features.reshape((features.shape[0], optshape))

    # add the feature & labels into HDF5 dataset
    dataset.add(features, batchLabels)

    # update pbar
    pbar.update(i)
    pass

# close the dataset when finished
dataset.close()
pbar.finish()


