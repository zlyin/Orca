#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import sys
sys.path.append("./nn/conv")
from lenet import LeNet
from minivggnet import MiniVGGNet
from shallownet import ShallowNet
from kfer_lenet import KFER_LeNet

import argparse
from keras.applications import VGG16, VGG19, ResNet50, InceptionV3

## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--include_top", type=int, default=1, \
        help="1/-1 indicates whether to include the head of neural network or not")
parser.add_argument("-m", "--model", type=str, default="vgg16", \
        help="which model model to inspect")
args = vars(parser.parse_args())

networkBanks = {
        "vgg16" : VGG16(weights="imagenet", include_top=args["include_top"] > 0),
        "vgg19" : VGG19(weights="imagenet", include_top=args["include_top"] > 0),
        "resnet50" : ResNet50(weights="imagenet", \
                include_top=args["include_top"] > 0),
        "inceptionv3" : InceptionV3(weights="imagenet", \
                include_top=args["include_top"] > 0),
        "shallownet" : ShallowNet.build(height=28, width=28, depth=3, classes=10),
        "lenet" : LeNet.build(height=28, width=28, depth=3, classes=10),
        "minivgg" : MiniVGGNet.build(height=28, width=28, depth=3, classes=10),
        #"kfer_lenet" : KFER_LeNet.build(height=48, width=48, depth=3, classes=7),
        }

## loading network
print("[INFO] loading network =", args["model"])
model = networkBanks[args["model"]]

# inspect layers
for i, layer in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))









