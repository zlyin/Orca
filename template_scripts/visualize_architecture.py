#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
print(tf.executing_eagerly())

import sys
sys.path.append("./nn/conv")
from lenet import LeNet
from minivggnet import MiniVGGNet
from shallownet import ShallowNet
#from kfer_lenet import KFER_LeNet
from alexnet import AlexNet
from alexnet2 import AlexNet2
from resnet import ResNet50, ResNet101, ResNet152

from keras.applications import VGG16, VGG19, InceptionV3
from keras.applications import ResNet50 as KerasResNet50
from keras.utils import plot_model

import argparse


## construct argument
parser = argparse.ArgumentParser()
parser.add_argument("-nn", "--network", required=True, \
        help="name of neural network")
parser.add_argument("-o", "--output", required=True, \
        help="path to output pictures")
args = vars(parser.parse_args())


networkBanks = {
        "vgg16" : VGG16(weights="imagenet"),
        "vgg19" : VGG19(weights="imagenet"),
        "kerasresnet50" : KerasResNet50(weights="imagenet"),
        "inceptionv3" : InceptionV3(weights="imagenet"),
        "shallownet" : ShallowNet.build(height=28, width=28, depth=3, classes=10),
        "lenet" : LeNet.build(height=28, width=28, depth=3, classes=10),
        "minivgg" : MiniVGGNet.build(height=28, width=28, depth=3, classes=10),
        #"kfer_lenet" : KFER_LeNet.build(height=48, width=48, depth=3, classes=7),
        "alexnet" : AlexNet.build(height=227, width=227, depth=3, classes=1000),
        "alexnet2" : AlexNet2.build(height=227, width=227, depth=3, classes=1000),
        "resnet50" : ResNet50(height=224, width=224, depth=3, classes=10),
        "resnet101" : ResNet101(height=224, width=224, depth=3, classes=10),
        "resnet152" : ResNet152(height=224, width=224, depth=3, classes=10),
        }



model = networkBanks[args["network"]]

arch_path = os.path.join(args["output"], \
        "{}_architecture.png".format(args["network"]))


"""
show_shapes = print out each layer's shape;

or use tf.compat.v1.keras.utils.plot_model() / tf.keras.utils.load_model(), with
    rankdir = TB/LR, plotting layers in V/H direction
"""
plot_model(model, to_file=arch_path, show_shapes=True)

#tf.compat.v1.keras.utils.plot_model(model, to_file=arch_path, show_shapes=True, \
#        rankdir="TB")

print("[INFO] ploting model Done!")


