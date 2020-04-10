#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TF only
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

#from PIL import Image
#print(Image.__file__)
#sys.modules['Image'] = Image 

import sys
import numpy as np
import argparse
import cv2



## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, \
        help="path to the input image")
parser.add_argument("-m", "--model", type=str, default="vgg16", \
        help="pretrained model to be used")
args = vars(parser.parse_args())


## create a {args : model} mapping dict
MODELS = {
        "vgg16" : VGG16,
        "vgg19" : VGG19,
        "inception" : InceptionV3,
        "xception" : Xception,
        "resnet" : ResNet50,
        }

if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be "
            "among vgg16/vgg19/inception/xception/resnet")

## set preprocessing method accroding to different models
"""
VGG16, VGG19, ResNet accpet 224 x 224;
InceptionV3, Xception accept 229 x 229;
"""
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ["xception", "inception"]:
    inputShape = (229, 229)
    preprocess = preprocess_input   # different from last one!


## create model & classify image
print("[INFO] loading {} ...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet") # load pretrained weights on imagenet

print("[INFO] loading & pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)     # image.shape(H, W, channel)

image = np.expand_dims(image, axis=0)
image = preprocess(image)       # call preprocess method to preprocess image

print("[INFO] classifying image with {} ...".format(args["model"]))
predictions = model.predict(image)

# call .decode_predictions() to return top-5 predictions
probs = imagenet_utils.decode_predictions(predictions)  

for i, (imagenetID, label, prob) in enumerate(probs[0]):
    print("{}. {} : {:.2f}%".format(i + 1, label, prob * 100))


## show the image with classified label
orig = cv2.imread(args["image"])
imagenetID, label, prob = probs[0][0]
cv2.putText(orig, "Label={}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,\
        1.0, (255, 0, 0), 2)
cv2.imshow("classified result", orig)
cv2.waitKey(5000)


sys.exit()
