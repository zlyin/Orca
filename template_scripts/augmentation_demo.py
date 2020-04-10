#!/usr/bin/python3.6

## import package
import os
import sys
import numpy as np
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


## construct argument
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="path to input image")
parser.add_argument("-o", "--output", required=True, help="path to augmented \
        images")

parser.add_argument("-p", "--prefix", type=str, default="image", help="output \
        file name")
args = vars(parser.parse_args())


## load in an image
print("[INFO] loading image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


## initiate data agumentation generator
# create a ImageDataGenerator(), with a bunch of augmentation types
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, \
        horizontal_flip=True, fill_mode="nearest")

total = 0 

# ImageDataGenerator.flow() create a list generator
ImageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"], \
        save_prefix=args["prefix"], save_format="jpg")

print("[INFO] augmenting image...")
for img in ImageGen:
    total += 1

    if total == 10: break
    pass








