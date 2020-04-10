#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn/conv/")
sys.path.append("./preprocessing/")
sys.path.append("./datasets/")
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from fcheadnet import FCHeadNet

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop
from keras.applications import VGG16

from keras.layers import Input  # important!
from keras.models import Model  # important!

from imutils import paths
import argparse
import numpy as np


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to dataset")
parser.add_argument("-o", "--output_dir", required=True, \
        help="path to output model directory")

# add arg "nargs='+'" to take in multiple input for this arg!
parser.add_argument("-ft", "--fine_tune_layers", required=True, nargs="+", \
        help="a list of indices that indicates unfrozen layers")

args = vars(parser.parse_args())

print(args["fine_tune_layers"])

## construct data augmentation for fine tuning
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, \
        horizontal_flip=True, fill_mode="nearest")


## loading images
print("[INFO] loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [path.split(os.path.sep)[-2] for path in imagePaths]
classNames = list(set(classNames))

## preprocess images to (224, 224) for VGG16
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])

data, labels = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, \
        stratify=labels, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


## network surgery
# load VGG16 network without head layers, explicitly define input_tensor
x = Input(shape=(224, 224, 3))
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=x) 
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# define a final model which takes baseModel.input as inputs and outputs headModel
model = Model(inputs=baseModel.input, outputs=headModel)

print("[INFO] layers of new model...")
for i, layer in enumerate(model.layers):
    print(i, layer.__class__.__name__)


## FREEZE the backbone & train the new head layers for 10-30 epochs
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model for new head layers warm-up...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training to warm-up new head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), \
        validation_data=(testX, testY), \
        steps_per_epoch=len(trainX) // 32, \
        epochs=25, verbose=1)

print("[INFO] evaluating after warm-up...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), \
        target_names=classNames))


## UNfreeze the CONV layers NEAR the new head & train to localize the weights
print("[INFO] fine tune the model in %d stages" % len(args["fine_tune_layers"]))
for i, num in enumerate(args["fine_tune_layers"]):
    # unfree layers
    index = int(num)
    for layer in model.layers[index:]:
        layer.trainable = True

    print("\n[INFO] stage %d/%d, unfreez layers from %d & recompile model..." %
            (i + 1, len(args["fine_tune_layers"]), index))
    sgd = SGD(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    print("[INFO] fine tuning the model...")
    model.fit_generator(aug.flow(trainX, trainY, batch_size=32), \
            validation_data=(testX, testY), \
            steps_per_epoch=len(trainX) // 32, \
            epochs=50, verbose=1)
   
    print("[INFO] evaluating...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), \
            target_names=classNames))
    
    # save current model
    outputName = "fine_tune_stage_%d_from_layer_%d.hdf5" % (i + 1, index)
    filepath = os.path.join(args["output_dir"], outputName)
    model.save(filepath)
    pass



