#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tfconfig)

import sys
sys.path.append("./nn/conv")
from resnet import ResNet
sys.path.append("./callbacks")
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
sys.path.append("./utils")
from label_smoothing import label_smooth    # test label_smooth function

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import imutils


"""
# USAGE
- End-to-End training process, using ResNet to train cifar10;
- Use Keras LearningRateScheduler & apply polynomial decay of learning rate
"""
## Build arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoints", required=True, \
        help="path to the model weights & learning curves")
parser.add_argument("-m", "--model", type=str, \
        help="path to a specific loaded model")
parser.add_argument("-s", "--start_epoch", type=int, default=0, \
        help="epoch to start training from")
args = vars(parser.parse_args())

assert os.path.exists(args["checkpoints"])


## Params
EPOCHS = 30
INIT_LR = 5e-3
BATCH = 128
SMOOTHING = 0.1

def poly_decay(epoch):
    """
    polynomial learning rate decay: alpha = alpha0 * (1 - epoch/num_epochs) ** p
    - alpha0 = initial learning rate
    - p = exp index, can be 1, 2, 3 ... etc
    - epoch = current epoch number of training process
    """
    maxEpochs = EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute
    lr = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return lr


## Fetch dataset & preprocessing
print("[INFO] Fetch CIFAR-10 ....")
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# apply mean substraction
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels from ints to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Label Smoothing Way 1 = add label smoothing to one-hot encoded labels!
trainY = trainY.astype("float")
testY = testY.astype("float")
print("[INFO] apply smoothing factor = ", SMOOTHING)
print("[INFO] before smoothing:", trainY[0])
trainY = label_smooth(trainY, factor=SMOOTHING) 
print("[INFO] after smoothing:", trainY[0])


# label names
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog",\
        "horse","ship","truck"]


## prepare model
# initalize data generator & apply data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        )

# build model
if args["model"] is None:
    print("[INFO] compiling model...")
    # exp1
    #model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (16, 16, 32, 64), reg=1e-4)
    # exp2
    #model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (16, 64, 128, 256), reg=1e-4)
    # exp3 & 4
    model = ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=5e-4)

    #opt = Adam(lr=INIT_LR)
    opt = SGD(lr=INIT_LR, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    ## Label Smoothing Way 2 = apply to CategoricalCrossEntropy directly
    #smoothloss = CategoricalCrossentropy(label_smoothing=SMOOTHING)
    #model.compile(loss=smoothloss, optimizer=opt, metrics=["accuracy"])
    
else:
    print("[INFO] loading %s ..." % args["model"])
    model = load_model(args["model"])

    # update learning rate to a smaller one
    print("[INFO] old learning rate =", K.get_value(model.optimizer.lr))
    #K.set_value(model.optimizer.lr, INIT_LR)
    print("[INFO] new learning rate =", K.get_value(model.optimizer.lr))
    
# set up callbacks
FIG_PATH = os.path.sep.join([args["checkpoints"], "resnet56_cifar10.png"])
JSON_PATH = os.path.sep.join([args["checkpoints"], "resnet56_cifar10.json"])

callbacks = [
        EpochCheckpoint(args["checkpoints"], every=5, \
                startAt=args["start_epoch"]), 
        TrainingMonitor(FIG_PATH, jsonPath=JSON_PATH, \
                startAt=args["start_epoch"]),
        LearningRateScheduler(poly_decay),  # Exp4
        ]

# train & evaluate
print("[INFO] training model...")
H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BATCH), 
        validation_data=(testX, testY), 
        steps_per_epoch=len(trainX) // BATCH, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        verbose=1,
        )

print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=BATCH)
print(classification_report(predictions.argmax(axis=1), testY.argmax(axis=1), \
        target_names=labelNames))

print("[INFO] Done!")


