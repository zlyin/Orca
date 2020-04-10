#!/usr/bin/python3.6

## load packages
import os
os.environ["TF_CPP_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("./nn")
sys.path.append("./preprocessing")
sys.path.append("./datasets")
from nn.conv.shallownet import ShallowNet
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=1.0
sess = tf.compat.v1.Session(config=config)
from keras.optimizers import SGD
from keras.models import load_model

import numpy as np
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import cv2


## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to input datasetdataset")
parser.add_argument("-m", "--model", required=True, help="name of output model")
parser.add_argument("-o", "--output_dir", required=True, help="name of output model")
#parser.add_argument("-e", "--epochs", type=int, default=40, help="path to input datasetdataset")
args = vars(parser.parse_args())

dataset_name = args["dataset"].split("/")[1]
classLabels = ["cat", "dog", "panda"]


## load sampled images
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
imgIndex = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[imgIndex]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
data, labels = sdl.load(imagePaths, verbose=1000)
data = data.astype("float") / 255.0


## reload the pre-trained NN
print("[INFO] loading pre-trained NN...")
model_file = args["model"] + "_on_%s.hdf5" % dataset_name
output_model_path = os.path.join(args["output_dir"], model_file)
model = load_model(output_model_path)

print("[INFO] predicting...")
predictions = model.predict(data, batch_size=32)
preds = predictions.argmax(axis=1)


## display images & preds
fig = plt.figure(figsize=(8, 6))
axs = []

for idx, imgpath in enumerate(imagePaths):
    image = cv2.imread(imgpath)
    cv2.putText(image, "Label: %s" % classLabels[preds[idx]], (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2) 

    axs.append(fig.add_subplot(2, 5, idx + 1))
    plt.imshow(image)
    pass
plt.show()





