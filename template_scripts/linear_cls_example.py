#!/usr/local/bin/python3.6

## import packages
import os
import sys
import numpy as np
import cv2


# constants
Width = 32
Height = 32
Channel = 3


## initialize class labels & weights & bias
labels = ["dog", "cat", "panda"]
np.random.seed(3)
W = np.random.randn(len(labels), Width * Height * Channel)
b = np.random.randn(len(labels))


## read a picture & compute scores
image_path = "../animal_image_dog_cat_and_panda/dogs/dogs_00005.jpg"
orig = cv2.imread(image_path)
image = cv2.resize(orig, (Width, Height)).flatten() # flatten in to vector

scores = W.dot(image) + b

## print out socresj & show image
for (label, score) in zip(labels, scores):
    print("[INFO] %s : %.2f" % (label, score))

cv2.putText(orig, "Label : %s" % labels[np.argmax(scores)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
cv2.imshow("orignal image", orig)
# display the image all the time until hit some key
cv2.waitKey(0) 


