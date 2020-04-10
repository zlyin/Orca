#!/usr/local/bin/python3.6

## import package
import os
import sys

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

## define convolve function
def convolve(image, K):
    # grab spatial dimension of image & kernel
    iH, iW = image.shape[:2]
    kH, kW = K.shape[:2]

    # calculate pad & pad image
    pad = (kH - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float") # convert to float to compute convolution
    
    # loop over pixels, do convolution
    for x in range(pad, iH + pad):
        for y in range(pad, iW + pad):
            # get ROI
            roi = image[x - pad:x + pad + 1, y - pad:y + pad + 1]
            # apply convolution
            k = (roi * K).sum() # get scalar
            # store k in output
            output[x - pad, y - pad] = k
        pass

    # rescale output after convolution to [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    # convert to int
    output = (output * 255).astype("uint8") # pixels = ints
    
    return output


## construct argparser & parse the argument
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(parser.parse_args())


## define a variety of kernels
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))    # because convolution will do sum!
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
    ], dtype="int")

# detech edge-like region
laplacian = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
    ], dtype="int")

# detect vertical edge
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
    ], dtype="int")
# detect horizontal edge
sobelY = np.array([
    [-1, -2, 1],
    [0, 0, 0],
    [1, 2, 1],
    ], dtype="int")

emboss = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
    ], dtype="int")

## lump all kernels in to a kernel bank
kernelBank = [
            ("small_blur", smallBlur),
            ("large_blur", largeBlur),
            ("sharpen", sharpen),
            ("laplacian", laplacian),
            ("sobel_x", sobelX),
            ("sobelY", sobelY),
            ("emboss", emboss),
            ]


fig = plt.figure()
axs = []
## loadin image and convert to grayscale
image = cv2.imread(args["image"])   # CV2 use BGR!
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

axs.append(fig.add_subplot(2, 4, 1))
plt.imshow(gray)
axs[-1].set_title("original")

# loop over kernels & apply convolution
for i, (kernelname, K) in enumerate(kernelBank):
    print("[INFO] applying %s kernel" % kernelname)
    
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)
   
    axs.append(fig.add_subplot(2, 4, i + 2))
    plt.imshow(convolveOutput)
    axs[-1].set_title(kernelname)

    # sanity check
    #cv2.imshow("Orginal", gray)
    #cv2.imshow("%s - convolve" % kernelname, convolveOutput)
    #cv2.imshow("%s - cv2" % kernelname, opencvOutput)
    #cv2.waitKey(0)
    pass

plt.show() 










