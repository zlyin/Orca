## import packages
import os
import sys

import imutils
import cv2
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt



## def a captchar process method for numbers_only
def preprocess(image, height, width):
    """
    2 stage preprocessing to maintain AR:
        resize the larger dim to target H & W;
        padding smaler dim to target H = W & resize to kill rounded number; 
    """
    h, w = image.shape[:2]
    
    # maintain the AR first
    if h > w:
        image = imutils.resize(image, height=height)
    else:
        image = imutils.resize(image, width=width)

    padH = int((height - image.shape[0]) / 2.0)
    padW = int((width - image.shape[1]) / 2.0)

    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, \
            cv2.BORDER_REPLICATE) 

    # avoid manipulating math, just resize one more time
    image = cv2.resize(image, (width, height))  # cv2.resize(img, (w, h))
    
    return image


## def a denoise method for numbers_only_with_noise 
def denoise(image):
    # tutorial = https://cagriuysal.github.io/Simple-Captcha-Breaker/
    
    # find noise pattern type
    #pattern = whichPattern(image, plotHist=True)
    pattern = whichPattern(image)
       
    """
    for pattern1 (stochastic black dots):
        takes median of kernel to replace central element in the kernel;
        highly effective against salt-and-pepper noise in the images;
        kernel size MUST be odd number!

    for pattern2 (regular pattern):
        use GaussianBlur; 
        highly effective in removing gaussian noise from the image;
    """
    if pattern == "1":
        image = cv2.medianBlur(image, 3)    # 5x5 is too much   
    else:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        #image = cv2.bilateralFilter(image, 5, 75, 75)
    
    # test several thresholds
    pattern1_thres = 170
    pattern2_thres = 155 

    thres = pattern1_thres if pattern == "1" else pattern2_thres

    # noting that use cv2.THRES_BINARY_IpathpathNV to decide foreground & background;
    #retval, thres_img = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY)
    retval, thres_img = cv2.threshold(image, thres, 255, cv2.THRESH_BINARY_INV)

    """
    pattern1 threshold image still has discrete black dots;
    labels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity):
        where stats = [x, y, w, h, area]
    """
    min_connected_dim = 15
    numCC, labelImage, stats, centroids = cv2.connectedComponentsWithStats(
            thres_img, connectivity=8)

    # loop over CCs & retrieve foreground/digits with size > min_connected_dim
    sizes = stats[:, -1]
    foreCompts = [i for i in range(1, numCC) if sizes[i] >= min_connected_dim] 

    clean_thres_img = np.zeros(thres_img.shape, dtype="uint8")
    for k in foreCompts:
        clean_thres_img[labelImage == k] = 255
   
    # extract ROI of all digits
    Xs, Ys, Ws, Hs = [], [], [], []
    for k in foreCompts:
        Xs.append(stats[k, 0])
        Ys.append(stats[k, 1])
        Ws.append(stats[k, 2])
        Hs.append(stats[k, 3])

    # observed that for images, minCol == 30; calculate biggest bounding box
    minCol = 30 
    minRow = min(Ys)
    
    Xbox = np.array(Xs) + np.array(Ws)   
    maxCol = Xbox.max(axis=0)

    Ybox = np.array(Ys) + np.array(Hs)
    maxRow = Ybox.max(axis=0)

    clean_thres_img = clean_thres_img[minRow -5 : maxRow + 5, minCol -5 : maxCol + 5]

    return clean_thres_img


def whichPattern(image, plotHist=False):
    """
    use image-histograms to stats pixel intensities on GRAYSCALE images;
    cv2.calcHist([images], [channels], mask, [histSize], ranges)
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]) # shape=(256, 1)
    
    if plotHist:
        plt.figure()
        plt.plot(hist)
        plt.title("image histogram - pixel intensity")
        plt.xlabel("pixel intensity")
        plt.ylabel("counts")
        plt.xlim([0, 256])
        plt.show()
        pass
 
    # if pattern 1, black pints = hist[0][0] generally > 600 
    threshold = 500
    if hist[0][0] > threshold:
        return "1"
    else:
        return "2"


def splitDigits(image, path):
    """
    split digits in each noised image into seperate digits
    """
    H, W = image.shape
    avgCol = int(W / 6)
    digits = []
    for i in range(6):
        digits.append(image[:, i * avgCol : (i + 1) * avgCol])  
    digits = np.array(digits, dtype="uint8")
        
    newDigits = [] 
    
    for i in range(0, 6): 
        """
        a lot of digits are broken, looking like missing pieces on white foreground
            digits;
        fix that cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        closedDigit = cv2.morphologyEx(digits[i], cv2.MORPH_CLOSE, kernel, 1)
        numCC, labelDigit, stats, centroids = cv2.connectedComponentsWithStats(\
                closedDigit, 8, cv2.CV_32S)
        
        """
        sort ndarray stats in order of [x, y, -area, width, height], 
        priority decrease from Right -> Left; return row_idx order
        """ 
        index = np.lexsort([stats[:, 3],stats[:, 2], -stats[:, 4], stats[:, 1],stats[:,0]])
        stats = stats[index, :]
        stats = stats[1:, :]     # row0 = background

        # skip if only background
        if len(stats) == 0:
            continue

        # record main digit & shift-left digits information 
        mainDigit = np.zeros(digits[i].shape) 
        shiftLeft = np.zeros(digits[i].shape) 
        shiftStats = []

        # find One major digit
        majorstats = stats[np.where(stats == np.amax(stats))[0], :] # 2D array
        [x, y, w, h, area] = majorstats[0]
        mainDigit[y : y + h, x : x + w] = digits[i][y : y + h, x : x + w]

        if len(stats) == 1:
            continue

        # deal with shift_left pieces   
        for j, ccstat in enumerate(stats):

            # skip the major digit
            #if ccstat in majorstats:
            #    continue

            [x, y, w, h, area] = ccstat
            
            # shift left piece must start from [:, 0]  
            if x == 0:
                shiftLeft[y : y + h, x : x + w] = digits[i][y : y + h, x : x +
                        w]
                shiftStats.append(ccstat)
            if x == majorstats[0][0]:
                mainDigit[y : y + h, x : x + w] = digits[i][y : y + h, x : x + w]

        # drop major digit cc
        if i != 0 and shiftLeft.any():
            fixDigit = merge(digits[i - 1], shiftLeft, shiftStats)
            digits[i - 1] = fixDigit
        digits[i] = mainDigit

    return digits


def merge(main, piece, pstats, mode="copy"):
    """
    add piece to main;
    two modes, copy => to a new drawing; superpose ==> add in-place
    """
    # check shape
    H, W = main.shape
    merge = np.zeros(main.shape)

    if main.shape != piece.shape:
        piece = cv2.resize(piece, main.shape)
    for x, y, w, h, area in pstats:
        merge[:, -w:] = piece[:, x : x + w]
        # equivalent shift main to left by w pixels
        merge[:, :W - w] = main[:, w:]
    return merge


def plot_image(ll):
    plots = []
    for i in range(0, len(ll)):
        plots.append(ll[i])
        plots.append(np.ones(ll[i].shape) * 255)
        drawing = np.concatenate(plots, axis=1)
    return drawing

