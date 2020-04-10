## import packages
import os
import sys
import numpy as np
import tf
from tensorflow.keras.losses import CategoricalCrossentropy

"""
- Label smoothing function
- Params:
    - labels, one-hot encoded labels
    - factor, smoothing factor, default set is 10%
"""
def label_smooth(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels
