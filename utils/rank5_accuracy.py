## import packages
import os
import sys
import numpy as np


## define function
def rank5_accuracy(probs, labels):
    """
    - compute rank1 & rank5 accuracies based on predicted probs & gt labels;
    - probs = N * T matrix for probs, N = number of samples, T = number of classes;
    - labels = int label encoded from LabelEncoder();
    """
    # initiate correct prediction counters
    rank1, rank5 = 0, 0

    for prob, gt in zip(probs, labels):
        # sort probs in reverse order
        pidx = np.argsort(prob)[::-1]

        # count rank5 
        if gt in pidx[:5]:
            rank5 += 1

        # count rank1
        if gt == pidx[0]:
            rank1 += 1

    # return accuracy
    rank1_acc = rank1 / float(len(labels))
    rank5_acc = rank5 / float(len(labels))
    return (rank1_acc, rank5_acc)
