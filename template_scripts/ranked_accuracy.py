#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

import sys
sys.path.append("./utils/")
from ranked import rank5_accuracy

import argparse
import h5py
import pickle


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-db", "--database", required=True, help="path to HDF5 database")
parser.add_argument("-m", "--model", required=True, help="path to pretrained model")
args = vars(parser.parse_args())


## load pre-trained model
print("[INFO] loading pre-trained model...")
with open(args["model"], "rb") as f:
    model = pickle.loads(f.read())
f.close()

## open HDF5 file to retrieve features
print("[INFO] loading extracted features...")
db = h5py.File(args["database"], "r")
index = int(db["labels"].shape[0] * 0.75)


"""
use pre-trained model to predict Probs!
output rank1 & rank5 accuracy;
"""
print("[INFO] predicting...")
preds = model.predict_proba(db["features"][index:])
rank1_acc, rank5_acc = rank5_accuracy(preds, db["labels"][index:])
print("[INFO] rank-1 acc = {:.2f}%".format(rank1_acc * 100))
print("[INFO] rank-5 acc = {:.2f}%".format(rank5_acc * 100))

## close database!
db.close()
