#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import argparse
import h5py
import pickle   # serialization tool


## construct arguments
parser = argparse.ArgumentParser()
parser.add_argument("-db", "--database", required=True, help="path to HDF5 database")
parser.add_argument("-o", "--output", required=True, help="path to output model")

parser.add_argument("-j", "--jobs", type=int, default=-1, \
        help="# of jobs to run when tuning hyperparameters")
args = vars(parser.parse_args())


"""
Open a HDF5 dataset and split the training & test dataset by index;
Noting to shuffle samples BEFORE creating the dataset!
Data before thres is training set; while the rest data is test set;
"""
db = h5py.File(args["database"], "r")
thres = int(db["labels"].shape[0] * 0.75)


"""
Fine tuning hyperparameter C for logistic regression;
Initiate GridSearchCV as model, feed in a classifier object;
Train the model on training set
"""
print("[INFO] tuning hyperparameters...")
params = {"C" : [0.1, 1.0, 10.0, 100.0, 1000.0]}

model = GridSearchCV(
        LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=500), \
        params, \
        scoring="accuracy", \
        cv=3, \
        n_jobs=args["jobs"])
model.fit(db["features"][:thres], db["labels"][:thres])

print("[INFO] best hyperparameters are {}".format(model.best_params_))


## evaluate the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][thres:])   # dim=1! no need to argmax
print(classification_report(preds, db["labels"][thres:], \
        target_names=db["label_names"]))


## serialize the model to disk
if "cpickle" not in args["output"]:
    raise ValueError("Output file format must be '.cpickle' format!")

print("[INFO] serializing best estimator to disk...")
with open(args["output"], "wb") as f:       # need to be "xxx.cpickle" file!
    f.write(pickle.dumps(model.best_estimator_))
f.close()

# close db
db.close()



