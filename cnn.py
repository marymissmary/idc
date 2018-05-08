import argparse
import logging

parser = argparse.ArgumentParser(description="Running CNN")
parser.add_argument("-b","--break",dest="tobreak",action="store",help="Set to 'True' for debugging (using subset of data)",default=False)
parser.add_argument("-s","--savedir",dest="save_dir",action="store",help="Directory to save results",required=True)
parser.add_argument("-e","--epochs",dest="epochs",action="store",help="Number of epochs to run",required=True)
parser.add_argument("-m","--model_name",dest="model_name",action="store",help="Name of CNN model to use",required=True)
args=parser.parse_args()

## set up logging
FORMAT = "[%(funcName)10s():%(levelname)s] %(message)s"
logging.basicConfig(level=logging.DEBUG,format=FORMAT,datefmt='%Y-%m-%d %H:%M:%S')
## we want our scripts to be at debug level, but python libraries to be more quiet.
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

known_models = ["mnist","too_simple"]
if args.model_name not in known_models:
	logging.error("Model name not known.")
	logging.error("Choices are: "+",".join(known_models))
	logging.error("Exiting")
	from sys import exit 
	exit(1)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import scipy
from scipy.misc.pilutil import imresize, imread
import itertools
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import json
import utilities as ut
import models_to_test as mtt


logging.info("Running "+args.model_name)
if args.tobreak:
	logging.warning("Only using a small subset of data!")
logging.info("Number of epochs: "+args.epochs)
logging.info("Saving output to: "+args.save_dir)

## Read in XY data
tstart = time.time()
X,Y = ut.get_input(tobreak=args.tobreak)
X = X / 255.
tend = time.time()
logging.info("Time to read in data: {0:.1f} s.".format(tend-tstart))
logging.debug("Current shapes: "+str(X.shape)+", "+str(Y.shape))

## Undersample the IDC(-) data to get even input set:
Xbal,Ybal = ut.balance_pos_neg(X,Y)
logging.info("*** Describe balanced data ***")
ut.describe_data(Xbal,Ybal)

## Shuffle XY data
Xs,Ys = ut.shuffle_input(Xbal,Ybal)
logging.info("*** Describe shuffled data ***")
ut.describe_data(Xs,Ys)

## separate training and validation:
X_train,Y_train,X_val,Y_val = ut.get_train_val(Xs,Ys)
logging.info("*** Describe training data ***")
ut.describe_data(X_train,Y_train)
logging.info("*** Describe validation data ***")
ut.describe_data(X_val,Y_val)

## Create model
tstart = time.time()
Y_train_c = to_categorical(Y_train,num_classes = 2)
Y_val_c   = to_categorical(Y_val,num_classes = 2)
if args.model_name == "mnist":
	model,history = mtt.mnist_cnn(X_train,Y_train_c,X_val,Y_val_c,batch_size=128,epochs=int(args.epochs))
elif args.model_name == "too_simple":
	model,history = mtt.too_simple(X_train,Y_train_c,X_val,Y_val_c,batch_size=128,epochs=int(args.epochs))
tend = time.time()
logging.info("Time to fit model: {0:.1f} s.".format(tend-tstart))

## Save model
ut.save_model(args.save_dir,model,history,X_val,Y_val_c)

logging.info("Finished cnn.py")
