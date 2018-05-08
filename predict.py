#import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import logging
## from p1.py at TACC:
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
import json
import utilities as ut

FORMAT = "[%(funcName)10s():%(levelname)s] %(message)s"
#FORMAT = "[%(asctime)s %(filename)20s:%(lineno)4s:%(funcName)20s():%(levelname)s] %(message)s"
logging.basicConfig(level=logging.DEBUG,format=FORMAT,datefmt='%Y-%m-%d %H:%M:%S')

## we want our scripts to be at debug level, but python libraries to be more quiet.
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

OutDir = "Output"

## load the model
#model = keras.models.load_model(OutDir+"/model.npy")
#logging.info(model.summary())

## load the validation data
xval = np.load(OutDir+"/xval.npy")
yval = np.load(OutDir+"/yval.npy")

#y_pred = model.predict(xval)


## read history
with open(OutDir+"/history.npy","r") as fg:
	history = json.loads(fg.readline())

ut.plot_learning_curves(history,OutDir)

### plot learning curves from history
#plt.figure(200)
#cs = iter(["b","r","g","k"])  #<-- iterate over line colors
#for quantity,display_name in zip(["acc","val_acc","loss","val_loss"],["Training Accuracy","Validation Accuracy","Training Loss","Validation Loss"]):
#	logging.debug("display_name: "+display_name)
	#c = next(cs)
#	l = np.array(history[quantity])
#	plt.plot(l,lw=2,c=c,label=display_name)
#plt.legend(loc="center right")
#plt.grid()
#plt.xlabel("Epoch number")
#plt.savefig("history_learning_curve.png")
	




