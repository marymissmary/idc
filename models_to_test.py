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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import json
import utilities as ut

def too_simple(x_train,y_train,x_test,y_test,batch_size=128,epochs=10):
	"""
	comment out some layers...trying to create something that has a high bias.
	"""
	num_classes = 2
	img_rows, img_cols = x_train.shape[1],x_train.shape[2]
	input_shape = (img_rows, img_cols, 3)
	logging.debug("Input data shape: "+str(input_shape))
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
	#model.add(Conv2D(64, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))
	model.add(Flatten())
	#model.add(Dense(128, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	logging.info(str(model.summary()))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
	history = model.fit(x_train, y_train,batch_size=batch_size,verbose=1,epochs=epochs,validation_data=(x_test, y_test),callbacks = [ut.MetricsCheckpoint('logs')])
	return model,history

def mnist_cnn(x_train,y_train,x_test,y_test,batch_size=128,epochs=10):
	"""
	https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
	"""
	num_classes = 2
	img_rows, img_cols = x_train.shape[1],x_train.shape[2]
	input_shape = (img_rows, img_cols, 3)
	logging.debug("Input data shape: "+str(input_shape))
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	logging.info(str(model.summary()))
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
	history = model.fit(x_train, y_train,batch_size=batch_size,verbose=1,epochs=epochs,validation_data=(x_test, y_test),callbacks = [ut.MetricsCheckpoint('logs')])
	return model,history

