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

def plot_learning_curves(history,OutDir,figID=200):
	## plot learning curves from history
	logging.debug("Plotting learning curves")
	plt.figure(figID)
	cs = iter(["b","r","g","k"])  #<-- iterate over line colors
	for quantity,display_name in zip(["acc","val_acc","loss","val_loss"],["Training Accuracy","Validation Accuracy","Training Loss","Validation Loss"]):
		c = next(cs)
		xvals = np.arange(1,len(history[quantity])+1)
		l = np.array(history[quantity])
		plt.plot(xvals,l,lw=2,c=c,label=display_name)
	plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2)
	plt.grid()
	plt.xlabel("Epoch number")
	plt.show()
	plt.savefig(OutDir+"/learning_curve.png")
	return

def plot_confusion_matrix(x_test, y_test, model,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues,outfile='confusion_matrix.png'):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	## use the model to predict values:
	score = model.evaluate(x_test, y_test, verbose=0)
	print('\nKeras CNN #1A - accuracy:', score[1],'\n')
	y_pred = model.predict(x_test)
	map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
	print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')
	y_pred_classes = np.argmax(y_pred,axis=1)
	y_true         = np.argmax(y_test,axis=1)
	cm  = confusion_matrix(y_true,y_pred_classes)
	classes = list(map_characters.values())
	plt.figure(figsize = (7,7))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=90)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
	plt.savefig(outfile)

class MetricsCheckpoint(Callback):
	"""Special Keras fit callback that saves metrics after each epoch"""
	def __init__(self, savepath):
		super(MetricsCheckpoint, self).__init__()
		self.savepath = savepath
		self.history = {}
	def on_epoch_end(self, epoch, logs=None):
		for k, v in logs.items():
				self.history.setdefault(k, []).append(v)
		np.save(self.savepath, self.history)

def get_input(tobreak=False):
	"""Read in the input data. For debugging, set 'tobreak' to True"""
	n_images = [0,0]
	X = np.empty((1,50,50,3),np.uint8)
	Y = np.empty((1),np.uint8)
	for pos_or_neg in [1,0]:	#<-- 0 = IDC(-), 1 = IDC(+)
		if pos_or_neg == 0: 
			logging.info("Getting input data for IDC(-)")
		else:
			logging.info("Getting input data for IDC(+)")
		input_path = "ProcessedData/" + str(pos_or_neg) + "/"
		files_list = os.listdir(input_path)
		just_x_files = [x for x in files_list if "X" in x] #<-- just the 'X_*_*.npy' files
		for f in just_x_files:
			xfile = input_path + f
			yfile = input_path + f.replace("X","Y")
			logging.debug("Reading from files:")
			logging.debug("  "+xfile)
			logging.debug("  "+yfile)
			Xtmp = np.load(xfile)
			Ytmp = np.load(yfile)
			n_images[pos_or_neg] = n_images[pos_or_neg] + len(Ytmp)
			X = np.append(X,Xtmp,axis=0)
			Y = np.append(Y,Ytmp,axis=0)
			logging.debug("				Number of images in "+f+": "+str(len(Ytmp)))
			if tobreak:
				if n_images[1] > 500:
					logging.warning("Breaking early")
					break
			#break
	## remove the empty initial image:
	X = np.delete(X,0,axis=0)
	Y = np.delete(Y,0,axis=0)
	logging.info("Number of IDC(-) images: " + str(n_images[0]))
	logging.info("Number of IDC(+) images: " + str(n_images[1]))
	return X,Y

def plot_one(fignum,Xp,Yp):
	"""Plot one of the images. fignum = frame of input array to plot"""
	plt.figure(fignum)
	plt.imshow(Xp[fignum-1])	#,interpolation='none')
	logging.debug("Frame "+str(fignum)+" is "+str(Yp[fignum-1])+" (0 => IDC(-), 1 => IDC(+))")
	if Yp[fignum-1] == 0:
		title = "IDC(-), Frame: "+str(fignum)
	else:
		title = "IDC(+), Frame: "+str(fignum)
	plt.title(title)
	plt.show(block=False)

def shuffle_input(X,Y):
	"""shuffle inputs"""
	logging.info("Shuffling X/Y data.")
	c = np.c_[X.reshape(len(X),-1), Y.reshape(len(Y), -1)]
	X2 = c[:, :X.size//len(X)].reshape(X.shape)
	Y2 = c[:, X.size//len(X):].reshape(Y.shape)
	np.random.shuffle(c)
	return X2,Y2

def balance_pos_neg(X,Y):
	"""Get training and validation data"""
	## need to account for the fact that we have many more IDC(-) than IDC(+)
	logging.info("Undersampling IDC(-) data")
	X_shape = X.shape[1]*X.shape[2]*X.shape[3]
	X_flat  = X.reshape(X.shape[0],X_shape)
	ros = RandomUnderSampler(ratio='auto')
	X_ros,Y_ros = ros.fit_sample(X_flat,Y)
	X_ros_reshap = X_ros.reshape(len(X_ros),50,50,3)  #<--hard coded that 50x50x3
	return X_ros_reshap,Y_ros

def get_train_val(X,Y):
	"""Get 80% of data to use as training, 20% for validation"""
	n_train = int(len(Y)*.8)
	X_train = X[:n_train]
	X_val   = X[n_train:]
	Y_train = Y[:n_train]
	Y_val   = Y[n_train:]
	return X_train,Y_train,X_val,Y_val

def describe_data(X,Y):
	"""print info about data"""
	logging.info("Total number of Images:  {:>7}".format(len(Y)))
	logging.info("Number of IDC(-) Images: {:>7}".format(np.sum(Y==0)))
	logging.info("Number of IDC(+) Images: {:>7}".format(np.sum(Y==1)))
	logging.info("Percantage of positive images: {:.2f}%".format(100*np.mean(Y)))

def plotKerasLearningCurve():
	plt.figure(figsize=(10,5))
	metrics = np.load('logs.npy')[()]
	filt = ['acc','loss'] # try to add 'loss' to see the loss learning curve
	xkcd_c = iter(["b","r","g","k"])  #<-- iterate over line colors
	for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
		logging.debug("Checking: "+k)
		l = np.array(metrics[k])
		c = next(xkcd_c)
		logging.debug("Color: "+c)
		plt.plot(l, lw=2, c=c, label=k)
		x = np.argmin(l) if 'loss' in k else np.argmax(l)
		y = l[x]
		plt.scatter(x,y, lw=0, alpha=0.25, s=100, c=c)
		plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color=c)
	plt.legend(loc="center right")
	plt.axis([0, None, None, None]);
	plt.grid()
	plt.xlabel('Number of epochs')
	plt.savefig("learning.png")
	plt.close('all')

def save_model(base_dir,model,history,X_val,Y_val_c):
	"""Save results"""
	model_file = base_dir+"/model.npy"
	hist_file  = base_dir+"/history.npy"
	xval_file  = base_dir+"/xval.npy"
	yval_file  = base_dir+"/yval.npy"
	model.save(model_file)
	logging.info("Saved model to: " + model_file)
	np.save(xval_file,X_val)
	logging.info("Saved xval data to: " + xval_file)
	np.save(yval_file,Y_val_c)
	logging.info("Saved yval data to: " + yval_file)
	with open(hist_file,"w") as fg:
		fg.write(json.dumps(history.history))
	logging.info("Saved history data to: " + hist_file)
