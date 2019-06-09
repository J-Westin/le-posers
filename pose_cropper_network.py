import os
import json
import random

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

import keras

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model #requires pydot and graphviz

from pose_utils import KerasDataGenerator, SatellitePoseEstimationDataset, Camera, checkFolders

from tqdm import tqdm


def create_bbox_json(margins = (.2, .2, .1)):
	# margins: (x, y, z) offset between satellite body and rectangular cage used as cropping target

	bbox_json_filepath_expected = os.path.join(dataset_dir, "train_bbox.json")

	if recreate_json or (not os.path.exists(bbox_json_filepath_expected)):
		my_dataset = SatellitePoseEstimationDataset(root_dir=dataset_dir)
		my_dataset.generate_bbox_json(margins)

	bbox_json_file = open(bbox_json_filepath_expected, 'r')
	bbox_training_labels = json.load(bbox_json_file)
	bbox_json_file.close()

	return bbox_training_labels

def create_bb_model():
	bb_model = Sequential()

	bb_model.add(Conv2D(filters=32, kernel_size=7,
					input_shape=(imgsize[1], imgsize[0], 1),
					padding='valid', use_bias=True))
	bb_model.add(BatchNormalization())
	bb_model.add(Activation('relu'))
	bb_model.add(MaxPooling2D(pool_size=(2,2)))

	bb_model.add(Conv2D(filters=32, kernel_size=5,
				padding='valid', use_bias=True))
	bb_model.add(BatchNormalization())
	bb_model.add(Activation('relu'))
	bb_model.add(MaxPooling2D(pool_size=(2,2)))

	bb_model.add(Conv2D(filters=32, kernel_size=3,
				padding='valid', use_bias=True))
	bb_model.add(BatchNormalization())
	bb_model.add(Activation('relu'))
	bb_model.add(MaxPooling2D(pool_size=(2,2)))

	bb_model.add(Conv2D(filters=32, kernel_size=3,
				padding='valid', use_bias=True))
	bb_model.add(BatchNormalization())
	bb_model.add(Activation('relu'))
	bb_model.add(MaxPooling2D(pool_size=(2,2)))

	bb_model.add(Conv2D(filters=16, kernel_size=3,
				padding='valid', use_bias=True))
	bb_model.add(BatchNormalization())
	bb_model.add(Activation('relu'))
	bb_model.add(MaxPooling2D(pool_size=(2,2)))

	#we need to use flattening instead of maxpooling, as this preserves
	#the location info
	# bb_model.add(GlobalMaxPooling2D())

	bb_model.add(Flatten())

	bb_model.add(Dense(units = 30, activation = 'relu'))
	bb_model.add(Dropout(0.3))

	#now make the output layer
	bb_model.add(Dense(units = 4, activation = 'sigmoid'))


	#### save the architecture info
	# model summary
	def myprint(s):
	    with open(f'{output_loc}/croppermodel_v{version}_c{target_label}_summary.txt','a') as f:
	        print(s, file=f)

	bb_model.summary(print_fn = myprint)


	#also make a flow chart of the model
	plot_model(bb_model, to_file = f'{output_loc}/croppermodel_v{version}_c{target_label}_arch.png', show_shapes = True, show_layer_names = True)


	return bb_model

def saveLoadModel(filename, model=None, save=False, load=False):
	"""
	Load or save a model easily given a path+filename. Filenames must have the format 'model.h5'.
	This saves/load model architecture, weights, loss function, optimizer, and optimizer state.

	Returns:
	model if load=True, else just saves the input model
	"""
	if save:
		print('Saving model to {}'.format(filename))
		model.save(filename)
	if load:
		if not os.path.exists(filename):
			print('Cannot find specified model, check if path or filename is correct')
			return
		print('Loading model from {0}'.format(filename))
		model = keras.models.load_model(filename)

		return model

def plot_save_losses(train_loss, test_loss):
	"""
	Save and plot the losses
	"""
	#save the losses
	np.savetxt(f'{output_loc}/Losses_v{version}_c{target_label}.txt', np.array([train_loss, test_loss]), header = 'train_loss test_loss')

	#plot
	plt.plot(np.arange(1, len(train_loss)+1), train_loss, label = 'Train loss')
	plt.plot(np.arange(1, len(test_loss)+1), test_loss, label = 'Test loss')

	#set ylims starting at 0
	ylims = plt.ylim()
	plt.ylim((0, ylims[1]))

	# plt.yscale('log')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Loss progression of cropping network with cluster label {target_label}')
	plt.legend(loc = 'best')
	plt.grid(alpha = 0.4)

	plt.savefig(f'{output_loc}/Cropping_losses_c{target_label}.png', dpi = 300, bbox_inches = 'tight')
	plt.close()

def plot_prediction(image, true_label, pred_label, i):
	def plot_bounding_box(ax, label, img_width, img_height, name):
		x = label[0]*img_width
		y = label[2]*img_height
		width = (label[1] - label[0])*img_width
		height = (label[3] - label[2])*img_height

		if name == 'Ground truth':
			colour = 'blue'
		else:
			colour = 'red'

		rect = patches.Rectangle((x, y), width, height,
						facecolor = 'none', edgecolor = colour,
						linewidth = 1.5, label = name)
		# Add the patch to the Axes
		ax.add_patch(rect)

	#use only the first channel (there is generally only one channel available)
	image = image[:,:,0]

	fig, ax = plt.subplots()

	plt.imshow(image, cmap = 'gray')

	img_height, img_width = image.shape
	#plot ground truth
	plot_bounding_box(ax, true_label, img_width, img_height, 'Ground truth')

	#plot prediction
	plot_bounding_box(ax, pred_label, img_width, img_height, 'Prediction')


	plt.legend(loc = 'upper right')
	# plt.axis('off')
	plt.savefig(f'{output_loc}/Prediction_c{target_label}_{i}.png', bbox_inches = 'tight', dpi = 200)
	plt.close()

def load_single_img(current_label):
	# current_label = random.choice(labels)
	current_filename = current_label["filename"]
	current_bbox = np.array(current_label["bbox"])

	current_filepath = os.path.join(train_dir, current_filename)
	current_image_pil = Image.open(current_filepath)
	current_image_pil = current_image_pil.resize(imgsize, resample=Image.BICUBIC)
	current_image_arr = np.array(current_image_pil, dtype=float)/256.
	current_image_pil.close()

	return current_bbox, current_image_arr

def load_and_select_labels(target_label):
	"""
	Load the image labels and select them based on their assigned
	cluster label and the target cluster label we are training for
	"""

	#load all the labels
	labels = create_bbox_json()

	#extract the filenames without the preceding path (only 'train' or
	# 'test')
	fnames = []
	for lab in labels:
		fnames.append(f'{settype}/{lab["filename"]}')

	#load the filenames and corresponding cluster labels
	data = np.load(clusterlabels_loc)
	cluster_fnames = data[:,0]
	cluster_labs = data[:,1]

	#remove the preceding path up to 'train' and 'test'
	cluster_fnames = np.core.defchararray.replace(cluster_fnames, '/data/s1530194/speed/images/', '')

	#make this into a dictionary
	cluster_labels = dict(zip(cluster_fnames, cluster_labs))
	del data, cluster_fnames, cluster_labs

	#now select labels of images which have a cluster label equal
	#to the target cluster label
	sel_labels = []
	for i in tqdm(range(len(fnames)), desc = 'Cluster based selection'):
		assigned_label = int(cluster_labels[fnames[i]])

		if assigned_label == target_label:
			sel_labels.append(labels[i])

	#shuffle labels (so also the images itself)
	np.random.shuffle(sel_labels)

	return sel_labels

def train_bb_model(use_frac, model, target_label, n_val_set = 20):
	"""
	Load the data and train the model
	"""
	#load and select labels based on the target cluster label
	labels = load_and_select_labels(target_label)

	#select the train & test set and the validation set
	labels_traintest = labels[:-n_val_set]
	labels_val = labels[-n_val_set:]

	#use only a fraction of the train & test labels
	if use_frac < 1:
		n_use = int(len(labels_traintest) * use_frac)
		labels_traintest = labels_traintest[:n_use]

	print(f'{len(labels_traintest)} images used for training/testing ({use_frac*100:0.01f}%) with cluster label {target_label}')

	#load train set
	image_array = []
	label_array = []
	print('Loading train & test set')
	for label in tqdm(labels_traintest):
		current_label, current_image_arr = load_single_img(label)
		label_array.append(current_label)
		image_array.append(np.expand_dims(current_image_arr, 2))

	image_array = np.array(image_array)
	label_array = np.array(label_array)

	#load validation set which will be used to generate predictions at
	#the end of training
	val_image_array = []
	val_label_array = []
	print('Loading validation set')
	for label in tqdm(labels_val):
		current_label, current_image_arr = load_single_img(label)
		val_label_array.append(current_label)
		val_image_array.append(np.expand_dims(current_image_arr, 2))

	val_image_array = np.array(val_image_array)
	val_label_array = np.array(val_label_array)

	#note: the early stopping restores the weights of the network with
	#the lowest test loss
	early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
					patience = 9, verbose = 1,
					restore_best_weights = True, min_delta = 0)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
					factor = 0.25, patience = 4, verbose = 1,
					min_delta = 0, min_lr = 1e-8)

	#data is shuffled in fit
	history = model.fit(image_array, label_array,
				epochs = 50, validation_split = 0.1,
				batch_size = 32, shuffle = True,
				callbacks = [early_stopping, reduce_lr])

	return history, image_array, label_array, val_image_array, val_label_array

def make_model(loadmodel = False):
	if not loadmodel:
		print ("Creating new model")
		bb_model = create_bb_model()
		print (" Created new model")
	else:
		print ("Loading architecture from file")
		bb_model = saveLoadModel(f'{output_loc}/{modelname}', save = False, load = True)
		print ("Successfully loaded architecture from file!")

	return bb_model

#### Fix some parameters
#which cluster label to train the network for. Possible: [0, 1, 2]
target_label = 2
#number of validation images to use for plotting examples
n_val_set = 20
#size of the input images for the network
imgsize = (512, 320) #(256, 160)
#version of the network
version = 7
#fraction of available images to use for training. We cannot define a
#number of images as this depends on the cluster label
use_frac = 1

#### Fixed parameters
#location of the dataset
dataset_dir = '/data/s1530194/speed'
# Creates a new JSON file with bbox training label even if one already exists
recreate_json = False
#where to output the results
output_loc = f'cropping_network_development/bbox_jw_{version}'
#check if the output folder is present and if not, make one
checkFolders([output_loc])
#location where the cluster labels are saved
clusterlabels_loc = 'clustering_labels/hierarch_cluster_labels.npy'

#name for the model
modelname = f'croppermodel{version}_c{target_label}.h5'

#init
create_bbox_json()
settype = 'train'
train_dir = os.path.join(dataset_dir, f'images/{settype}')


#### Make and compile a model, load data and train
bb_model = make_model(loadmodel = False) #can also instead load a model
bb_model.compile(optimizer = keras.optimizers.Adam(lr = 0.001), loss = 'mean_absolute_error') #mse

history, image_array, label_array, val_image_array, val_label_array = train_bb_model(use_frac, bb_model, target_label, n_val_set = n_val_set)
train_loss = history.history['loss']
test_loss = history.history['val_loss']


#### Save the model
print ('Saving model')
saveLoadModel(f'{output_loc}/{modelname}', model = bb_model, save = True, load = False)
print ('Saving model successful\n')


#### After training processing
#plot loss
plot_save_losses(train_loss, test_loss)

#make some predictions
pred_lbls = bb_model.predict(val_image_array)

for i in tqdm(range(n_val_set), desc = 'Saving val predictions'):
	plot_prediction(val_image_array[i], val_label_array[i], pred_lbls[i], i)
