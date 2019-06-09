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

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model #requires pydot and graphviz

from pose_utils import KerasDataGenerator, SatellitePoseEstimationDataset, Camera, checkFolders

from tqdm import tqdm

## ~~ Settings ~~ ##
#  Change these to match your setup

# dataset_dir = "..\\speed"	  # Root directory of dataset (contains /images, LICENSE.MD, *.json)
dataset_dir = '/data/s1530194/speed'

default_margins = (.2, .2, .1) # (x, y, z) offset between satellite body and rectangular cage used as cropping target

recreate_json = False # Creates a new JSON file with bbox training label even if one already exists

train_dir = os.path.join(dataset_dir, "images/train")

output_loc = 'bbox_jw_6'
#check if the output folder is present and if not, make one
checkFolders([output_loc])

imgsize = (256, 160) #used to be(120, 75)

def create_bbox_json(margins=default_margins):
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

	bb_model.add(Conv2D(filters=32, kernel_size=5,
				padding='valid', use_bias=True))
	bb_model.add(BatchNormalization())
	bb_model.add(Activation('relu'))
	bb_model.add(MaxPooling2D(pool_size=(2,2)))

	bb_model.add(Conv2D(filters=32, kernel_size=5,
				padding='valid', use_bias=True))
	bb_model.add(BatchNormalization())
	bb_model.add(Activation('relu'))
	bb_model.add(MaxPooling2D(pool_size=(2,2)))

	'''
	bb_model.add(Flatten())

	intermediate_sizes = [50, 20]

	for layersize in intermediate_sizes:
		bb_model.add(Dense(units=layersize, activation="relu"))
		bb_model.add(Dropout(0.3))
	'''

	bb_model.add(GlobalAveragePooling2D())

	#now make the output layer
	bb_model.add(Dense(units=4, activation="sigmoid"))

	#also make a flow chart of the model
	plot_model(bb_model, to_file = f'{output_loc}/model_arch.png', show_shapes = True, show_layer_names = True)


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
	#save
	# np.savetxt(f'{self.pose_nn.output_loc}Losses_v{self.pose_nn.version}.txt', np.array([train_loss, test_loss]), header = 'train_loss test_loss')

	#plot
	plt.plot(np.arange(1, len(train_loss)+1), train_loss, label = 'Train loss')
	plt.plot(np.arange(1, len(test_loss)+1), test_loss, label = 'Test loss')

	#set ylims starting at 0
	ylims = plt.ylim()
	plt.ylim((0, ylims[1]))

	plt.yscale('log')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Loss progression of cropping network')
	plt.legend(loc = 'best')
	plt.grid(alpha = 0.4)

	plt.savefig(f'{output_loc}/Cropping_Losses.png', dpi = 300, bbox_inches = 'tight')
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
	plt.savefig(f'{output_loc}/Prediction_{i}.png', bbox_inches = 'tight', dpi = 200)
	plt.close()

def load_single_img(current_label):
	# current_label = random.choice(labels)
	current_filename = current_label["filename"]
	current_bbox = np.array([current_label["bbox"]])

	current_filepath = os.path.join(train_dir, current_filename)
	current_image_pil = Image.open(current_filepath)
	current_image_pil = current_image_pil.resize(imgsize, resample=Image.BICUBIC)
	current_image_arr = np.array(current_image_pil, dtype=float)/256.
	current_image_pil.close()

	return current_bbox, current_image_arr

def train_bb_model(n_images, model, n_val_set = 20):
	labels = create_bbox_json()

	#shuffle labels (so also the images itself)
	np.random.shuffle(labels)

	if n_images + n_val_set > 12000:
		print('WARNING: reducing train set size')
		n_images -= n_val_set

	#load train set
	image_array = []
	label_array = []
	for label in tqdm(labels[:n_images]):
		current_label, current_image_arr = load_single_img(label)
		label_array.append(current_label)
		image_array.append(np.expand_dims(current_image_arr, 2))
	image_array = np.array(image_array)
	label_array = np.array(label_array)[:,0]

	#load validation set which will be used to generate predictions at
	#the end of training
	val_image_array = []
	val_label_array = []
	for label in tqdm(labels[n_images:n_images+n_val_set]):
		current_label, current_image_arr = load_single_img(label)
		val_label_array.append(current_label)
		val_image_array.append(np.expand_dims(current_image_arr, 2))
	val_image_array = np.array(val_image_array)
	val_label_array = np.array(val_label_array)[:,0]

	early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
					patience = 16, verbose = 1,
					restore_best_weights = True, min_delta = 0)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
					factor = 0.5, patience = 4, verbose = 1,
					min_delta = 0, min_lr = 1e-8)

	#data is shuffled in fit
	history = model.fit(image_array, label_array,
				epochs = 100, validation_split = 0.1,
				batch_size = 32, shuffle = True,
				callbacks = [early_stopping, reduce_lr])

	return history, image_array, label_array, val_image_array, val_label_array

def make_model(loadmodel = False):
	if loadmodel:
		print ("Loading architecture from file")
		bb_model = saveLoadModel(f'{output_loc}/model.h5', save = False, load = True)
		print (" Successfully loaded architecture from file!")
	else:
		print ("Creating new model")
		bb_model = create_bb_model()
		print (" Created new model")

	return bb_model


create_bbox_json()
bb_model = make_model()


bb_model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')


n_val_set = 20
history, image_array, label_array, val_image_array, val_label_array = train_bb_model(11000, bb_model, n_val_set = n_val_set)
train_loss = history.history['loss']
test_loss = history.history['val_loss']

print ("Saving model")
saveLoadModel(f'{output_loc}/model.h5', model = bb_model, save = True, load = False)
print (" Saving model successful\n")

#plot loss
plot_save_losses(train_loss, test_loss)

### now make some predictions
pred_lbls = bb_model.predict(val_image_array)

for i in range(n_val_set):
	plot_prediction(val_image_array[i], val_label_array[i], pred_lbls[i], i)
