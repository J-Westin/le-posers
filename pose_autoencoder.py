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

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input, UpSampling2D, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model #requires pydot and graphviz

from pose_utils import KerasDataGenerator, SatellitePoseEstimationDataset, Camera, checkFolders

from tqdm import tqdm

## ~~ Settings ~~ ##
#  Change these to match your setup

# dataset_dir = "..\\speed"	  # Root directory of dataset (contains /images, LICENSE.MD, *.json)
dataset_dir = '/data/s1530194/speed'

recreate_json = False # Creates a new JSON file with bbox training label even if one already exists
default_margins = (.2, .2, .1) # (x, y, z) offset between satellite body and rectangular cage used as cropping target

train_dir = os.path.join(dataset_dir, "images/train")

imgsize = (256, 160) #used to be(120, 75)

def create_json(margins=default_margins):
	bbox_json_filepath_expected = os.path.join(dataset_dir, "train_bbox.json")

	if recreate_json or (not os.path.exists(bbox_json_filepath_expected)):
		my_dataset = SatellitePoseEstimationDataset(root_dir=dataset_dir)
		my_dataset.generate_bbox_json(margins)

	bbox_json_file = open(bbox_json_filepath_expected, 'r')
	bbox_training_labels = json.load(bbox_json_file)
	bbox_json_file.close()

	return bbox_training_labels

def create_model():
	# the input layer
	input_img = Input(shape = (imgsize[1], imgsize[0], 1))

	x = Conv2D(32, 7, padding='same')(input_img)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(32, 5, padding='same')(input_img)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)

	x = Conv2D(32, 5, padding='same')(input_img)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)

	x = Flatten()(x)
	x = Dense(48, activation = 'linear')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	#the final encoder layer
	x = Dense(5, activation = 'linear')(x)
	x = BatchNormalization()(x)
	encoded = Activation('tanh')(x)

	x = Dense(32, activation = 'linear')(encoded)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Dense(80, activation = 'linear')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Reshape((5, 8, 2))(x)
	x = UpSampling2D((4, 4))(x)

	x = Conv2D(32, 3, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)

	x = Conv2D(32, 5, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)

	x = Conv2D(32, 5, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)

	decoded = Conv2D(1, 5, activation='sigmoid', padding='same')(x)

	encoder = Model(input_img, encoded)
	autoencoder = Model(input_img, decoded)

	#also make a flow chart of the model
	plot_model(autoencoder, to_file = f'{output_loc}/model_arch.png', show_shapes = True, show_layer_names = True)

	encoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
	autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')

	return encoder, autoencoder

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

	# plt.yscale('log')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Loss progression of cropping network')
	plt.legend(loc = 'best')
	plt.grid(alpha = 0.4)

	plt.savefig(f'{output_loc}/Autoencoder_Losses.png', dpi = 300, bbox_inches = 'tight')
	plt.close()

def plot_encoder_output(encoder_output):
	print(encoder_output.shape)
	plt.scatter(encoder_output[:,0], encoder_output[:,1], s = 2, edgecolor = None, facecolor = 'black', alpha = 0.3)

	plt.savefig(f'{output_loc}/Encoder_val_output.png', bbox_inches = 'tight', dpi = 200)
	plt.close()

def plot_autoencoder_output(val_image_array, autoencoder_output):
	print(autoencoder_output.shape)
	for i in tqdm(range(autoencoder_output.shape[0])):
		fig = plt.figure(figsize = (12, 12))

		ax = fig.add_subplot(211)
		plt.imshow(val_image_array[i,:,:,0], cmap = 'gray')
		plt.title('Input image')

		ax = fig.add_subplot(212)
		plt.imshow(autoencoder_output[i,:,:,0], cmap = 'gray')
		plt.title('Autoencoder output')

		plt.savefig(f'{output_loc}/Autoencoder_output_{i}.png', bbox_inches = 'tight', dpi = 200)
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

def load_images(labs):
	"""
	Load a set of images
	"""
	image_array = []
	for label in tqdm(labs):
		current_label, current_image_arr = load_single_img(label)
		image_array.append(np.expand_dims(current_image_arr, 2))
	return np.array(image_array)

def load_data(n_images, n_val_set, load_train_images = True):
	"""
	Loads the desired data
	"""
	labels = create_json()
	#shuffle labels (so also the images itself)
	np.random.shuffle(labels)

	if n_images + n_val_set > 12000:
		print('WARNING: reducing train set size')
		n_images -= n_val_set

	#load validation set which will be used to generate predictions at
	#the end of training
	val_images = load_images(labels[n_images:n_images+n_val_set])

	if load_train_images:
		#load train set
		train_images = load_images(labels[:n_images])

		return train_images, val_images
	else:
		return val_images

def train_model(n_images, n_val_set, autoencoder):
	train_images, val_images = load_data(n_images, n_val_set, load_train_images = True)

	early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
					patience = 12, verbose = 1,
					restore_best_weights = True, min_delta = 0)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
					factor = 0.5, patience = 4, verbose = 1,
					min_delta = 0, min_lr = 1e-8)

	#data is shuffled in fit
	history = autoencoder.fit(image_array, image_array,
				epochs = 100, validation_split = 0.1,
				batch_size = 32, shuffle = True,
				callbacks = [early_stopping, reduce_lr])

	return history, train_images, val_images

def train_or_load_model(n_images, n_val_set, loadmodel = False):
	if loadmodel:
		print ("Loading architecture from file")
		encoder = saveLoadModel(f'{output_loc}/encoder.h5', save = False, load = True)
		autoencoder = saveLoadModel(f'{output_loc}/autoencoder.h5', save = False, load = True)
		print (" Successfully loaded architecture from file!")

		encoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
		autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')

		val_images = load_images(labels[n_images:n_images+n_val_set], load_val_images = True)
	else:
		print ("Creating new model")
		encoder, autoencoder = create_model()
		print (" Created new model")

		history, train_images, val_images = train_model(n_images - n_val_set, n_val_set, autoencoder)
		train_loss = history.history['loss']
		test_loss = history.history['val_loss']

		print ("Saving model")
		saveLoadModel(f'{output_loc}/autoencoder.h5', model = autoencoder, save = True, load = False)
		saveLoadModel(f'{output_loc}/encoder.h5', model = encoder, save = True, load = False)
		print (" Saving model successful\n")

		#plot loss
		plot_save_losses(train_loss, test_loss)

	return encoder, autoencoder, val_images



#### Tweakable parameters
version = 2
n_images = 12000
n_val_set = 950
loadmodel = True


output_loc = f'autoencoder_{version}'
#check if the output folder is present and if not, make one
checkFolders([output_loc])


#### Train the network
encoder, autoencoder, val_images = train_or_load_model(n_images, n_val_set, loadmodel = loadmodel)


#### now make some predictions
encoder_output = encoder.predict(val_images)

#save output to file if we want to analyse it later
np.save(f'{output_loc}/encoder_val_output.npy', encoder_output)

plot_encoder_output(encoder_output)

#also plot autoencoder output to see the compression it applies
plot_autoencoder_output(val_images, autoencoder.predict(val_images[:20]))
