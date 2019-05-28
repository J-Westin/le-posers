import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx

from PIL import Image

import keras

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Input, UpSampling2D, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model #requires pydot and graphviz

#Hierarchial DBSCAN: https://github.com/scikit-learn-contrib/hdbscan
import hdbscan
#Parallel tSNE: https://github.com/DmitryUlyanov/Multicore-TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import sklearn.cluster as skcluster
# import sklearn.manifold as skmanifold

from tqdm import tqdm

from pose_utils import KerasDataGenerator, SatellitePoseEstimationDataset, Camera, checkFolders


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
		print('Loading model from {0}...'.format(filename))
		model = keras.models.load_model(filename)
		print('Finished loading model')

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

def cluster_encoder_output(encoder_output, val_images, algorithm = 'k-means', use_tSNE = True):
	"""
	Applies a clustering algorithm from sklearn to the output of an encoder
	to see if the observed clustering is representative of real grouping
	"""

	def scatterplot_2dim(X, cluster_labels = None, unique_labels = None):
		"""
		Make a 2D scatter plot of some data using labels where possible
		"""

		#process the labels for nice colour maps
		if cluster_labels is not None:
			#obtain a colour map for the different cluster labels
			jet = plt.get_cmap('jet')
			cNorm  = colors.Normalize(vmin = 0, vmax = len(unique_labels))
			scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)

		#plot the points
		if cluster_labels is not None:
			#with label markers
			for i, cluster_lab in enumerate(unique_labels):
				plt.scatter(X[cluster_labels == cluster_lab,0],
							X[cluster_labels == cluster_lab,1],
							s = 2, edgecolor = None, facecolor = scalarMap.to_rgba(i),
							alpha = 0.6, label = cluster_lab)

			plt.legend(loc = 'best')

			return scalarMap
		else:
			#without label markers (as there are no labels available)
			plt.scatter(X[:,0],
						X[:,1],
						s = 2, edgecolor = None, facecolor = 'black',
						alpha = 0.6)


	if algorithm == 'HDBSCAN':
		#start clustering with HDBSCAN
		clusterer = hdbscan.HDBSCAN()
	elif algorithm == 'DBSCAN':
		#start clustering with HDBSCAN
		clusterer = skcluster.DBSCAN(eps = 0.08)
	elif algorithm == 'hierarch':
		#agglomerative (hierarchial) clustering
		clusterer = skcluster.AgglomerativeClustering(n_clusters = 4)
	elif algorithm == 'k-means':
		#agglomerative (hierarchial) clustering
		clusterer = skcluster.KMeans(n_clusters = 4)
	else:
		raise ValueError(f'{algorithm}: invalid algorithm specified')

	#fit the algorithm and extract labels
	clusterer.fit(encoder_output)
	cluster_labels = clusterer.labels_

	np.save(f'{encoder_clustering_output_loc}/{algorithm}_cluster_labels.npy', cluster_labels)

	unique_labels = np.unique(cluster_labels)


	if use_tSNE:
		#apply tSNE dimensionality reduction
		starttime = time.time()
		print('Starting tSNE...')

		#apply dimensionality reduction: 5 to 2 dimensions
		X_embedded = TSNE(n_components = 2, n_jobs = 8).fit_transform(encoder_output)

		print(f'Finished tSNE. Runtime: {time.time()-starttime:0.03f} s')

		scalarMap = scatterplot_2dim(X_embedded, cluster_labels, unique_labels)
	else:
		scalarMap = scatterplot_2dim(encoder_output, cluster_labels, unique_labels)

	name_addition = ''
	if use_tSNE:
		name_addition += '_tSNE'

	if use_tSNE:
		plt.title(f'Encoder output in 2 tSNE dimensions with {algorithm} clustering')
	else:
		plt.title(f'Encoder output in 2 of 5 dimensions with {algorithm} clustering')

	plt.savefig(f'{encoder_clustering_output_loc}/Encoder_val_{algorithm}_clustering{name_addition}.png', bbox_inches = 'tight', dpi = 200)
	plt.close()


	#### Now we will plot images of examples of every cluster found
	assert len(val_images) == encoder_output.shape[0], 'Number of validation images not equal to encoder output. Did you actually load the validation images?'

	#we first want to shuffle all three arrays so we are not introducing
	#systematic bias
	shuffle_loc = np.random.permutation(len(val_images))
	val_images = val_images[shuffle_loc]
	encoder_output = encoder_output[shuffle_loc]
	cluster_labels = cluster_labels[shuffle_loc]

	#number of images per cluster label
	n_per_cluster = 12

	for j, lab in enumerate(tqdm(unique_labels, desc = 'Saving images per cluster label')):
		#extract the first n_per_cluster positions
		locs = np.where(cluster_labels == lab)[0][:n_per_cluster]

		for i in tqdm(range(len(locs))):
			plt.imshow(val_images[locs[i],:,:,0], cmap = 'gray')

			imgheight = val_images[locs[i],:,:,0].shape[0]
			imgwidth = val_images[locs[i],:,:,0].shape[1]

			#add a box with colour matching that of the scatter plot
			rectwidth = 10
			rectheight = 10
			rect = patches.Rectangle((0, 0),
			 				rectwidth, rectheight,
							facecolor = scalarMap.to_rgba(j),
							linewidth = 0)
			# Add the patch to the Axes
			ax = plt.gca()
			ax.add_patch(rect)

			#fix limits so we don't get a whitespace due to the Rectangle
			#being close to the axes
			plt.xlim((0, imgwidth))
			plt.ylim((imgheight, 0))

			#also make the title have the colour matching the scatter plot
			font = {'family': 'serif',
		        'color': scalarMap.to_rgba(j),
		        'weight': 'normal'
	        	}
			plt.title(f'Validation image cluster {lab} number {i}', fontdict = font)

			plt.savefig(f'{encoder_clustering_output_loc}/Val_image_{algorithm}_c{lab}_idx{i}.png', bbox_inches = 'tight', dpi = 200)
			plt.close()

def plot_autoencoder_output(val_image_array, autoencoder_output):
	for i in tqdm(range(autoencoder_output.shape[0])):
		fig = plt.figure(figsize = (12, 12))

		#the input image
		ax = fig.add_subplot(211)
		plt.imshow(val_image_array[i,:,:,0], cmap = 'gray')
		plt.title('Input image')

		#the output of the autoencoder
		ax = fig.add_subplot(212)
		plt.imshow(autoencoder_output[i,:,:,0], cmap = 'gray')
		plt.title('Autoencoder output')

		plt.savefig(f'{autoencoder_imgs_output_loc}/Autoencoder_output_{i}.png', bbox_inches = 'tight', dpi = 200)
		plt.close()

def load_single_img(current_label):
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

def load_data(n_images, n_val_set, load_train_images = True, random_shuffle = True):
	"""
	Loads the desired data
	"""
	labels = create_json()
	#shuffle labels (so also the images itself)
	if random_shuffle:
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

		return train_images, val_images, labels
	else:
		return val_images, labels

def train_model(n_images, n_val_set, autoencoder):
	train_images, val_images, labels = load_data(n_images, n_val_set, load_train_images = True)

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

	return history, train_images, val_images, labels

def train_or_load_model(n_iTritaniummages, n_val_set, loadmodel = False, load_val_imgs = True):
	if loadmodel:
		print ("Loading architecture from file")
		encoder = saveLoadModel(f'{output_loc}/encoder.h5', save = False, load = True)
		autoencoder = saveLoadModel(f'{output_loc}/autoencoder.h5', save = False, load = True)
		print (" Successfully loaded architecture from file!")

		encoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
		autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')

		if load_val_imgs:
			val_images, labels = load_data(n_images, n_val_set, load_train_images = False, random_shuffle = False)
		else:
			val_images = np.array([])
	else:
		print ("Creating new model")
		encoder, autoencoder = create_model()
		print (" Created new model")

		history, train_images, val_images, labels = train_model(n_images - n_val_set, n_val_set, autoencoder)
		train_loss = history.history['loss']
		test_loss = history.history['val_loss']

		print ("Saving model")
		saveLoadModel(f'{output_loc}/autoencoder.h5', model = autoencoder, save = True, load = False)
		saveLoadModel(f'{output_loc}/encoder.h5', model = encoder, save = True, load = False)
		print (" Saving model successful\n")

		#plot loss
		plot_save_losses(train_loss, test_loss)

	return encoder, autoencoder, val_images, labels


def run_predictions(encoder, autoencoder, val_images, labels, load_val_imgs = True):
	if load_val_imgs:
		encoder_output = encoder.predict(val_images)

		#save output to file if we want to analyse it later
		np.save(f'{encoder_clustering_output_loc}/encoder_output_{imageset}.npy', encoder_output)
		np.save(f'{encoder_clustering_output_loc}/image_labels_{imageset}.npy', np.array(labels))
	else:
		encoder_output = np.load(f'{encoder_clustering_output_loc}/encoder_output_{imageset}.npy')
		labels = np.load(f'{encoder_clustering_output_loc}/image_labels_{imageset}.npy')

	cluster_encoder_output(encoder_output, val_images, algorithm = 'hierarch', use_tSNE = False)

	#also plot autoencoder output to see the compression it applies
	np.random.shuffle(val_images)
	plot_autoencoder_output(val_images, autoencoder.predict(val_images[:20]))


#### Tweakable parameters
version = 2
#number of images for training (10% will be split off for the test set)
n_images = 0
#number of images for validation
n_val_set = 12000

imageset = 'test'

loadmodel = True
#True: load all validation images. False: load only the encoder output
load_val_imgs = True


output_loc = f'autoencoder_{version}'
#check if the output folder is present and if not, make one
checkFolders([output_loc])
#make folders in the output loc
autoencoder_imgs_output_loc = f'{output_loc}/Autoencoder_imgs'
encoder_clustering_output_loc = f'{output_loc}/Encoder_clustering'
checkFolders([autoencoder_imgs_output_loc, encoder_clustering_output_loc])


#### Train the network
encoder, autoencoder, val_images, labels = train_or_load_model(n_images, n_val_set, loadmodel = loadmodel, load_val_imgs = load_val_imgs)

#### now make some predictions
run_predictions(encoder, autoencoder, val_images, labels, load_val_imgs = load_val_imgs)
