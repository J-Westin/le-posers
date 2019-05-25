import json

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import tensorflow as tf
import keras
from sklearn import model_selection
from keras.utils import Sequence
from keras.preprocessing import image as keras_image
from keras.optimizers import Adam



from PIL import Image

import numpy as np
import os
import argparse
import time


from pose_submission import SubmissionWriter
from pose_utils import KerasDataGenerator, checkFolders, OutputResults
from cgan_architecture_3 import model_generator, model_discriminator, model_total

from tensorflow.python import debug as tf_debug



"""
	Example script demonstrating training on the SPEED dataset using Keras.
	Usage example: python keras_example.py --epochs [num epochs] --batch [batch size]
"""
class DiscriminatorDataGenerator(Sequence):

	"""
	DataGenerator to be used for training of a CGAN
	"""

	def __init__(self, label_list, speed_root,
				batch_size=32, dim=(1920, 1200,1),
				shuffle=True, seed = 1, real_p = 0.5,
				generator = None, latent_dim = None, hidden_dim = None):

		# loading dataset
		self.image_root = os.path.join(speed_root, 'images', 'train')

		# Initialization
		self.dim = dim
		self.real_p = real_p
		self.generator = generator
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
									 for label in label_list}
		self.list_IDs = [label['filename'] for label in label_list]
		self.shuffle = shuffle
		self.indexes = None
		self.on_epoch_end()

	def __len__(self):

		""" Denotes the number of batches per epoch. """

		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):

		""" Generate one batch of data """

		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):

		""" Updates indexes after each epoch """

		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	
	def __data_generation(self, list_IDs_temp):

		""" Generates data containing batch_size samples """

		# Initialization
		X = np.empty((self.batch_size, *self.dim))
		y = np.empty((self.batch_size, 1), dtype=float)
		C = np.empty((self.batch_size, 7), dtype=float)
		
		

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			real = np.random.random()
			if real<self.real_p:
				img_path = os.path.join(self.image_root, ID)
				img = Image.open(img_path)
				q, r = self.labels[ID]['q'], self.labels[ID]['r']
				
				
				img = img.resize((self.dim[1],self.dim[0]))
	
	
				# flatten and output
				x = np.expand_dims(keras_image.img_to_array(img)[...,0]/255.,2)
				pose = np.concatenate([q, r])
				y[i] = 1
			else:
				noise = np.random.rand(self.latent_dim).reshape(1,self.latent_dim)
				# Generate a random direction
				q = np.random.randn(4)
				q = q/np.linalg.norm(q)
				# Generate random direction components
				### TODO: Statistical analyisis of test data for expected values
				r = np.array([np.random.randn()*2.,np.random.randn()*2.,
				  np.random.poisson(3.)])
				pose = np.concatenate([q, r]).reshape(1,7)
#				print(np.asarray(noise).shape)
#				print(pose.shape)
				ingen = [noise,pose]
				
				x = self.generator.predict(ingen)	
				y[i] = 0
				

			X[i,] = x
			C[i,] = pose
			
		Xf=[X,C]
		return Xf, y
	
class GeneratorDataGenerator(Sequence):

	"""
	DataGenerator to be used for training of a CGAN
	"""

	def __init__(self, label_list, speed_root,
				batch_size=32, dim=(1200, 1920),
				shuffle=True, seed = 1, real_p = 0.5,
				generator = None, latent_dim = None):

		# loading dataset
		self.image_root = os.path.join(speed_root, 'images', 'train')

		# Initialization
		self.dim = dim
		self.real_p = real_p
		self.generator = generator
		self.latent_dim = latent_dim
		self.batch_size = int(batch_size/2)
		self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
									 for label in label_list}
		self.list_IDs = [label['filename'] for label in label_list]
		self.shuffle = shuffle
		self.indexes = None
		self.on_epoch_end()

	def __len__(self):

		""" Denotes the number of batches per epoch. """

		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):

		""" Generate one batch of data """

		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):

		""" Updates indexes after each epoch """

		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	
	def __data_generation(self, list_IDs_temp):

		""" Generates data containing batch_size samples """

		# Initialization
		X = np.empty((self.batch_size, self.latent_dim))
		y = np.empty((self.batch_size, 1), dtype=float)
		C = np.empty((self.batch_size, 7), dtype=float)

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			X[i,] = np.random.random(self.latent_dim)
			# Generate a random direction
			q = np.random.randn(4)
			q = q/np.linalg.norm(q)
			# Generate random direction components
			### TODO: Statistical analyisis of test data for expected values
			r = np.array([np.random.randn()*2.,np.random.randn()*2.,
			  np.random.poisson(3.)])
			pose = np.concatenate([q, r])
			C[i,] = pose
			y[i] = 1
			
		Xf=[X,C]
		return Xf, y

class CGAN(object):

	def __init__(self, batch_size, epochs, version, load_model, reps = 4):
		#### tweakable parameters
		self.batch_size = batch_size
		self.epochs = epochs
		self.version = version
		self.load_model = load_model

		self.dim = (200, 320,1)
		#size of the test set as a fraction of the total amount of data
		self.test_size = 0.1
		self.latent_dim = 64
		self.hidden_dim = 64

		#dropout percentage
		self.dropout = 0.3

		self.learning_rate = 0.001
		self.learning_rate_decay = 0
		self.reps = reps
		self.epochs = epochs

		#### constant parameters
		# self.dataset_loc = '../../speed'
		self.dataset_loc = 'speed/speed/'


		self.output_loc = f'./Version_CGAN_{self.version}/'
		self.model_summary_name = f'model_summlisary_v{self.version}.txt'

		#### initialize some stuff
		#check if folders are present and make them if necessary
		checkFolders([self.output_loc])

		self.gen_output = OutputResults(self)


#		self.sess = sess
		self.params = {'dim': self.dim,
				  'batch_size': self.batch_size,
				  'latent_dim': self.latent_dim,
				  'shuffle': True,
				  'seed': 1}

		self.dataloader()
		if self.load_model < 0:
			print(self.learning_rate)
			self.generator = model_generator(
					latent_dim = self.latent_dim,
					input_shape = self.dim,
					hidden_dim = self.hidden_dim)
			self.discriminator = model_discriminator(
					input_shape = self.dim,
					hidden_dim = self.hidden_dim)
			self.total = model_total(
					latent_dim = self.latent_dim,
					input_shape = self.dim,
					hidden_dim = self.hidden_dim)
		else:
			print(self.load_model)
			self.generator = self.gen_output.saveLoadModel(f'Version_CGAN_{self.load_model}/generator_v{self.load_model}.h5', load=True)
			self.discriminator = self.gen_output.saveLoadModel(f'Version_CGAN_{self.load_model}/discriminator_v{self.load_model}.h5', load=True)
			self.total = self.gen_output.saveLoadModel(f'Version_CGAN_{self.load_model}/total_v{self.load_model}.h5', load=True)


	def dataloader(self):
		"""
		Initialize the data
		"""
		# Loading and splitting dataset labels
		with open(os.path.join(self.dataset_loc, 'train' + '.json'), 'r') as f:
			label_list = json.load(f)

		# label_list = label_list[:n_imgs]

		#shuffle and split
		train_labels, validation_labels = model_selection.train_test_split(
				label_list, test_size = self.test_size, shuffle = True)

		# Data generators for training and validation
		self.training_discriminator = DiscriminatorDataGenerator(
				train_labels,
				self.dataset_loc,
				**self.params)
		self.validation_discriminator = DiscriminatorDataGenerator(
				validation_labels,
				self.dataset_loc,
				**self.params)
		self.training_generator = GeneratorDataGenerator(
				train_labels,
				self.dataset_loc,
				**self.params)
		self.validation_generator = GeneratorDataGenerator(
				validation_labels,
				self.dataset_loc,
				**self.params)

	def train_cgan(self, loss='mse'):
		callbacks=[keras.callbacks.ProgbarLogger(count_mode='steps')]

		# Initialize discriminator with real images
		print("Training initial discriminator")
		self.discriminator.compile(
				loss = loss,
				optimizer = Adam(
						lr = self.learning_rate*2,
						decay = self.learning_rate_decay),
				metrics = ['accuracy'])
		self.training_discriminator.real_p = 1
		self.validation_discriminator.real_p = 1
			
		self.total.compile(
				loss = loss,
				optimizer = Adam(
						lr = self.learning_rate/2,
						decay = self.learning_rate_decay),
				metrics = ['accuracy'])
		self.training_discriminator.generator = self.generator
		self.validation_discriminator.generator = self.generator
	
		history = self.discriminator.fit_generator(
				self.training_discriminator,
				epochs=self.epochs,
				validation_data=self.validation_discriminator,
				callbacks=callbacks)
		#save the model
		self.gen_output.saveLoadModel(
				f'{self.output_loc}discriminator_v{self.version}.h5',
				model = self.discriminator,
				save = True)
		
		train_loss_discriminator = history.history['loss']
		test_loss_discriminator = history.history['val_loss']

		print('Training losses: ', history.history['loss'])
		print('Validation losses: ', history.history['val_loss'])
		
		#save and plot losses
		self.gen_output.plot_save_losses(
				train_loss_discriminator,
				test_loss_discriminator)
		
		
		train_loss_total = []
		test_loss_total = []
		
		# Train alternatively generator and discriminator
		for rep in range(self.reps):
			# Train generator
			print('Training iteration %i of generator' % rep)
			# Load discriminator weights
			self.total.load_weights(f'{self.output_loc}discriminator_v{self.version}.h5', by_name=True)
			history = self.total.fit_generator(
					self.training_generator,
					epochs=self.epochs,
					validation_data=self.validation_generator,
					callbacks=callbacks)
			#save the model
			self.gen_output.saveLoadModel(
					f'{self.output_loc}total_v{self.version}.h5',
					model = self.total,
					save = True)
			
			train_loss_total.extend(history.history['loss'])
			test_loss_total.extend(history.history['val_loss'])
	
			print('Training losses: ', history.history['loss'])
			print('Validation losses: ', history.history['val_loss'])
			
			#save and plot losses
			self.gen_output.plot_save_losses(
					train_loss_total,
					test_loss_total)
			
			self.generator.load_weights(f'{self.output_loc}total_v{self.version}.h5', by_name=True)
			#save the model
			self.gen_output.saveLoadModel(
					f'{self.output_loc}generator_v{self.version}.h5',
					model = self.generator,
					save = True)
			self.generator._make_predict_function()
			
			self.generate_examples(10)
			
			if rep is not self.reps-1:
				# Train discriminator
				print('Training iteration %i of discriminator' % rep)
				self.training_discriminator.real_p = 0.1
				self.validation_discriminator.real_p = 0.1
				self.training_discriminator.generator = self.generator
				self.validation_discriminator.generator = self.generator
				history = self.discriminator.fit_generator(
					self.training_discriminator,
					epochs=self.epochs,
					validation_data=self.validation_discriminator,
					callbacks=callbacks)
				#save the model
				self.gen_output.saveLoadModel(
						f'{self.output_loc}discriminator_v{self.version}.h5',
						model = self.discriminator,
						save = True)
				
				train_loss_discriminator.extend(history.history['loss'])
				test_loss_discriminator.extend(history.history['val_loss'])
		
				print('Training losses: ', history.history['loss'])
				print('Validation losses: ', history.history['val_loss'])
				
				#save and plot losses
				self.gen_output.plot_save_losses(
						train_loss_discriminator,
						test_loss_discriminator)

	def savePrint(self, s):
		"""
		Write a given Keras model summary to file
		"""

		with open(self.output_loc + self.model_summary_name, 'a') as f:
			print(s, file = f)
			
	'''
	def generator_loss(self,y_pred,y_real):
		discriminator = tf.keras.estimator.model_to_estimator(
				keras_model=self.discriminator)
		print(y_real.shape)
		print(y_pred)
		print([y_pred[0],y_pred[1]])
		return y_real-self.discriminator.predict([y_pred[:,:,],y_pred[1]],steps = 1)
	'''
		
	def generate_examples(self, examples):
		# Generate and save images
		for ID in range(examples):
			noise = np.random.random(self.latent_dim).reshape(1,self.latent_dim)
			# Generate a random direction
			q = np.random.randn(4)
			q = q/np.linalg.norm(q)
			# Generate random direction components
			### TODO: Statistical analyisis of test data for expected values
			r = np.array([np.random.randn()*2.,np.random.randn()*2.,
			  np.random.poisson(3.)])
			pose = np.concatenate([q, r]).reshape(1,7)
			x = (self.generator.predict([noise,pose])*255).reshape(self.dim[0],self.dim[1])
			plt.imsave(f'cgantest/TestOutput_{ID}.png', x, cmap = 'gray', dpi = 300)
			with open(f'cgantest/TestOutput_{ID}.txt', "w") as myfile:
				myfile.write(str(pose))
	


def main(batch_size, epochs, version, load_model):

	""" Setting up data generators and model, training, and evaluating model on test and real_test sets. """

	
	#initialize parameters, data loading and the network architecture
	cgan = CGAN(batch_size, epochs, version, load_model, reps = 8)

	#train the network
	cgan.train_cgan()

	#evaluate the results
	cgan.generate_examples(10)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
parser.add_argument('--epochs', help='Number of epochs for training.', default = 1)
parser.add_argument('--batch', help='number of samples in a batch.', default = 8)
parser.add_argument('--version', help='version of the neural network.', default = 0)
parser.add_argument('--load', help='load a previously trained network.', default = -1)
args = parser.parse_args()

main(int(args.batch), int(args.epochs), int(args.version), int(args.load))
