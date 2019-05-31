import json

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import tensorflow as tf
import keras
from sklearn import model_selection
from keras.utils import Sequence
from keras.preprocessing import image as keras_image
from keras.optimizers import Adam, SGD
import keras.backend as K

from keras.utils.vis_utils import plot_model



from PIL import Image
import cv2

import numpy as np
import os
import argparse
import time
import warnings


from pose_submission import SubmissionWriter
from pose_utils import KerasDataGenerator, checkFolders, OutputResults
from cgan_architecture_10 import model_generator, model_discriminator

from tensorflow.python import debug as tf_debug



"""
	Example script demonstrating training on the SPEED dataset using Keras.
	Usage example: python keras_example.py --epochs [num epochs] --batch [batch size]
"""

def generator_loss(x,y):
#	return -keras.losses.logcosh(x,y)
	return -keras.losses.binary_crossentropy(x,y)




class EarlyStoppingByLossVal(keras.callbacks.Callback):
	'''
	From: https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
	'''
	def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
		super(keras.callbacks.Callback, self).__init__()
		self.monitor = monitor
		self.value = value
		self.verbose = verbose

	def on_epoch_end(self, epoch, logs={}):
		current = logs.get(self.monitor)
		if current is None:
			warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

		if current < self.value:
			if self.verbose > 0:
				print("Epoch %05d: early stopping THR" % epoch)
			self.model.stop_training = True

class DiscriminatorDataGenerator(Sequence):

	"""
	DataGenerator to be used for training of a CGAN
	"""

	def __init__(self, label_list, speed_root,labels_sel=None,
				batch_size=32, dim=(1920, 1200,1),
				shuffle=True, seed = 1, real_p = 0.5,
				generator = None, latent_dim = None, hidden_dim = None,
				clip_range = None):

		# loading dataset
		self.image_root = os.path.join(speed_root, 'images', 'train')
		self.clip_range=clip_range
		self.seen=0
		# Initialization
		self.dim = dim
		self.real_p = real_p
		self.generator = generator
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
									 for label in label_list}
		print(labels_sel)
		if labels_sel is None:
			self.list_IDs = [label['filename'] for label in label_list]
		else:
			self.list_IDs = labels_sel
		self.shuffle = shuffle
		self.indexes = None
		self.on_epoch_end()
		self.r_factor = 64

	def __len__(self):

		""" Denotes the number of batches per epoch. """

#		return int(np.floor(len(self.list_IDs) / self.batch_size)/self.r_factor)
		return 1

	def __getitem__(self, index_i):

		""" Generate one batch of data """
		index = (index_i+self.seen)%int(np.floor(len(self.list_IDs) / self.batch_size))
		print(index)

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
		self.seen += self.batch_size
		if self.seen%int(np.floor(len(self.list_IDs) / self.batch_size))<self.batch_size:
#		if self.shuffle:
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
				y[i] = np.random.rand()*0.1+0.9
#				y[i] = 1
	
	
				# flatten and output
#				np.expand_dims(keras_image.img_to_array(img)[...,0]/255.,2)
				x = np.expand_dims((np.clip(keras_image.img_to_array(img)[...,0],self.clip_range[0],self.clip_range[1])-self.clip_range[0])/(-self.clip_range[0]+self.clip_range[1]),2)
				pose = np.concatenate([q, r])
			else:
#				noise = np.random.rand(self.latent_dim).reshape(1,self.latent_dim)
				noise = np.random.standard_normal(size=self.latent_dim).reshape(1,self.latent_dim)
				# Generate a random direction
				q = np.random.randn(4)
				q = q/np.linalg.norm(q)
				# Generate random direction components
				### TODO: Statistical analyisis of test data for expected values
				r = np.array([np.random.normal(loc=-0.001765,scale=0.321561),
				  np.random.normal(loc=-0.003058,scale=0.411131),
				  np.random.poisson(10.898330)])
				pose = np.concatenate([q, r]).reshape(1,7)
#				print(np.asarray(noise).shape)
#				print(pose.shape)
				ingen = [noise,pose]
				
				x = self.generator.predict(ingen)	
#				y[i] = np.random.rand()*0.1
#				y[i] = 0
				y[i] = 1
				

			X[i,] = x
			C[i,] = pose
			
		Xf=[X,C]
		return Xf, y
	
class GeneratorDataGenerator(Sequence):

	"""
	DataGenerator to be used for training of a CGAN
	"""

	def __init__(self, label_list, speed_root,labels_sel=None,
				batch_size=32, dim=(1200, 1920),
				shuffle=True, seed = 1, real_p = 0.5,
				generator = None, latent_dim = None, clip_range = None):

		# loading dataset
		self.image_root = os.path.join(speed_root, 'images', 'train')
		self.clip_range=clip_range

		# Initialization
		self.dim = dim
		self.real_p = real_p
		self.generator = generator
		self.latent_dim = latent_dim
#		self.batch_size = int(batch_size/2)
		self.batch_size = batch_size
		self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
									 for label in label_list}
		if labels_sel is None:
			self.list_IDs = [label['filename'] for label in label_list]
		else:
			self.list_IDs = labels_sel
		self.shuffle = shuffle
		self.seen=0
		self.indexes = None
		self.on_epoch_end()
		self.init = True
		self.r_factor = 40

	def __len__(self):

		""" Denotes the number of batches per epoch. """
#		# Small initialization first time
#		if self.init:
#			return 1
#		else:
#			return int(np.floor(len(self.list_IDs) / self.batch_size)/self.r_factor)
		return 1

	def __getitem__(self, index_i):

		""" Generate one batch of data """
		index = (index_i+self.seen)%int(np.floor(len(self.list_IDs) / self.batch_size))
		print(index)

		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):

		""" Updates indexes after each epoch """
		self.seen+=self.batch_size

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
#			X[i,] = np.random.random(self.latent_dim)
			X[i,] = np.random.standard_normal(size=self.latent_dim).reshape(1,self.latent_dim)
			# Generate a random direction
			q = np.random.randn(4)
			q = q/np.linalg.norm(q)
			# Generate random direction components
			### TODO: Statistical analyisis of test data for expected values
			r = np.array([np.random.normal(loc=-0.001765,scale=0.321561),
				  np.random.normal(loc=-0.003058,scale=0.411131),
				  np.random.poisson(10.898330)])
			pose = np.concatenate([q, r])
			C[i,] = pose
			y[i] = 0
			
		Xf=[X,C]
		return Xf, y

class CGAN(object):

	def __init__(self, batch_size, epochs, version, load_model, reps = 4):
		
		#### tweakable parameters
		self.batch_size = batch_size
		self.epochs = epochs
		self.version = version
		self.load_model = load_model
		
		self.clip_range=(0,255)
		
		self.cluster = 2

		self.dim = (160, 256,1)
		#size of the test set as a fraction of the total amount of data
		self.test_size = 0.02
		self.latent_dim = 100
		self.hidden_dim = 100

		#dropout percentage
		self.dropout = 0.3

		self.learning_rate_g = 0.0002
		self.learning_rate_d = 0.0002
		self.learning_rate_decay = 0.0001
		self.reps = reps
		self.epochs = epochs

		#### constant parameters
		# self.dataset_loc = '../../speed'
		self.dataset_loc = 'speed/speed/'
		self.image_root = os.path.join(self.dataset_loc, 'images', 'train')
#		self.loss_g=keras.losses.mean_squared_error
#		self.loss_g=keras.losses.logcosh
#		self.loss_g=generator_loss
		self.loss_g=keras.losses.binary_crossentropy
#		self.loss_d=keras.losses.logcosh
		self.loss_d=keras.losses.binary_crossentropy


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
				  'seed': 1,
				  'clip_range': self.clip_range}

		self.dataloader()
		if self.load_model < 0:
			self.generator = model_generator(
					latent_dim = self.latent_dim,
					input_shape = self.dim,
					hidden_dim = self.hidden_dim)
			self.generator.compile(
					loss = self.loss_g,
					optimizer = Adam(
							lr = self.learning_rate_g,
							decay = self.learning_rate_decay,
							beta_1=0,
							beta_2=0.9),
					metrics = ['accuracy'])
			self.discriminator = model_discriminator(
					input_shape = self.dim,
					hidden_dim = self.hidden_dim)
			self.discriminator.compile(
					loss = self.loss_d,
					optimizer = Adam(
							lr = self.learning_rate_d,
							decay = self.learning_rate_decay,
							beta_1=0,
							beta_2=0.9),
					metrics = ['accuracy'])
			self.total = self.model_total()
			self.total.compile(
					loss = self.loss_g,
					optimizer = Adam(
							lr = self.learning_rate_g,
							decay = self.learning_rate_decay,
							beta_1=0,
							beta_2=0.9),
					metrics = ['accuracy'])
		else:
			print(self.load_model)
			self.generator = self.gen_output.saveLoadModel(f'Version_CGAN_{self.load_model}/generator_v{self.load_model}.h5', load=True,loss=self.loss_d)
			self.discriminator = self.gen_output.saveLoadModel(f'Version_CGAN_{self.load_model}/discriminator_v{self.load_model}.h5', load=True,loss=self.loss_g)
			self.total = self.model_total()
			self.total.compile(
					loss = self.loss_g,
					optimizer = Adam(
							lr = self.learning_rate_g,
							decay = self.learning_rate_decay,
							beta_1=0.5),
					metrics = ['accuracy'])
					
		plot_model(self.generator, to_file = f'{self.output_loc}generator_arch_v{self.version}.png', show_shapes = True, show_layer_names = True)
		plot_model(self.discriminator, to_file = f'{self.output_loc}discriminator_arch_v{self.version}.png', show_shapes = True, show_layer_names = True)
		plot_model(self.total, to_file = f'{self.output_loc}total_arch_v{self.version}.png', show_shapes = True, show_layer_names = True)

	def dataloader(self):
		"""
		Initialize the data
		"""
		# Loading and splitting dataset labels
		with open(os.path.join(self.dataset_loc, 'train' + '.json'), 'r') as f:
			label_list = json.load(f)

#		label_list = label_list[:6000]
		labels_autoenc = np.load('autoencoder/image_labels.npy')
		clusters_autoenc = np.load('autoencoder/hierarch_cluster_labels.npy')
		labels_autoenc = labels_autoenc[clusters_autoenc==self.cluster]
		labels_sel=np.array([x['filename'] for x in labels_autoenc])

		#shuffle and split
		train_labels, validation_labels = model_selection.train_test_split(
				label_list, test_size = self.test_size, shuffle = True)

		# Data generators for training and validation
		self.training_discriminator = DiscriminatorDataGenerator(
				label_list,
				self.dataset_loc,
				labels_sel=labels_sel,
				**self.params)
		self.training_generator = GeneratorDataGenerator(
				label_list,
				self.dataset_loc,
				labels_sel=labels_sel,
				**self.params)

	def train_cgan(self):
		
		print("Training initial discriminator")
#		tensorboard = keras.callbacks.TensorBoard(log_dir=f'{self.output_loc}/logs', write_grads=True)
#		callbacks=[keras.callbacks.ProgbarLogger(count_mode='steps'), tensorboard]
#		value_stopping = EarlyStoppingByLossVal(monitor='val_loss', value=0.3, verbose=1)
#		callbacks_d=[keras.callbacks.ProgbarLogger(count_mode='steps'), value_stopping]
#		value_stopping = EarlyStoppingByLossVal(monitor='val_loss', value=-0.18, verbose=1)
#		value_stopping = EarlyStoppingByLossVal(monitor='val_loss', value=-0.5, verbose=1)
#		callbacks_g=[keras.callbacks.ProgbarLogger(count_mode='steps'), value_stopping]
		callbacks_d=[keras.callbacks.ProgbarLogger(count_mode='steps')]
		callbacks_g=[keras.callbacks.ProgbarLogger(count_mode='steps')]
		# Initialize discriminator with real images
		self.training_discriminator.real_p = 1
#		self.validation_discriminator.real_p = 1
		self.training_discriminator.r_factor = 200
#		self.validation_discriminator.r_factor = 1
#		self.validation_generator.r_factor = 1
		self.discriminator.trainable = True
		history = self.discriminator.fit_generator(
				self.training_discriminator,
				epochs=self.epochs,
#				validation_data=self.validation_discriminator,
				callbacks=callbacks_d)
		#save the model
		self.gen_output.saveLoadModel(
				f'{self.output_loc}discriminator_v{self.version}.h5',
				model = self.discriminator,
				save = True)
		
		train_loss_discriminator = history.history['loss']
#		test_loss_discriminator = history.history['val_loss']

		print('Training losses: ', history.history['loss'])
#		print('Validation losses: ', history.history['val_loss'])
		
#		train_loss_discriminator = []
		test_loss_discriminator = [1]
		
		
		#save and plot losses
#		self.gen_output.plot_save_losses(
#				train_loss_discriminator,
#				test_loss_discriminator)
		
		
	
		train_loss_total = []
		test_loss_total = [1]
		rep_g = 0
		rep_d = 1
#		self.training_discriminator.r_factor = 180
#		self.training_discriminator.r_factor = 256
		
		# Train alternatively generator and discriminator
		for rep in range(self.reps):
			
#			K.set_value(self.total.optimizer.lr, self.learning_rate_g)
#			K.set_value(self.discriminator.optimizer.lr, self.learning_rate_d)
			# Train generator
			print('Training iteration %i of generator' % rep)
			# Load discriminator weights
#			if rep == 0:
#				self.training_generator.init = True
#			else:
#				self.training_generator.init = False
			self.training_generator.r_factor = 64
			rep_d = 0
			rep_g += 1
			self.training_generator.init = False
#			self.validation_generator.init = False
				
			self.discriminator.trainable = False
			history = self.total.fit_generator(
					self.training_generator,
					epochs=self.epochs,
#					validation_data=self.validation_generator,
					callbacks=callbacks_g)
			#save the model
			
			
			train_loss_total.extend(history.history['loss'])
			history = self.total.fit_generator(
					self.training_generator,
					epochs=self.epochs,
#					validation_data=self.validation_generator,
					callbacks=callbacks_g)
			#save the model
			
			
			train_loss_total.extend(history.history['loss'])
#			test_loss_total.extend(history.history['val_loss'])
	
#				print('Training losses: ', history.history['loss'])
#				print('Validation losses: ', history.history['val_loss'])
			
			#save and plot losses
			self.gen_output.plot_save_losses_gan(
					train_loss_total,
					train_loss_discriminator)
#			
			#save the model
			
			if rep%100==0:
				self.gen_output.saveLoadModel(
						f'{self.output_loc}total_v{self.version}.h5',
						model = self.total,
						save = True)
				self.gen_output.saveLoadModel(
						f'{self.output_loc}generator_v{self.version}.h5',
						model = self.generator,
						save = True)
				self.gen_output.saveLoadModel(
						f'{self.output_loc}discriminator_v{self.version}.h5',
						model = self.discriminator,
						save = True)
				self.generate_examples(10)
			
			if (rep is not self.reps-1):
				rep_g = 0
				rep_d += 1
				print('Training iteration %i of discriminator' % rep)
				self.discriminator.trainable = True
				self.training_discriminator.real_p = 0.5
#				self.validation_discriminator.real_p = 0.5
				self.training_discriminator.generator = self.generator
#				self.validation_discriminator.generator = self.generator
				history = self.discriminator.fit_generator(
						self.training_discriminator,
						epochs=self.epochs,
#						validation_data=self.validation_discriminator,
						callbacks=callbacks_d)
				#save the model
				
				
				train_loss_discriminator.extend(history.history['loss'])
#				test_loss_discriminator.extend(history.history['val_loss'])
		
#				print('Training losses: ', history.history['loss'])
#				print('Validation losses: ', history.history['val_loss'])
				
				#save and plot losses
#				self.gen_output.plot_save_losses(
#						train_loss_discriminator,
#						test_loss_discriminator)

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
#			noise = np.random.randint(2, size=self.latent_dim).reshape(1,self.latent_dim)
			noise = np.random.standard_normal(size=self.latent_dim).reshape(1,self.latent_dim)
			# Generate a random direction
			q = np.random.randn(4)
			q = q/np.linalg.norm(q)
			# Generate random direction components
			### TODO: Statistical analyisis of test data for expected values
			r = np.array([np.random.normal(loc=-0.001765,scale=0.321561),
				  np.random.normal(loc=-0.003058,scale=0.411131),
				  np.random.poisson(10.898330)])
			pose = np.concatenate([q, r]).reshape(1,7)
			x = ((self.generator.predict([noise,pose]))*(self.clip_range[1]-self.clip_range[0])+self.clip_range[0]).reshape(self.dim[0],self.dim[1])
#			print(np.amax(x))
			plt.imsave(f'cgantest/TestOutput_{ID}.png', x, cmap = 'gray', dpi = 300, vmin = 0, vmax = 255)
			
			with open(f'cgantest/TestOutput_{ID}.txt', "w") as myfile:
				myfile.write(str(pose))
				
			ID_e='img014933.jpg'
			img_path = os.path.join(self.image_root, ID_e)
			img = Image.open(img_path)
			img = img.resize((self.dim[1],self.dim[0]))
			img_a = keras_image.img_to_array(img)
			
			hist = cv2.calcHist([img_a], [0], None, [256], [0, 256])
			plt.figure()
			plt.title("Grayscale Histogram")
			plt.xlabel("Bins")
			plt.ylabel("# of Pixels")
			plt.plot(hist)
#			plt.xlim([0, 50])
			plt.savefig(f'cgantest/{ID_e}_hist.png')
				
	def model_total(self):
		self.discriminator.trainable=False
		inputnoise = keras.layers.Input(shape=(self.latent_dim,), name="total_i1")
		inputpose = keras.layers.Input(shape=(7,), name="total_i2")
		generator = self.generator([inputnoise,inputpose])
		discriminator =  self.discriminator([generator, inputpose])
		model_final =  keras.models.Model(inputs = [inputnoise,inputpose], outputs=discriminator)
		
		return model_final
	
	


def main(batch_size, epochs, version, load_model):

	""" Setting up data generators and model, training, and evaluating model on test and real_test sets. """

	
	#initialize parameters, data loading and the network architecture
	cgan = CGAN(batch_size, epochs, version, load_model, reps = 2000000)

	#train the network
	cgan.train_cgan()

	#evaluate the results
	cgan.generate_examples(10)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
parser.add_argument('--epochs', help='Number of epochs for training.', default = 1)
parser.add_argument('--batch', help='number of samples in a batch.', default = 8)
parser.add_argument('--version', help='version of the neural network.', default = 21)
parser.add_argument('--load', help='load a previously trained network.', default = -1)
args = parser.parse_args()

main(int(args.batch), int(args.epochs), int(args.version), int(args.load))
