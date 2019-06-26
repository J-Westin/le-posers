'''
https://machinelearningmastery.com/save-load-keras-deep-learning-models/

Remove last layer of model with
model.layers.pop()

Freeze layers
for layer in model.layers:
    layer.trainable = False
'''


import json

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.optimizers import Adam
import tensorflow as tf
import keras
from sklearn import model_selection


from PIL import Image

import numpy as np
import os
import argparse
import time
import warnings


from pose_submission import SubmissionWriter
from pose_utils import KerasDataGenerator, checkFolders, OutputResults
from pose_architecture_ortn import create_model as create_model_ortn
from pose_architecture_pstn import create_model as create_model_pstn


class POSE_NN(object):

	def __init__(self, batch_size, epochs, version, load_model, loss = "MSE", use_early_stop = False, crop = True, cluster = None, output = 'BOTH'):
		#### tweakable parameters
		self.batch_size = batch_size
		self.epochs = epochs
		self.version = version
		self.load_model = load_model
		self.loss = loss
		self.use_early_stop = use_early_stop
		self.crop = crop
		self.cluster = cluster
		self.output = output

		self.imgsize = 224
		#size of the test set as a fraction of the total amount of data
		self.test_size = 0.1

		#dropout percentage
		self.dropout = 0.3

		self.learning_rate = 0.001

		#### constant parameters
		# self.dataset_loc = '../../speed'
		self.dataset_loc = '/data/s1530194/speed'

		self.output_loc = f'./Version_{self.version}/'
		self.model_summary_name = f'model_summary_v{self.version}_c{self.cluster}_o{self.output}.txt'

		#### initialize some stuff
		#check if folders are present and make them if necessary
		checkFolders([self.output_loc])

		self.gen_output = OutputResults(self)


		# Load cropper if selected
		if self.crop:
			# load json and create model
			loaded_model = keras.models.load_model('cropping_network_development/croppermodel8_c'+str(self.cluster)+'.h5')
			# Compile model for evaluating only
			loaded_model._make_predict_function()
#			loaded_model.compile(optimizer='Adam',loss='mse')
			self.cropper_model = loaded_model

			self.params = {'dim': (self.imgsize, self.imgsize),
					  'batch_size': self.batch_size,
					  'n_channels': 3,
					  'shuffle': True,
					  'randomRotations': False,
					  'seed': 1,
					  'crop': self.crop,
					  'cropper_model': self.cropper_model,
					  'output': self.output}

		else:
			self.params = {'dim': (self.imgsize, self.imgsize),
					  'batch_size': self.batch_size,
					  'n_channels': 3,
					  'shuffle': True,
					  'randomRotations': False,
					  'seed': 1,
					  'crop': self.crop,
					  'output': self.output}

		self.dataloader()
		if self.load_model < 0:
			# This will result in an error if the wrong architectire is used
			if self.output=='PSTN':
				model_basic = create_model_pstn(self)
				img_in = keras.layers.Input(shape=(self.imgsize, self.imgsize,3), name="img_in")
				model =  model_basic(img_in)
				model_rx = keras.layers.Dense(1, activation = 'linear')(model)
				model_ry = keras.layers.Dense(1, activation = 'linear')(model)
				model_rz = keras.layers.Dense(1, activation = 'softplus')(model)
				model_r =  keras.layers.concatenate([model_rx,model_ry,model_rz])
				model_final =  keras.models.Model(inputs = img_in, outputs=model_r)
				optimizer = keras.optimizers.Adam(lr = self.learning_rate)
				model_final.compile(loss = self.loss_function,
							optimizer = optimizer,
							metrics = [self.metrics_function])
				self.model = model_final
			elif self.output=='ORTN':
				model_basic = create_model_ortn(self)
				img_in = keras.layers.Input(shape=(self.imgsize, self.imgsize,3), name="img_in")
				crop_in = keras.layers.Input(shape=(4,))
				model =  model_basic([img_in,crop_in])
				model = keras.layers.Dense(4, activation = 'tanh')(model)
				model = keras.layers.Lambda(tf.nn.l2_normalize )(model)
				model_final =  keras.models.Model(inputs = [img_in,crop_in], outputs=model)
				optimizer = keras.optimizers.Adam(lr = self.learning_rate)
				model_final.compile(loss = self.loss_function,
							optimizer = optimizer,
							metrics = [self.metrics_function])
				self.model = model_final
			elif self.output=='BOTH':
				model_basic = create_model_ortn(self)
				img_in = keras.layers.Input(shape=(self.imgsize, self.imgsize,3), name="img_in")
				crop_in = keras.layers.Input(shape=(4,))
				model =  model_basic([img_in,crop_in])
				model_rx = keras.layers.Dense(1, activation = 'linear')(model)
				model_ry = keras.layers.Dense(1, activation = 'linear')(model)
				model_rz = keras.layers.Dense(1, activation = 'softplus')(model)
				model_r =  keras.layers.concatenate([model_rx,model_ry,model_rz])
				model_q = keras.layers.Dense(4, activation = 'tanh')(model)
				model_q = keras.layers.Lambda(tf.nn.l2_normalize )(model_q)
				predictions = keras.layers.concatenate([model_q,model_r])
				model_final =  keras.models.Model(inputs = [img_in,crop_in], outputs=predictions)
				optimizer = keras.optimizers.Adam(lr = self.learning_rate)
				model_final.compile(loss = self.loss_function,
							optimizer = optimizer,
							metrics = [self.metrics_function])
				self.model = model_final

		else:
			print(self.load_model)
			self.model = self.gen_output.saveLoadModel(f'Version_{self.load_model}/model_v{self.load_model}_c{self.cluster}_o{self.output}.h5', load=True)


	def evaluate(self, model, dataset, append_submission, dataset_root):

		""" Running evaluation on test set, appending results to a submission. """

		with open(os.path.join(dataset_root, dataset + '.json'), 'r') as f:
			image_list = json.load(f)

		print('Running evaluation on {} set...'.format(dataset))

		for img_id in image_list:
			img_path = os.path.join(dataset_root, 'images', dataset, img_id['filename'])
			img = Image.open(img_path)

			if self.crop:
				img_height, img_width = np.array(img, dtype=float)[:,:].shape[:2]


				# Preprocess image for cropper and obtain corners
				current_image_pil = img.resize((240, 150), resample=Image.BICUBIC)
				current_image_arr = np.array(current_image_pil, dtype=float)/256.

				coordinates = self.cropper_model.predict(np.expand_dims(current_image_arr[:,:],axis=0))


				left=int(np.minimum(coordinates[0,0],coordinates[0,1])*img_width)
				right=int(np.maximum(coordinates[0,0],coordinates[0,1])*img_width)
				lower=int(np.minimum(coordinates[0,2],coordinates[0,3])*img_height)
				upper=int(np.maximum(coordinates[0,2],coordinates[0,3])*img_height)
				len_dif=abs(left-right)-abs(lower-upper)
				if len_dif>0:
					img = img.crop((int(left),int(lower-len_dif/2),int(right),int(upper+len_dif/2)))
				else:
					img = img.crop((int(left+len_dif/2),int(lower),int(right-len_dif/2),int(upper)))

			img = img.resize(self.params["dim"])
			x = image.img_to_array(img)
			#repeat single channel to create three channel image
			if x.shape[2] < 2:
				#only runs on the simulated test set
				x = np.concatenate((x, x, x), axis = 2)
			x = preprocess_input(x)
			x = np.expand_dims(x, 0)

			if self.crop:
				x = [x, coordinates]

			output = model.predict(x)
			append_submission(img_id['filename'], output[0, :4], output[0, 4:])

	def generate_submission(self):
		# Generating submission
		submission = SubmissionWriter()
		self.evaluate(self.model, 'test', submission.append_test, self.dataset_loc)
		self.evaluate(self.model, 'real_test', submission.append_real_test, self.dataset_loc)
		submission.export(out_dir = self.output_loc, suffix = f'v{self.version}')

	def dataloader(self):
		"""
		Initialize the data
		"""
		# Loading and splitting dataset labels
		with open(os.path.join(self.dataset_loc, 'train' + '.json'), 'r') as f:
			label_list = json.load(f)

		# label_list = label_list[:n_imgs]

		if self.cluster not in [0,1,2,3]:
			warnings.warn('No valid cluster has been defined, using the whole dataset')
			#shuffle and split
			train_labels, validation_labels = model_selection.train_test_split(label_list, test_size = self.test_size, shuffle = True)

			# Data generators for training and validation
			self.training_generator = KerasDataGenerator(preprocess_input, train_labels, self.dataset_loc, **self.params)
			self.validation_generator = KerasDataGenerator(preprocess_input, validation_labels, self.dataset_loc, **self.params)
		else:
			#		label_list = label_list[:6000]
			labels_autoenc = np.load('clustering_labels/hierarch_cluster_labels.npy')
			string_process = np.array([x.replace('/data/s1530194/speed/images/','').split('/') for x in labels_autoenc[:,0]])
			img_labels = string_process[:,1]
			set_labels = string_process[:,0]
			clusters_autoenc = labels_autoenc[:,1]
			labels_sel = img_labels[np.logical_and(clusters_autoenc==str(self.cluster),set_labels=='train')]

			#shuffle and split
			train_labels, validation_labels = model_selection.train_test_split(
					labels_sel, test_size = self.test_size, shuffle = True)
			# Data generators for training and validation
			self.training_generator = KerasDataGenerator(preprocess_input, label_list, self.dataset_loc, labels_sel=train_labels, **self.params)
			self.validation_generator = KerasDataGenerator(preprocess_input, label_list, self.dataset_loc, labels_sel=validation_labels, **self.params)


	def train_model(self):
		"""
		Train the model
		"""
		#time training
		starttime = time.time()

		# Choose the early stopping and learning rate configuration
		early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
						patience = 16, verbose = 1,
						restore_best_weights = True, min_delta = 1e-2)
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
						factor = 0.5, patience = 5, verbose = 1,
						min_delta = 1e-2, min_lr = 1e-6)
		# save_model = keras.callbacks.ModelCheckpoint(os.path.join(self.output_loc, 'modelSave_{epoch:04d}-{val_loss:.5f}.h5'),save_best_only=True)

		if self.use_early_stop:
			callbacks=[keras.callbacks.ProgbarLogger(count_mode='steps'),early_stopping,reduce_lr]
		else:
			callbacks=[keras.callbacks.ProgbarLogger(count_mode='steps')]

		# Training the model (transfer learning)
		history = self.model.fit_generator(
			self.training_generator,
			epochs=self.epochs,
			validation_data=self.validation_generator,
			callbacks=callbacks)

		train_duration = time.time() - starttime

		print('\n--------------------------')
		print(f'Training duration: {round(train_duration, 3)} s')

		train_loss = history.history['loss']
		test_loss = history.history['val_loss']

		print('Training losses: ', history.history['loss'])
		print('Validation losses: ', history.history['val_loss'])

		#save and plot losses
		self.gen_output.plot_save_losses(train_loss, test_loss)

		#save the model
		self.gen_output.saveLoadModel(f'{self.output_loc}model_v{self.version}_c{self.cluster}_o{self.output}.h5', model = self.model, save = True)
		self.gen_output.saveLoadModel(f'{self.output_loc}model_v{self.version}_c{self.cluster}_o{self.output}_weights.h5', model = self.model, save = True, weights = True)

		print('--------------------------\n')

	def savePrint(self, s):
		"""
		Write a given Keras model summary to file
		"""

		with open(self.output_loc + self.model_summary_name, 'a') as f:
			print(s, file = f)

	def loss_function(self, y, x):
		"""
		Loss function to be used during training.
		If the option "MSE" is used, it uses mean squared error,
		if the option "MAPE" is usedm it uses the mean average percent error,
		if "POSE" is used, it uses the cost from the competition,
		if "NPOSE" is used, a smoother version of teh scoring function is used.
		"""
		if self.loss == 'MSE':
			return tf.losses.mean_squared_error(x,y)
		elif self.loss == 'MAPE':
			return tf.reduce_mean(tf.abs(tf.divide(tf.subtract(x,y),(y + 1e-10))))
		elif self.loss == 'POSE':
			nn_r=x[:,0:3]
			gt_r=y[:,0:3]
			score_r=tf.reduce_mean(tf.divide(tf.norm(gt_r-nn_r,axis=1),tf.norm(gt_r,axis=1)))
			nn_q_norm=tf.norm(x[:,3:7],axis=1)
			nn_q=tf.stack([tf.divide(x[:,3],nn_q_norm),
				   tf.divide(x[:,4],nn_q_norm),
				   tf.divide(x[:,5],nn_q_norm),
				   tf.divide(x[:,6],nn_q_norm)],
				   axis=1)
			gt_q=y[:,3:7]
			score_q=tf.reduce_mean(tf.abs(2*tf.acos(tf.clip_by_value(tf.tensordot(nn_q,gt_q,[1,1]),-1,1))))
			return tf.add(score_r,score_q)
		elif self.loss == 'NPOSE':
			if self.output=='BOTH':
				nn_r=x[:,0:3]
				gt_r=y[:,0:3]
				score_r=tf.reduce_mean(tf.square(tf.divide(tf.norm(gt_r-nn_r,axis=1),tf.norm(gt_r,axis=1))))
				nn_q_norm=tf.expand_dims(tf.norm(x[:,3:7],axis=1),axis=1)
#				nn_q=tf.stack([tf.divide(x[:,0],nn_q_norm),
#					   tf.divide(x[:,1],nn_q_norm),
#					   tf.divide(x[:,2],nn_q_norm),
#					   tf.divide(x[:,3],nn_q_norm)],
#					   axis=1)
				nn_q=tf.divide(x,nn_q_norm)
				gt_q=y[:,3:7]
				score_q=tf.reduce_mean(tf.square(tf.tensordot(nn_q,gt_q,[1,1])-1))
				return tf.add(score_r,score_q)
			elif self.output=='PSTN':
				nn_r=x
				gt_r=y
#				score_r=tf.reduce_mean(tf.norm(gt_r-nn_r,axis=1))
				score_r=tf.reduce_mean(tf.square(tf.divide(tf.norm(gt_r-nn_r,axis=1),tf.norm(gt_r,axis=1))))
				return score_r
			elif self.output=='ORTN':
				nn_q_norm=tf.expand_dims(tf.norm(x,axis=1),axis=1)
#				nn_q=tf.stack([tf.divide(x[:,0],nn_q_norm),
#					   tf.divide(x[:,1],nn_q_norm),
#					   tf.divide(x[:,2],nn_q_norm),
#					   tf.divide(x[:,3],nn_q_norm)],
#					   axis=1)
				nn_q=tf.divide(x,nn_q_norm)
				gt_q=y
				score_q=tf.reduce_mean(tf.square(tf.tensordot(nn_q,gt_q,[1,1])-1))
				return score_q

		else:
			raise ValueError('The loss "'+self.loss+'" is not a valid loss.')

	def metrics_function(self, y, x):
		"""
		Final score function used in the competition.
		"""
		if self.output=='BOTH':
			nn_r=x[:,0:3]
			gt_r=y[:,0:3]
			score_r=tf.reduce_mean(tf.divide(tf.norm(gt_r-nn_r,axis=1),tf.norm(gt_r,axis=1)))
			nn_q_norm=tf.expand_dims(tf.norm(x[:,3:7],axis=1),axis=1)
#			nn_q=tf.stack([tf.divide(x[:,0],nn_q_norm),
#				   tf.divide(x[:,1],nn_q_norm),
#				   tf.divide(x[:,2],nn_q_norm),
#				   tf.divide(x[:,3],nn_q_norm)],
#				   axis=1)
			nn_q=tf.divide(x,nn_q_norm)
			gt_q=y[:,3:7]
			score_q=tf.reduce_mean(tf.abs(2*tf.acos(tf.clip_by_value(tf.tensordot(nn_q,gt_q,[1,1]),-1,1))))
			return tf.add(score_r,score_q)
		elif self.output=='PSTN':
			nn_r=x
			gt_r=y
			score_r=tf.reduce_mean(tf.divide(tf.norm(gt_r-nn_r,axis=1),tf.norm(gt_r,axis=1)))
#			score_r=tf.reduce_mean(tf.norm(gt_r-nn_r,axis=1))
			return score_r
		elif self.output=='ORTN':
			nn_q_norm=tf.expand_dims(tf.norm(x,axis=1),axis=1)
#			nn_q=tf.stack([tf.divide(x[:,0],nn_q_norm),
#				   tf.divide(x[:,1],nn_q_norm),
#				   tf.divide(x[:,2],nn_q_norm),
#				   tf.divide(x[:,3],nn_q_norm)],
#				   axis=1)
			nn_q=tf.divide(x,nn_q_norm)
			gt_q=y
			score_q=tf.reduce_mean(tf.abs(2*tf.acos(tf.clip_by_value(tf.tensordot(nn_q,gt_q,[1,1]),-1,1))))
			return score_q


def main(batch_size, epochs, version, load_model, loss_function, use_early_stop, crop, cluster,output):

	""" Setting up data generators and model, training, and evaluating model on test and real_test sets. """
	if output=='PSTN':
		for cluster in [0,1,2]:
	#		tf.reset_default_graph()
			#initialize parameters, data loading and the network architecture
			pose = POSE_NN(batch_size, epochs, version, load_model, loss_function, use_early_stop, False, cluster,'PSTN')
			#train the network
			pose.train_model()
	elif output=='ORTN':
		print(f'Running {output} with cluster {cluster}')
#		tf.reset_default_graph()
		#initialize parameters, data loading and the network architecture
		pose = POSE_NN(batch_size, epochs, version, load_model, loss_function, use_early_stop, True, cluster,'ORTN')
		#train the network
		pose.train_model()
	else:
		raise ValueError('Incorrect value for --output parameter. Allowed: "PSTN" or "ORTN"')

	#evaluate the results
#	pose.generate_submission()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
parser.add_argument('--epochs', help='Number of epochs for training.', default = 100)
parser.add_argument('--batch', help='number of samples in a batch.', default = 32)
parser.add_argument('--version', help='version of the neural network.', default = 100)
parser.add_argument('--load', help='load a previously trained network.', default = -1)
parser.add_argument('--loss', help='loss to use. Options: POSE, NPOSE, MAPE, MSE', default = 'NPOSE')
parser.add_argument('--early_stopping', help='use early stopping.', default = True)
parser.add_argument('--crop', help='use crop.', default = True)
parser.add_argument('--cluster', help='Cluster of images to use for training.', default = 2)
parser.add_argument('--output', help='Output in which to train the network. Options: PSTN, ORTN, BOTH.', default = 'ORTN')
args = parser.parse_args()

# Important: If PSTN then crop should be False, if ORTN or BOTH it should be True

main(int(args.batch), int(args.epochs), int(args.version), int(args.load), str(args.loss), bool(args.early_stopping), bool(args.crop), int(args.cluster), str(args.output))
