import json
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import keras
from sklearn import model_selection

import numpy as np
import os
import argparse
import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from pose_submission import SubmissionWriter
from pose_utils import KerasDataGenerator, checkFolders, OutputResults
from pose_architecture_3 import create_model


""" 
	Example script demonstrating training on the SPEED dataset using Keras.
	Usage example: python keras_example.py --epochs [num epochs] --batch [batch size]
"""

class POSE_NN(object):

	def __init__(self, batch_size, epochs, version, load_model, loss="mse"):
		#### tweakable parameters
		self.batch_size = batch_size
		self.epochs = epochs
		self.version = version
		self.load_model = load_model
		self.loss = loss

		self.imgsize = 224
		#size of the test set as a fraction of the total amount of data
		self.test_size = 0.1

		#dropout percentage
		self.dropout = 0.3

		self.learning_rate = 0.0001
		self.learning_rate_decay = 0


		self.params = {'dim': (self.imgsize, self.imgsize),
				  'batch_size': self.batch_size,
				  'n_channels': 3,
				  'shuffle': True,
				  'randomRotations': True,
				  'seed': 1}

		#### constant parameters
		self.dataset_loc = '../../speed'



		self.output_loc = f'./Version_{self.version}/'
		self.model_summary_name = f'model_summary_v{self.version}.txt'

		#### initialize some stuff
		#check if folders are present and make them if necessary
		checkFolders([self.output_loc])

		self.gen_output = OutputResults(self)

		self.dataloader()
		if self.load_model < 0:
			self.model = create_model(self)
		else:
			self.model = self.gen_output.saveLoadModel(f'Version_{self.load_model}/model_v{self.load_model}.h5', load=True)


	def evaluate(self, model, dataset, append_submission, dataset_root):

		""" Running evaluation on test set, appending results to a submission. """

		with open(os.path.join(dataset_root, dataset + '.json'), 'r') as f:
			image_list = json.load(f)

		print('Running evaluation on {} set...'.format(dataset))

		for img in image_list:
			img_path = os.path.join(dataset_root, 'images', dataset, img['filename'])
			pil_img = image.load_img(img_path, target_size=(self.imgsize, self.imgsize))
			x = image.img_to_array(pil_img)
			x = preprocess_input(x)
			x = np.expand_dims(x, 0)
			output = model.predict(x)
			append_submission(img['filename'], output[0, :4], output[0, 4:])

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

		#shuffle and split
		train_labels, validation_labels = model_selection.train_test_split(label_list, test_size = self.test_size, shuffle = True)

		# Data generators for training and validation
		self.training_generator = KerasDataGenerator(preprocess_input, train_labels, self.dataset_loc, **self.params)
		self.validation_generator = KerasDataGenerator(preprocess_input, validation_labels, self.dataset_loc, **self.params)

	def train_model(self):
		"""
		Train the model
		"""
		#time training
		starttime = time.time()

		# Training the model (transfer learning)
		history = self.model.fit_generator(
			self.training_generator,
			epochs=self.epochs,
			validation_data=self.validation_generator,
			callbacks=[keras.callbacks.ProgbarLogger(count_mode='steps')])

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
		self.gen_output.saveLoadModel(f'{self.output_loc}model_v{self.version}.h5', model = self.model, save = True)

		print('--------------------------\n')

	def savePrint(self, s):
		"""
		Write a given Keras model summary to file
		"""
		
		with open(self.output_loc + self.model_summary_name, 'a') as f:
			print(s, file = f) 
			
	def loss_function(self, x, y):
		"""
		Loss function to be used during training. 
		If the option "mse" is used, it uses mean squared error,
		if "pose" is used, it uses the cost from the competition.
		"""
		if self.loss=="mse":
			return tf.losses.mean_squared_error(x,y)
		elif self.loss=="pose":
			nn_r=x[:,0:2]
			gt_r=y[:,0:2]
			score_r=tf.reduce_mean(tf.divide(tf.norm(gt_r-nn_r,axis=1),tf.norm(gt_r,axis=1)))
			nn_q=tf.nn.l2_normalize(x[:,3:6],axis=1)
			gt_q=y[:,3:6]
			score_q=tf.reduce_mean(2*tf.acos(tf.clip_by_value(tf.tensordot(nn_q,gt_q,[1,1]),-1,1)))
			return tf.add(score_r,score_q)
		else:
			raise ValueError('The loss '+self.loss+' is not a valid loss.')
        

def main(batch_size, epochs, version, load_model, loss_function):

	""" Setting up data generators and model, training, and evaluating model on test and real_test sets. """

	#initialize parameters, data loading and the network architecture
	pose = POSE_NN(batch_size, epochs, version, load_model, loss_function)

	#train the network
	pose.train_model()

	#evaluate the results
	pose.generate_submission()
	

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
parser.add_argument('--epochs', help='Number of epochs for training.', default = 20)
parser.add_argument('--batch', help='number of samples in a batch.', default = 8)
parser.add_argument('--version', help='version of the neural network.', default = 0)
parser.add_argument('--load', help='load a previously trained network.', default = -1)
parser.add_argument('--loss', help='loss to use.', default = "pose")
args = parser.parse_args()

main(int(args.batch), int(args.epochs), int(args.version), int(args.load), str(args.loss))


'''
https://machinelearningmastery.com/save-load-keras-deep-learning-models/

Remove last layer of model with 
model.layers.pop()

Freeze layers
for layer in model.layers:
    layer.trainable = False
'''
