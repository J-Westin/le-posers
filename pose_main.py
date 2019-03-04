import json
import tensorflow
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import keras
from sklearn import model_selection

import numpy as np
import os
import argparse

from pose_submission import SubmissionWriter
from pose_utils import KerasDataGenerator, checkFolders
from pose_architecture_1 import create_model


""" 
	Example script demonstrating training on the SPEED dataset using Keras.
	Usage example: python keras_example.py --epochs [num epochs] --batch [batch size]
"""

class POSE_NN(object):

	def __init__(self, batch_size, epochs, version):
		#### tweakable parameters
		self.batch_size = batch_size
		self.epochs = epochs
		self.version = version

		self.imgsize = 224
		#size of the test set as a fraction of the total amount of data
		self.test_size = 0.2

		#dropout percentage
		self.dropout = 0.3


		self.params = {'dim': (self.imgsize, self.imgsize),
				  'batch_size': batch_size,
				  'n_channels': 3,
				  'shuffle': True}

		#### constant parameters
		self.dataset_loc = '/data/s1530194/speed'

		self.output_loc = f'./Version_{self.version}/'
		self.model_summary_name = 'model_summary.txt'

		#### initialize some stuff
		#make folders if necessary
		checkFolders([self.output_loc])

		self.dataloader()
		self.model = create_model(self)

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
		submission.export(suffix=f'{self.output_loc}keras_example')

	def dataloader(self):
		"""
		Initialize the data
		"""
		# Loading and splitting dataset labels
		with open(os.path.join(self.dataset_loc, 'train' + '.json'), 'r') as f:
			label_list = json.load(f)

		#shuffle and split
		train_labels, validation_labels = model_selection.train_test_split(label_list, test_size = self.test_size, shuffle = True)

		# train_labels = label_list[:int(len(label_list)*.8)]
		# validation_labels = label_list[int(len(label_list)*.8):]

		# Data generators for training and validation
		self.training_generator = KerasDataGenerator(preprocess_input, train_labels, self.dataset_loc, **self.params)
		self.validation_generator = KerasDataGenerator(preprocess_input, validation_labels, self.dataset_loc, **self.params)

	def train_model(self):
		"""
		Train the model
		"""
		# Training the model (transfer learning)
		history = self.model.fit_generator(
			self.training_generator,
			epochs=self.epochs,
			validation_data=self.validation_generator,
			callbacks=[keras.callbacks.ProgbarLogger(count_mode='steps')])

		print('Training losses: ', history.history['loss'])
		print('Validation losses: ', history.history['val_loss'])

	def savePrint(self, s):
		"""
		Write a given Keras model summary to file
		"""
		
		with open(self.model_summary_loc + self.model_summary_name, 'a') as f:
			print(s, file = f) 

def main(batch_size, epochs, version):

	""" Setting up data generators and model, training, and evaluating model on test and real_test sets. """

	#initialize parameters, data loading and the network architecture
	pose = POSE_NN(batch_size, epochs, version)

	#train the network
	pose.train_model()

	#evaluate the results
	pose.generate_submission()
	

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
parser.add_argument('--epochs', help='Number of epochs for training.', default = 20)
parser.add_argument('--batch', help='number of samples in a batch.', default = 32)
parser.add_argument('--version', help='version of the neural network.', default = 0)
args = parser.parse_args()

main(int(args.batch), int(args.epochs), int(args.version))


'''
https://machinelearningmastery.com/save-load-keras-deep-learning-models/

Remove last layer of model with 
model.layers.pop()
'''