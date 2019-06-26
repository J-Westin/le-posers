"""
Create in this file your model. Make a new file with a different number
in the filename if you want to drastically change the architecture

Pablo Gomez Perez, 24-3-2019
"""

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

from keras.optimizers import Adam

from keras.utils import plot_model #requires pydot and graphviz

import tensorflow as tf

#things needed for adding layers
# from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Dropout, Activation, Add, MaxPooling2D

import keras

import sys

def create_model(pose):
	"""
	Custom made architecture for position estimation
	"""

	#### This network processes the image
	input = Input(shape=(pose.imgsize, pose.imgsize, pose.3))

	cnn = keras.layers.Conv2D(filters=48, kernel_size=5, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(input)
	cnn = keras.layers.BatchNormalization()(cnn)
	cnn = keras.layers.Activation('relu')(cnn)
	cnn = keras.layers.MaxPooling2D(pool_size=(2,2))(cnn)

	cnn = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(cnn)
	cnn = keras.layers.BatchNormalization()(cnn)
	cnn = keras.layers.Activation('relu')(cnn)
	cnn = keras.layers.MaxPooling2D(pool_size=(2,2))(cnn)

	cnn = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(cnn)
	cnn = keras.layers.BatchNormalization()(cnn)
	cnn = keras.layers.Activation('relu')(cnn)
	cnn = keras.layers.MaxPooling2D(pool_size=(2,2))(cnn)

	cnn = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(cnn)
	cnn = keras.layers.BatchNormalization()(cnn)
	cnn = keras.layers.Activation('relu')(cnn)
	cnn = keras.layers.MaxPooling2D(pool_size=(2,2))(cnn)

	cnn = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(cnn)
	cnn = keras.layers.BatchNormalization()(cnn)
	cnn = keras.layers.Activation('relu')(cnn)
	cnn = keras.layers.MaxPooling2D(pool_size=(2,2))(cnn)

	#flatten the network
	cnn = keras.layers.Flatten()(cnn)
	dnn = keras.layers.Dense(64, activation = 'relu')(cnn)
	dnn = keras.layers.Dropout(0.3)(dnn)
	#output (extra layers are added in the main)
	x = keras.layers.Dense(32, activation = 'relu')(cnn)

	#combine model
	model_final = keras.models.Model(inputs = input, outputs=x)

	#reset summary file
	with open(pose.output_loc + pose.model_summary_name, 'w') as f:
		pass
	#save summary to file
	model_final.summary(print_fn = pose.savePrint)

	#also make a flow chart of the model
	plot_model(model_final, to_file = f'{pose.output_loc}model_arch_v{pose.version}_c{pose.cluster}_o{pose.output}.png', show_shapes = True, show_layer_names = True)

	optimizer = keras.optimizers.Adam(lr = pose.learning_rate, decay = pose.learning_rate_decay)
	model_final.compile(loss = pose.loss_function,
							optimizer = optimizer,
							metrics = [pose.metrics_function])

	return model_final
