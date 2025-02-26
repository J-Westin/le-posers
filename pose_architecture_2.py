"""
Create in this file your model. Make a new file with a different number
in the filename if you want to drastically change the architecture

Jelle Mes, 25-2-2019
"""

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

from keras.optimizers import Adam

from keras.utils.vis_utils import plot_model #requires pydot and graphviz

#things needed for adding layers
# from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Dropout, Activation, Add, MaxPooling2D

import keras

import sys

def create_model(pose):
	"""
	A hand designed network that needs to be trained completely
	"""
	input = keras.layers.Input(shape=(pose.imgsize, pose.imgsize, 3))

	x = keras.layers.Conv2D(filters=64, kernel_size=7, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(input)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=64, kernel_size=5, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=64, kernel_size=5, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=32, kernel_size=5, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)


	#now flatten
	# x = keras.layers.Flatten()(x)
	# x = keras.layers.Dense(15, activation = 'relu')(x)
	# x = keras.layers.Dropout(pose.dropout)(x)

	x = keras.layers.GlobalAveragePooling2D()(x)

	#output layer
	predictions = keras.layers.Dense(7, activation = 'linear')(x)

	#combine model
	model_final = keras.models.Model(inputs = input, outputs = predictions)

	#reset summary file
	with open(pose.output_loc + pose.model_summary_name, 'w') as f:
		pass
	#save summary to file
	model_final.summary(print_fn = pose.savePrint)

	#also make a flow chart of the model
	plot_model(model_final, to_file = f'{pose.output_loc}model_arch_v{pose.version}.png', show_shapes = True, show_layer_names = True)

	model_final.compile(loss = pose.loss_function, 
							optimizer = Adam(lr = pose.learning_rate,
							decay = pose.learning_rate_decay))

	return model_final