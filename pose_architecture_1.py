"""
Create in this file your model. Make a new file with a different number
in the filename if you want to drastically change the architecture

Jelle Mes, 25-2-2019


This architecture uses the first two layers of ResNet and builds on this
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
	# Loading and freezing pre-trained model
	keras.backend.set_learning_phase(0)
	pretrained_model = keras.applications.ResNet50(weights='imagenet', 
										include_top=False,
										input_shape=(pose.imgsize, pose.imgsize, 3))
	keras.backend.set_learning_phase(1)
	
	#find the number of layers of the pretrained network
	nl_pre = len(pretrained_model.layers)

	#pop all layers until you end up at the desired number
	#if a layer splits, both sides also add up to the layer count
	nl_from_pre = 17 
	test_model = pretrained_model
	for i in reversed(range(nl_from_pre+1, nl_pre)):
		#pop one layer at a time
		test_model.layers.pop()

	plot_model(test_model, to_file='resnet_first_15_layers.png', show_shapes=True, show_layer_names=True)

	# Adding new trainable hidden and output layers to the model
	# x = test_model.output
	x = test_model.layers[-1].output

	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=24, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', 
					 activation='relu', use_bias=True)(x)
	x = keras.layers.Dropout(pose.dropout)(x)
	x = keras.layers.Activation('relu')(x)

	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', 
					 activation='relu', use_bias=True)(x)
	x = keras.layers.Dropout(pose.dropout)(x)
	x = keras.layers.Activation('relu')(x)

	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)


	#now flatten
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(15, activation='relu')(x)
	x = keras.layers.Dropout(pose.dropout)(x)

	#output layer
	predictions = keras.layers.Dense(7, activation='linear')(x)

	#combine model
	model_final = keras.models.Model(inputs=test_model.input, outputs=predictions)

	#make a flow chart of the model
	plot_model(model_final, to_file=f'{pose.output_loc}model_arch_v{pose.version}.png', show_shapes=True, show_layer_names=True)

	model_final.compile(loss='mean_squared_error', 
				optimizer=Adam(lr = pose.learning_rate))

	return model_final