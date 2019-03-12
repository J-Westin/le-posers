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
	#number of layers to use from the pretrained network
	nl_from_pre = 8


	# Loading and freezing pre-trained model
	keras.backend.set_learning_phase(0)
	pretrained_model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(pose.imgsize, pose.imgsize, 3))

	#unfreeze the rest of the model
	keras.backend.set_learning_phase(1)

	# print('\nSaving model plot...\n')
	# plot_model(pretrained_model, to_file='VGG19_arch.png', show_shapes=True, show_layer_names=True)
	# sys.exit('Stopped')
	
	
	#find the number of layers of the pretrained network
	nl_pre = len(pretrained_model.layers)

	#pop all layers until you end up at the desired number
	#if a layer splits, both sides also add up to the layer count
	test_model = pretrained_model
	for i in reversed(range(nl_from_pre+1, nl_pre)):
		#pop one layer at a time
		test_model.layers.pop()
	

	# Adding new trainable hidden and output layers to the model
	x = test_model.layers[-1].output

	'''
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', 
					 activation='relu', use_bias=True)(x)
	x = keras.layers.Dropout(pose.dropout)(x)
	x = keras.layers.Activation('relu')(x)

	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	'''
	'''
	#make a second connection which quickly downsamples
	y = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', 
					 activation='relu', use_bias=True)(pretrained_model.input)
	y = keras.layers.Dropout(pose.dropout)(y)
	y = keras.layers.MaxPooling2D(pool_size=(3,3))(y)

	y = keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', 
					 activation='relu', use_bias=True)(y)
	y = keras.layers.Dropout(pose.dropout)(y)
	y = keras.layers.MaxPooling2D(pool_size=(3,3))(y)

	y = keras.layers.Conv2D(filters=512, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', 
					 activation='relu', use_bias=True)(y)
	y = keras.layers.Dropout(pose.dropout)(y)
	y = keras.layers.MaxPooling2D(pool_size=(3,3))(y)



	#add layers to form a skip connection
	x = keras.layers.Add()([pre_x, y])
	'''

	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
	x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	# x = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
	# 				 kernel_initializer='glorot_uniform', 
	# 				 activation='relu', use_bias=True)(x)
	# x = keras.layers.Dropout(pose.dropout)(x)
	# x = keras.layers.Activation('relu')(x)
	# x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	#now flatten
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(15, activation = 'relu')(x)
	x = keras.layers.Dropout(pose.dropout)(x)

	#output layer
	predictions = keras.layers.Dense(7, activation='linear')(x)

	#combine model
	model_final = keras.models.Model(inputs=pretrained_model.input, outputs=predictions)

	#reset summary file
	with open(pose.output_loc + pose.model_summary_name, 'w') as f:
		pass
	#save summary to file
	model_final.summary(print_fn = pose.savePrint)

	#also make a flow chart of the model
	plot_model(model_final, to_file=f'{pose.output_loc}model_arch_v{pose.version}.png', show_shapes=True, show_layer_names=True)

	model_final.compile(loss='mean_squared_error', 
				optimizer=Adam(lr = pose.learning_rate))

	return model_final