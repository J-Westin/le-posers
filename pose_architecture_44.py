"""
Create in this file your model. Make a new file with a different number
in the filename if you want to drastically change the architecture

Pablo Gomez Perez, 24-3-2019
"""

from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing import image

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.utils.vis_utils import plot_model #requires pydot and graphviz

import tensorflow as tf

#things needed for adding layers
# from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Dropout, Activation, Add, MaxPooling2D

import keras

import sys

def create_model(pose):
	"""
	An architecture based on VGG19. Best results are obtained when using
	all the layers except the output layer of VGG16. 
	This specific version is linear, so NO SKIP CONNECTION.
	"""
	#number of layers to use from the pretrained network
	# nl_from_pre = 8


	# Loading and freezing pre-trained model
	keras.backend.set_learning_phase(0)
	pretrained_model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(pose.imgsize, pose.imgsize, 3))

	#unfreeze the rest of the model
	keras.backend.set_learning_phase(1)

	# print('\nSaving model plot...\n')
	# plot_model(pretrained_model, to_file='VGG19_arch.png', show_shapes=True, show_layer_names=True)
	# sys.exit('Stopped')
	
	'''
	#find the number of layers of the pretrained network
	nl_pre = len(pretrained_model.layers)

	#pop all layers until you end up at the desired number
	#if a layer splits, both sides also add up to the layer count
	test_model = pretrained_model
	for i in reversed(range(nl_from_pre+1, nl_pre)):
		#pop one layer at a time
		test_model.layers.pop()
	'''

	# Adding new trainable hidden and output layers to the model
	x = pretrained_model.layers[-1].output

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
	
	x = keras.layers.Conv2D(filters=64, kernel_size=5, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)

	x = keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid',
					 kernel_initializer='glorot_uniform', use_bias=True)(x)
	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('relu')(x)
'''

	#pool across each kernel layer
#	x = keras.layers.GlobalAveragePooling2D()(x)
	


	#now flatten
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(128, activation = 'relu')(x)
	# x = keras.layers.Dropout(pose.dropout)(x)

	#output layer
	x = keras.layers.Dense(64, activation = 'relu')(x)
	x = keras.layers.Dense(32, activation = 'relu')(x)

	#combine model
	model_final = keras.models.Model(inputs = pretrained_model.input, outputs=x)

	#reset summary file
	with open(pose.output_loc + pose.model_summary_name, 'w') as f:
		pass
	#save summary to file
	model_final.summary(print_fn = pose.savePrint)

	#also make a flow chart of the model
	plot_model(model_final, to_file = f'{pose.output_loc}model_arch_v{pose.version}_c{pose.cluster}_o{pose.output}.png', show_shapes = True, show_layer_names = True)

	print(pose.learning_rate)
	model_final.compile(
			loss = pose.loss_function, 
			optimizer = Adam(lr = pose.learning_rate,decay = pose.learning_rate_decay),
			metrics = [pose.metrics_function]
			)

	return model_final