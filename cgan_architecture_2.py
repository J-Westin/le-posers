# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:56:24 2019

@author: pablo

Create architecture to define a CGAN
"""


import numpy as np
import keras
import keras.backend as K

S_0 = 1
S_1 = 4
S_2 = 2
S_3 = 2

def model_generator(latent_dim, input_shape, hidden_dim=1024):
#	b0r = (input_shape[0]/(int(input_shape[0]/S_0)),input_shape[1]/(int(input_shape[1]/S_0)))
	b0r = (1,1)
	b1c1 = ((input_shape[0]/b0r[0]-3)/1+1,(input_shape[1]/b0r[1]-3)/1+1,input_shape[2])
	b2c1 = ((b1c1[0]-3)/1+1,(b1c1[1]-3)/1+1,64)
	b3c1 = ((b2c1[0]-3)/1+1,(b2c1[1]-3)/1+1,128)
	d1r = (int(b3c1[0]),int(b3c1[1]),int(b3c1[2]))
	
	
	inputs2 = keras.layers.Input(shape=(7,), name="generator_i2")
	in2 = inputs2
	in2 = keras.layers.Dense(7, activation = 'relu', name="generator_d1")(in2)
	model_inputs2=keras.models.Model(inputs = inputs2, outputs=in2)
	# Noise block
	inputs1 = keras.layers.Input(shape=(latent_dim,), name="generator_i1")
	in1 = inputs1
	in1 = keras.layers.Dense(hidden_dim, activation='relu', name='fc1_g')(in1)
	in1 = keras.layers.Dense(hidden_dim, activation='relu', name='fc2_g')(in1)
	model_inputs1=keras.models.Model(inputs = inputs1, outputs=in1)
	
	x = keras.layers.concatenate([model_inputs1.output, model_inputs2.output],
							  name="generator_c1")
	
	x = keras.layers.Dense(d1r[0]*d1r[1]*d1r[2], activation='relu', name='flatten_g')(x)
	x = keras.layers.Reshape(d1r, name="generator_x")(x)
	x = keras.layers.Conv2DTranspose(128, (3, 3),
					  activation='relu',
					  padding='valid',
					  name='block3_conv1_g')(x)
	x = keras.layers.Conv2DTranspose(64, (3, 3),
					  activation='relu',
					  padding='valid',
					  name='block2_conv1_g')(x)
	x = keras.layers.Conv2DTranspose(1, (3, 3),
					  activation='relu',
					  padding='valid',
					  name='block1_conv1_g')(x)
#	x = keras.layers.Lambda(lambda x:K.resize_images(x,b0r[0],b0r[1], 'channels_last'), name='block0_pool_g')(x)
#	print(b0r)
#	print(b1c1)
#	print(input_shape)
#	x = keras.layers.Lambda(lambda x:K.resize_images(x,1.77,1.77, 'channels_last'), name='block2_pool')(x)
	x = keras.layers.Reshape(input_shape, name="generator_y")(x)
	model_final = keras.models.Model(
			inputs = [model_inputs1.input, model_inputs2.input],
			outputs= x)
	
	return model_final


def model_discriminator(input_shape, hidden_dim=1024, output_activation="sigmoid", trainable=True):	
	inputs1 = keras.layers.Input(shape=input_shape, name="discriminator_i1")
#	in1 = keras.layers.MaxPooling2D((S_0, S_0), strides=(S_0, S_0), name='block0_pool',
#					  trainable=trainable)(inputs1)
	# Block 1
	in1 = keras.layers.Conv2D(64, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block1_conv1',
					  trainable=trainable)(inputs1)
	in1 = keras.layers.Conv2D(128, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block2_conv1',
					  trainable=trainable)(in1)
	in1 = keras.layers.Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv4',
					  trainable=trainable)(in1)
	in1 = keras.layers.Flatten(name='flatten')(in1)
	model_inputs1=keras.models.Model(inputs = inputs1, outputs=in1)
	
	inputs2 = keras.layers.Input(shape=(7,), name="discriminator_i2")
	in2 = inputs2
	in2 = keras.layers.Dense(7, activation = 'relu', name="discriminator_d1", trainable=trainable)(in2)
	model_inputs2=keras.models.Model(inputs = inputs2, outputs=in2)
	
	x = keras.layers.concatenate([model_inputs1.output, model_inputs2.output],
							  name="discriminator_c1", trainable=trainable)
	# Classification block
	x = keras.layers.Dense(hidden_dim, activation='relu', name='fc1',
					  trainable=trainable)(x)
	x = keras.layers.Dense(hidden_dim, activation='relu', name='fc2',
					  trainable=trainable)(x)
	x = keras.layers.Dense(1, activation=output_activation, name='predictions',
					  trainable=trainable)(x)
	model_final = keras.models.Model(
			inputs = [model_inputs1.input, model_inputs2.input],
			outputs=x)

	
	return model_final

def model_total(latent_dim, input_shape, hidden_dim=1024,
				output_activation="sigmoid"):
	inputnoise = keras.layers.Input(shape=(latent_dim,), name="total_i1")
	inputpose = keras.layers.Input(shape=(7,), name="total_i2")
	generator = model_generator(latent_dim, input_shape, hidden_dim)([inputnoise,inputpose])
	keras.backend.set_learning_phase(0)
	discriminator = model_discriminator(input_shape, hidden_dim,
									 output_activation=output_activation,
									 trainable=False)([generator, inputpose])

	#unfreeze the rest of the model
	keras.backend.set_learning_phase(1)
	model_final =  keras.models.Model(inputs = [inputnoise,inputpose], outputs=discriminator)
	
	return model_final
	


