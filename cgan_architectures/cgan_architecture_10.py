# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:56:24 2019

@author: pablo

Create architecture to define a CGAN
"""


import numpy as np
import keras
import keras.backend as K


def model_generator(latent_dim, input_shape, hidden_dim=1024):
	
	inputs2 = keras.layers.Input(shape=(7,), name="generator_i2")
	in2 = inputs2
	in2 = keras.layers.Dense(7, activation = None, name="generator_d1")(in2)
#	in2 = keras.layers.LeakyReLU(0.2)(in2)
	in2 = keras.layers.ReLU()(in2)
	model_inputs2=keras.models.Model(inputs = inputs2, outputs=in2)
	
	# Noise block
	inputs1 = keras.layers.Input(shape=(latent_dim,), name="generator_i1")
	in1 = inputs1
	in1 = keras.layers.Dense(hidden_dim, activation=None, name='fc1_g')(in1)
#	in1 = keras.layers.LeakyReLU(0.2)(in1)
	in1 = keras.layers.ReLU()(in1)
#	in1 = keras.layers.BatchNormalization()(in1)
	in1 = keras.layers.Dropout(0.1)(in1)
	model_inputs1=keras.models.Model(inputs = inputs1, outputs=in1)
	
	x = keras.layers.concatenate([model_inputs1.output, model_inputs2.output],
							  name="generator_c1")
#	print(input_shape[0]*input_shape[1]*256/(S_T)**2)
#	print((int(input_shape[0]/S_T),int(input_shape[1]/S_T),256))
	x = keras.layers.Dense(5*8*1024, activation=None, name='flatten_g')(x)
	x = keras.layers.Reshape((5,8,1024), name="generator_x")(x)
	x = keras.layers.BatchNormalization(momentum=0.6)(x)
#	x = keras.layers.LeakyReLU(0.2)(x)
	x = keras.layers.ReLU()(x)
	x = keras.layers.Dropout(0.4)(x)
#	x = keras.layers.Lambda(lambda im:K.resize_images(im,b3r[0],b3r[1], 'channels_last'), name='block3_pool_g')(x)
	x = keras.layers.Conv2DTranspose(512, (5, 5),
					  padding='same',
					  strides=(2,2),
					  name='block3_conv1_g')(x)
	x = keras.layers.BatchNormalization(momentum=0.6)(x)
#	x = keras.layers.LeakyReLU(0.2)(x)
	x = keras.layers.ReLU()(x)
#	x = keras.layers.UpSampling2D(size=(4, 4), name='block3_pool_g', interpolation='bilinear')(x)
	
	x = keras.layers.Conv2DTranspose(256, (5, 5),
					  padding='same',
					  strides=(2,2),
					  name='block2_conv2_g')(x)
	x = keras.layers.BatchNormalization(momentum=0.6)(x)
#	x = keras.layers.LeakyReLU(0.2)(x)
	x = keras.layers.ReLU()(x)
	x = keras.layers.Conv2DTranspose(128, (5, 5),
					  padding='same',
#					  strides=(2,2),
					  name='block2_conv1_g')(x)
	x = keras.layers.BatchNormalization(momentum=0.6)(x)
#	x = keras.layers.LeakyReLU(0.2)(x)
	x = keras.layers.ReLU()(x)
#	x = keras.layers.UpSampling2D(size=(4, 4), name='block2_pool_g', interpolation='bilinear')(x)
	x = keras.layers.Conv2DTranspose(64, (5, 5),
					  strides=(2,2),
					  padding='same',
					  name='block1_conv1_g')(x)
	x = keras.layers.BatchNormalization(momentum=0.6)(x)
#	x = keras.layers.LeakyReLU(0.2)(x)
	x = keras.layers.ReLU()(x)
	x = keras.layers.Conv2DTranspose(32, (5, 5),
					  strides=(2,2),
					  padding='same',
					  name='block1_conv0_g')(x)
	x = keras.layers.BatchNormalization(momentum=0.6)(x)
#	x = keras.layers.LeakyReLU(0.2)(x)
	x = keras.layers.ReLU()(x)
	x = keras.layers.Conv2DTranspose(1, (5, 5),
					  strides=(2,2),
					  padding='same',
					  name='block0_conv0_g')(x)
#	x = keras.layers.BatchNormalization(momentum=0.8)(x)
#	x = keras.layers.UpSampling2D(size=(2, 2), name='block2_pool_g', interpolation='bilinear')(x)
#	x = keras.layers.LeakyReLU(0.2)(x)
#	x = keras.layers.ReLU()(x)
#	x = keras.layers.Conv2D(1, (5, 5),
#					  padding='valid',
#					  name='block0_conv1_g')(x)
#	x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Activation('sigmoid')(x)
#	x = keras.layers.Lambda(lambda x:K.resize_images(x,b0r[0],b0r[1], 'channels_last'), name='block0_pool_g')(x)

	# For debugging
#	x = keras.layers.Lambda(lambda x:K.resize_images(x,1,1.77, 'channels_last'), name='block2_pool')(x)
#	x = keras.layers.Reshape(input_shape, name="generator_y")(x)
	model_final = keras.models.Model(
			inputs = [model_inputs1.input, model_inputs2.input],
			outputs= x)
	
	return model_final


def model_discriminator(input_shape, hidden_dim=1024, output_activation="sigmoid", trainable=True):	
	inputs1 = keras.layers.Input(shape=input_shape, name="discriminator_i1")
	# Block 1
	in1 = keras.layers.Conv2D(64, (5, 5),
					  padding='valid',
					  strides=(2,2),
					  name='block1_conv1',
					  trainable=trainable)(inputs1)
	in1 = keras.layers.BatchNormalization()(in1)
	in1 = keras.layers.LeakyReLU(0.2)(in1)
	in1 = keras.layers.Conv2D(128, (5, 5),
					  padding='valid',
					  strides=(2,2),
					  name='block1_conv2',
					  trainable=trainable)(inputs1)
	in1 = keras.layers.BatchNormalization()(in1)
	in1 = keras.layers.LeakyReLU(0.2)(in1)
	# Block 2
	in1 = keras.layers.Conv2D(256, (5, 5),
					  padding='valid',
					  strides=(2,2),
					  name='block2_conv1',
					  trainable=trainable)(in1)
	in1 = keras.layers.BatchNormalization()(in1)
	in1 = keras.layers.LeakyReLU(0.2)(in1)
	in1 = keras.layers.Conv2D(512, (5, 5),
					  padding='valid',
					  strides=(2,2),
					  name='block2_conv2',
					  trainable=trainable)(in1)
	in1 = keras.layers.BatchNormalization()(in1)
	in1 = keras.layers.LeakyReLU(0.2)(in1)
	in1 = keras.layers.Conv2D(1024, (5, 5),
					  padding='valid',
					  strides=(2,2),
					  name='block3_conv1',
					  trainable=trainable)(in1)
	in1 = keras.layers.BatchNormalization()(in1)
	in1 = keras.layers.LeakyReLU(0.2)(in1)
#	in1 = keras.layers.Lambda(lambda x:K.resize_images(x,1,1.777, 'channels_last'), name='block0_pool_g')(in1)
	in1 = keras.layers.Flatten(name='flatten')(in1)
	model_inputs1=keras.models.Model(inputs = inputs1, outputs=in1)
	
	inputs2 = keras.layers.Input(shape=(7,), name="discriminator_i2")
	in2 = inputs2
	in2 = keras.layers.Dense(7, activation = None, name="discriminator_d1", trainable=trainable)(in2)
	in2 = keras.layers.LeakyReLU(0.2)(in2)
	model_inputs2=keras.models.Model(inputs = inputs2, outputs=in2)
	
	x = keras.layers.concatenate([model_inputs1.output, model_inputs2.output],
							  name="discriminator_c1", trainable=trainable)
	# Classification block
	x = keras.layers.Dense(hidden_dim, activation=None, name='fc1',
					  trainable=trainable)(x)
	x = keras.layers.LeakyReLU(0.2)(x)
	x = keras.layers.BatchNormalization()(x)
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
	


