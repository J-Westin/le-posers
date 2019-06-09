# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:56:24 2019

@author: pablo

Create architecture to define a CGAN
"""


import numpy as np
import keras


def model_generator(latent_dim, input_shape, hidden_dim=1024):
	inputs1 = keras.layers.Input(shape=(latent_dim,), name="generator_i1")
	in1 = inputs1
	in1 = keras.layers.Dense(int(hidden_dim / 2), name="generator_h1",
						  activation = 'relu')(in1)
	model_inputs1=keras.models.Model(inputs = inputs1, outputs=in1)
	
	inputs2 = keras.layers.Input(shape=(7,), name="generator_i2")
	in2 = inputs2
	in2 = keras.layers.Dense(7, activation = 'relu', name="generator_d1")(in2)
	model_inputs2=keras.models.Model(inputs = inputs2, outputs=in2)
	
	x = keras.layers.concatenate([model_inputs1.output, model_inputs2.output],
							  name="generator_c1")
	x = keras.layers.Dense(int(hidden_dim / 2), name="generator_h2",
						activation = 'relu')(x)
	x = keras.layers.Dense(hidden_dim, name="generator_h3",
						activation = 'relu')(x)
	x = keras.layers.Dense(np.prod(input_shape), name="generator_x_flat",
						activation = 'sigmoid')(x)
	x = keras.layers.Reshape(input_shape, name="generator_x")(x)
	
	model_final = keras.models.Model(
			inputs = [model_inputs1.input, model_inputs2.input],
			outputs= x)
	
	return model_final


def model_discriminator(input_shape, hidden_dim=1024, output_activation="sigmoid", trainable=True):
	inputs1 = keras.layers.Input(shape=input_shape, name="discriminator_i1")
	in1 = inputs1
	in1 = keras.layers.Flatten(name="discriminator_flatten", trainable=trainable)(in1)
	in1 = keras.layers.Dense(hidden_dim, name="discriminator_h1",
						  activation = 'relu', trainable=trainable)(in1)
	in1 = keras.layers.Dense(int(hidden_dim / 2), name="discriminator_h2",
						activation = 'relu', trainable=trainable)(in1)
	model_inputs1=keras.models.Model(inputs = inputs1, outputs=in1)
	
	inputs2 = keras.layers.Input(shape=(7,), name="discriminator_i2")
	in2 = inputs2
	in2 = keras.layers.Dense(7, activation = 'relu', trainable=trainable)(in2)
	model_inputs2=keras.models.Model(inputs = inputs2, outputs=in2)
	
	
	x = keras.layers.concatenate([model_inputs1.output, model_inputs2.output],
							  name="discriminator_c1", trainable=trainable)
	x = keras.layers.Dense(int(hidden_dim / 2), name="generator_h3",
						activation = 'relu', trainable=trainable)(x)
	x = keras.layers.Dense(1, name="discriminator_y",
						activation = output_activation, trainable=trainable)(x)
	model_final = keras.models.Model(
			inputs = [model_inputs1.input, model_inputs2.input],
			outputs=x)
	
	return model_final

def model_total(latent_dim, input_shape, hidden_dim=1024,
				output_activation="sigmoid"):
	inputnoise = keras.layers.Input(shape=(latent_dim,), name="total_i1")
	inputpose = keras.layers.Input(shape=(7,), name="total_i2")
	inputg = keras.layers.concatenate([inputnoise, inputpose], name="total_c1")
	generator = model_generator(latent_dim, input_shape, hidden_dim)([inputnoise,inputpose])
	keras.backend.set_learning_phase(0)
	discriminator = model_discriminator(input_shape, hidden_dim,
									 output_activation=output_activation,
									 trainable=False)([generator, inputpose])

	#unfreeze the rest of the model
	keras.backend.set_learning_phase(1)
	model_final =  keras.models.Model(inputs = [inputnoise,inputpose], outputs=discriminator)
	
	return model_final
	


