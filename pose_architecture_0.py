"""
Create in this file your model. Make a new file with a different number
in the filename if you want to drastically change the architecture

Jelle Mes, 25-2-2019
"""

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import keras

def create_model():
	# Loading and freezing pre-trained model
	keras.backend.set_learning_phase(0)
	pretrained_model = keras.applications.ResNet50(weights='imagenet', 
										include_top=False,
										input_shape=(224, 224, 3))

	# Adding new trainable hidden and output layers to the model
	keras.backend.set_learning_phase(1)
	x = pretrained_model.output
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(1024, activation='relu')(x)
	predictions = keras.layers.Dense(7, activation='linear')(x)
	model_final = keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
	model_final.compile(loss='mean_squared_error', optimizer='adam')

	return model_final