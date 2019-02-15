# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:36:51 2019

@author: pablo
"""
import os
# Use '0' to run on GPU, '-1' on CPU if using CUDA with NVIDIA GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# GPU does not seem to work properly with fir_generator

import json
from keras import backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.initializers import VarianceScaling
from keras.callbacks import EarlyStopping, TensorBoard,ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from utils import KerasDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from submission import SubmissionWriter
from datetime import datetime

# Evaluation from keras_example
def evaluate(model, dataset, append_submission, dataset_root):

    """ Running evaluation on test set, appending results to a submission. """

    with open(os.path.join(dataset_root, dataset + '.json'), 'r') as f:
        image_list = json.load(f)

    print('Running evaluation on {} set...'.format(dataset))

    for img in image_list:
        img_path = os.path.join(dataset_root, 'images', dataset, img['filename'])
        pil_img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(pil_img)
        x = preprocess_input(x)
        x = np.expand_dims(x, 0)
        output = model.predict(x)
        append_submission(img['filename'], output[0, :4], output[0, 4:])

# Make sure no old graphs are kept
K.clear_session()

speed_root='speed'
# Use DEBUG=True to use only 10% of data for debuging
DEBUG=False

# Setting up parameters
params = {'dim': (224, 224),
          'batch_size': 8,
          'n_channels': 3,
          'shuffle': True}

# Loading and splitting dataset
with open(os.path.join(speed_root, 'train' + '.json'), 'r') as f:
    label_list = json.load(f)

if DEBUG:
    train_labels = label_list[:int(len(label_list)*.08)]
    validation_labels = label_list[int(len(label_list)*.08):int(len(label_list)*.1)]
else:
    train_labels = label_list[:int(len(label_list)*.8)]
    validation_labels = label_list[int(len(label_list)*.8):int(len(label_list)*1)]
    

print("Creating generators")
# Data generators for training and validation from the utils file
training_generator = KerasDataGenerator(preprocess_input, train_labels, speed_root, **params)
validation_generator = KerasDataGenerator(preprocess_input, validation_labels, speed_root, **params)


## Load pretrained model of CNN from online version
#from keras.applications.vgg16 import VGG16
#pretrained_model = VGG16(include_top=False,
#                         weights='imagenet', 
#                         input_shape=params["dim"]+(params["n_channels"],),
#                         pooling=None,
#                         classes=None)

# Alternative: load local version (only if the internet connection is very shitty)
pretrained_model = Sequential([
Conv2D(64, (3, 3), input_shape=params["dim"]+(params["n_channels"],), padding='same', activation='relu'),
Conv2D(64, (3, 3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation='relu', padding='same'),
Conv2D(128, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
])
pretrained_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')




# Add some fully connected layers to transform the features into regression
# Xavier's initializer for weights
initializer=VarianceScaling(scale=2.0, mode='fan_avg', distribution='uniform', seed=42) # For ReLU
# Transform output of CNN to a flat layer
pretrained_model.add(Flatten())
# Add some layers to transform the features from CNN to the regression problem
pretrained_model.add(Dense(512, activation="relu",kernel_initializer=initializer))
pretrained_model.add(Dense(128, activation="relu",kernel_initializer=initializer))
pretrained_model.add(Dense(32, activation="relu",kernel_initializer=initializer))
# Output layer
pretrained_model.add(Dense(7, activation="linear",kernel_initializer=initializer))
# Compile with the optimizer selected
optimizer=Adam(lr=1e-3)
pretrained_model.compile(loss='mean_squared_error', optimizer=optimizer)



# Callback to plot while training
tensor_board=TensorBoard(log_dir=datetime.now().strftime("%Y%m%d-%H%M")+'_logs',
                         write_grads=True, write_images=True)

# Choose the early stopping and learning rate configuration
# End optimization if no improvement for 10 epochs
early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1,
                             restore_best_weights=True,min_delta=1e-5)
# Reduce learning rate if no improvement for 4 epochs
reduce_lr=ReduceLROnPlateau(monitor = "val_loss", factor = 0.1,
                              patience = 4, verbose = 1,
                              min_delta = 1e-05,)
# Save best models with loss and epoch in the name
save_model=ModelCheckpoint('modelSave1_{epoch:04d}-{val_loss:.5f}.h5',save_best_only=True)
callbacks=[early_stopping,tensor_board,reduce_lr,save_model]

print("Training model")

# Fit model
history=pretrained_model.fit_generator(training_generator,
                    epochs=10000,initial_epoch=0,
                    validation_data=validation_generator,
                    callbacks=callbacks,
                    workers=0,
                    use_multiprocessing=False,
                    shuffle=True)


print('Training losses: ', history.history['loss'])
print('Validation losses: ', history.history['val_loss'])

# Generating submission
submission = SubmissionWriter()
evaluate(pretrained_model, 'test', submission.append_test, speed_root)
evaluate(pretrained_model, 'real_test', submission.append_real_test, speed_root)
submission.export(suffix='keras_test_1a')











