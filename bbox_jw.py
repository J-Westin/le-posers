import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import keras

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from pose_utils import KerasDataGenerator, SatellitePoseEstimationDataset, Camera

## ~~ Settings ~~ ##
#  Change these to match your setup

dataset_dir = "..\\speed"      # Root directory of dataset (contains /images, LICENSE.MD, *.json)
default_margins = (.2, .2, .1) # (x, y, z) offset between satellite body and rectangular cage used as cropping target

recreate_json    = True # Creates a new JSON file with bbox training label even if one already exists
#recreate_network = True # Creates a new neural network even if one has already been saved

train_dir = os.path.join(dataset_dir, "images\\train")

def create_bbox_json(margins=default_margins):
    bbox_json_filepath_expected = os.path.join(dataset_dir, "train_bbox.json")
    
    if recreate_json or (not os.path.exists(bbox_json_filepath_expected)):
        my_dataset = SatellitePoseEstimationDataset(root_dir=dataset_dir)
        my_dataset.generate_bbox_json(margins)
    
    bbox_json_file = open(bbox_json_filepath_expected, 'r')
    bbox_training_labels = json.load(bbox_json_file)
    bbox_json_file.close()
    
    return bbox_training_labels



def create_bb_model():
    bb_model = Sequential()
    
    bb_model.add(Conv2D(filters=32, kernel_size=5, input_shape=(int(Camera.nv/16), int(Camera.nu/16), 1), padding='valid', use_bias=True))
    bb_model.add(BatchNormalization())
    bb_model.add(Activation('softmax'))
    bb_model.add(MaxPooling2D(pool_size=(2,2)))
    
    bb_model.add(Conv2D(filters=32, kernel_size=5, padding='valid', use_bias=True))
    bb_model.add(BatchNormalization())
    bb_model.add(Activation('softmax'))
    bb_model.add(MaxPooling2D(pool_size=(2,2)))
    
    bb_model.add(Conv2D(filters=16, kernel_size=5, padding='valid', use_bias=True))
    bb_model.add(BatchNormalization())
    bb_model.add(Activation('softmax'))
    bb_model.add(MaxPooling2D(pool_size=(2,2)))
    
    
    
    # for k in range(2,7):
        # bb_model.add(BatchNormalization())
        # bb_model.add(Activation('softmax'))
        # bb_model.add(MaxPooling2D(pool_size=(2,2)))
        # bb_model.add(Conv2D(filters=32, kernel_size=5, padding='valid', use_bias=True))
    
    # bb_model.add(BatchNormalization())
    # bb_model.add(Activation('softmax'))
    
    
    
    bb_model.add(Flatten())
    
    intermediate_sizes = [400, 200, 100, 50, 20, 10, 6, 4]
    
    for layersize in intermediate_sizes:
        bb_model.add(Dense(units=layersize, activation="softmax"))
    
    return bb_model




def train_bb_model(n_images, model):
    labels = create_bbox_json()
    
    image_array = []
    label_array = []
    
    for _ in range(n_images):
        current_label = random.choice(labels)
        current_filename = current_label["filename"]
        current_bbox = np.array([current_label["bbox"]])
        
        label_array.append(current_label["bbox"])
        
        current_filepath = os.path.join(train_dir, current_filename)
        current_image_pil = Image.open(current_filepath)
        current_image_pil = current_image_pil.resize((120, 75), resample=Image.BICUBIC)
        current_image_arr = np.array(current_image_pil, dtype=float)/256.
        current_image_pil.close()
        
        #current_image_arr = np.array([np.expand_dims(current_image_arr, 2)])
        
        image_array.append(np.expand_dims(current_image_arr, 2))
        
    model.fit(np.array(image_array), np.array(label_array))

create_bbox_json()

try:
    print ("Loading architecture from file")
    architecture_file = open("bbox_jw_model\\model.json", "r")
    bb_model = keras.models.model_from_json(architecture_file.read())
    architecture_file.close()
    bb_model.load_weights("bbox_jw_model\\weights.h5")
    print (" Successfully loaded architecture from file!")
except:
    print ("Failed to load architecture from file, creating new model instead")
    bb_model = create_bb_model()
    print (" Created new model")

bb_model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')

for k in range(1,11):
    print ("Commencing round", k)
    train_bb_model(1000, bb_model)
    
    print ("Saving weights")
    bb_model.save_weights("bbox_jw_model\\weights.h5")
    print (" Saving weights successful")
    
    print ("Saving model")
    architecture_file = open("bbox_jw_model\\model.json", "w")
    architecture_file.write(bb_model.to_json())
    architecture_file.close()
    print (" Saving model successful\n")