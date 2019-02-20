import json
import tensorflow
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import keras
import numpy as np
import os
from sklearn import model_selection

from pose_submission import SubmissionWriter
from pose_utils import KerasDataGenerator


""" 
    Example script demonstrating training on the SPEED dataset using Keras.
    Usage example: python keras_example.py --dataset [path to speed] --epochs [num epochs] --batch [batch size]
"""


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


def main(speed_root, epochs, batch_size):

    """ Setting up data generators and model, training, and evaluating model on test and real_test sets. """

    # Setting up parameters
    params = {'dim': (224, 224),
              'batch_size': batch_size,
              'n_channels': 3,
              'shuffle': True}

    # Loading and splitting dataset
    with open(os.path.join(speed_root, 'train' + '.json'), 'r') as f:
        label_list = json.load(f)

    #shuffle and split
    train_labels, validation_labels = model_selection.train_test_split(label_list, test_size = 0.2, shuffle = True)

    # train_labels = label_list[:int(len(label_list)*.8)]
    # validation_labels = label_list[int(len(label_list)*.8):]

    # Data generators for training and validation
    training_generator = KerasDataGenerator(preprocess_input, train_labels, speed_root, **params)
    validation_generator = KerasDataGenerator(preprocess_input, validation_labels, speed_root, **params)

    # Loading and freezing pre-trained model
    keras.backend.set_learning_phase(0)
    pretrained_model = keras.applications.ResNet50(weights="imagenet", include_top=False,
                                                              input_shape=(224, 224, 3))

    # Adding new trainable hidden and output layers to the model
    keras.backend.set_learning_phase(1)
    x = pretrained_model.output
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    predictions = keras.layers.Dense(7, activation="linear")(x)
    model_final = keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
    model_final.compile(loss="mean_squared_error", optimizer='adam')

    # Training the model (transfer learning)
    history = model_final.fit_generator(
        training_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[keras.callbacks.ProgbarLogger(count_mode='steps')])

    print('Training losses: ', history.history['loss'])
    print('Validation losses: ', history.history['val_loss'])

    # Generating submission
    submission = SubmissionWriter()
    evaluate(model_final, 'test', submission.append_test, speed_root)
    evaluate(model_final, 'real_test', submission.append_real_test, speed_root)
    submission.export(suffix='keras_example')


import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
parser.add_argument('--epochs', help='Number of epochs for training.', default=20)
parser.add_argument('--batch', help='number of samples in a batch.', default=32)
args = parser.parse_args()

main(args.dataset, int(args.epochs), int(args.batch))


'''
https://machinelearningmastery.com/save-load-keras-deep-learning-models/

Remove last layer of model with 
model.layers.pop()
'''