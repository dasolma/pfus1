import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import argparse
import json
import numpy as np
import cv2
from matplotlib import pyplot as plt
import traceback
import statistics as stat
import segmentation_models as sm
import tensorflow as tf
import random
import pickle as pk
from glob import glob
import time
import sys
import tqdm
import shutil
from collections import defaultdict
from utils import *

tf.keras.backend.set_image_data_format('channels_last')



PRO_DIR = '../data'
COLORS = [(255,0,0),(0,255,0),(255,255,0),(0,0,255),(255,0,255),(0,255,255),(128,255,0),(128,0,255)]


class GineDataGenerator(tf.keras.utils.Sequence):
    # Data generator class for training a neural network using image and polygon data

    def __init__(self, data_dir, cases, batch_size=32, input_dim=(128, 128), 
                 shuffle=True, batches_per_epoch=1000, 
                 preprocess_input=lambda x: x / 255):
        # Initialization method

        self.batch_size = batch_size  # Number of samples per batch
        self.input_dim = input_dim  # Dimension of input images
        self.shuffle = shuffle  # Whether to shuffle data after each epoch
        self.batches_per_epoch = batches_per_epoch  # Number of batches per epoch
        self.cases = cases  # List of cases to include in the data generator
        self.data_dir = data_dir  # Directory containing the data
        self.preprocess_input = preprocess_input  # Function to preprocess input images

        self._raw_files = []  # List to store file paths of raw data

        # Loop through each case directory and collect file paths for images and corresponding JSON files
        for case in os.listdir(data_dir):
            if case in cases:
                self._raw_files += sorted([f"{case}/{f.replace('.png', '')}" 
                                 for f in os.listdir(os.path.join(data_dir, case)) 
                                 if '.png' in f])

        self.raw_indexes = np.arange(len(self._raw_files))  # Array of indexes for the raw files

        # Shuffle the indexes if shuffle is set to True
        if self.shuffle:
            np.random.shuffle(self.raw_indexes)

    def __len__(self):
        # Denotes the number of batches per epoch
        if self.batches_per_epoch > (len(self.raw_indexes) // self.batch_size):
            return max(1, len(self.raw_indexes) // self.batch_size)
        else:
            return self.batches_per_epoch 

    def __exists(self, file):
        # Check if both image and JSON files exist for a given file prefix
        return (os.path.exists(os.path.join(self.data_dir, f"{file}.png")) and 
                os.path.exists(os.path.join(self.data_dir, f"{file}.json")))


    def __getitem__(self, index):
        # Generate one batch of data

        # Get the indexes for the current batch
        indexes = self.raw_indexes[index * self.batch_size:(index + 1) * self.batch_size]
        files = [self._raw_files[i] for i in indexes]

        # Filter out files that do not exist
        files = [f for f in files if self.__exists(f)]

        # Initialize arrays for input images (X) and output masks (Y)
        X = np.zeros(shape=(len(files),) + self.input_dim)
        Y = np.zeros(shape=(len(files),) + self.input_dim + (9,))

        for i, file in enumerate(files):
            # Read image and masks using the new method
            im, masks = read_image_and_masks(self.data_dir, file, self.input_dim)

            X[i, :, :] = im
            Y[i, :, :, :] = masks

        # Apply preprocessing function to the input images
        X = self.preprocess_input(X)

        # Return the batch of data with expanded dimensions for grayscale channel
        return np.expand_dims(X, axis=-1), Y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.raw_indexes = np.arange(len(self._raw_files))
        if self.shuffle:
            np.random.shuffle(self.raw_indexes)

            
def crossval(cases, test_cases, backbone, model_name, 
             max_epochs=10000, folds=5, gpu=1, debug=False):    
    # Perform cross-validation for model training and evaluation

    # Print model and backbone information
    print("Backbone", backbone.upper())
    print("Model", model_name)
    print("=" * 20)
    print("\n\n")

    # Define result directory path
    result_dir = os.path.join(base_result_dir, model_name, backbone)

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

    # Calculate the number of samples and samples per validation fold
    nsamples = len(cases)
    val_samples = len(cases) // folds

    models = []  # List to store models

    # Define loss functions
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 1, 1, 1, 1, 1, 0.5]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    
    # Set image data format
    tf.keras.backend.set_image_data_format('channels_last')

    # Get preprocessing function for the given backbone
    preprocess_input = sm.get_preprocessing(backbone)
    
    # Initialize test data generator
    test_gen = GineDataGenerator(PRO_DIR, test_cases, input_dim=(128, 128), batch_size=32,
                                 preprocess_input=preprocess_input,
                                 batches_per_epoch=1 if debug else 100, shuffle=False)

    # Create result directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Print message indicating the start of model creation
    print("Creating models")
    
    # Shuffle cases
    cases = random.sample(cases, len(cases))
    
    for i in range(folds):
        # Split cases into training and validation sets for each fold
        val_cases = cases[i*val_samples:(i+1)*val_samples]
        train_cases = cases[:i*val_samples] + cases[(i+1)*val_samples:]

        # Define optimizer
        optim = tf.keras.optimizers.Adam(LR)

        # Define metrics
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # Initialize training and validation data generators
        train_gen = GineDataGenerator(PRO_DIR, train_cases, input_dim=(128, 128), 
                                      preprocess_input=preprocess_input, batch_size=8,
                                      batches_per_epoch=1 if debug else 25)
        val_gen = GineDataGenerator(PRO_DIR, val_cases, input_dim=(128, 128), batch_size=32,
                                    preprocess_input=preprocess_input,
                                    batches_per_epoch=1 if debug else 5, shuffle=False)

        # Initialize model
        model = model_class(backbone_name=backbone, classes=9, 
                            activation='softmax', input_shape=(None, None, 1),
                            encoder_weights=None)

        # Compile model with defined optimizer, loss, and metrics
        model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
        metric_names = [m.name for m in metrics] + ['loss']

        # Append model and related data to the models list
        models.append({
            'model': model,
            'train_gen': train_gen,
            'val_gen': val_gen,
            'train_cases': train_cases,
            'val_cases': val_cases,
            'scores': {m: [] for m in metric_names},
            'test_scores': {m: [] for m in metric_names},
        })

    # Print message indicating the start of training
    print("Start training")
    min_loss = 0
    last_improvement_epoch = 0

    for epoch in range(max_epochs):
        print("\n\nEpoch", epoch, "\n----------")
        for i in range(folds):
            print("Fold", i)

            # Define early stopping callback
            es = tf.keras.callbacks.EarlyStopping(patience=5)
            
            # Fit model for one epoch
            model = models[i]['model']
            history = model.fit(
                x=models[i]['train_gen'],
                epochs=1,
            )

            # Evaluate model on validation data
            scores = model.evaluate(models[i]['val_gen'], verbose=0)
            metric_names = [m.name for m in model.metrics]
            
            for name, score in zip(metric_names, scores):
                models[i]['scores'][name].append(score)
                print(f"{name}: {score}")

            models[i]['model'] = model
        
        # Create progress directory if it doesn't exist
        if not os.path.exists(os.path.join(result_dir, 'progress')):
            os.makedirs(os.path.join(result_dir, 'progress'))
        
        # Generate and save epoch progress graph
        generate_epoch_graph(models, PRO_DIR, os.path.join(result_dir, 'progress'))

        print("")
        for name in metric_names:
            score = np.mean([models[i]['scores'][name][-1] for i in range(len(models))])
            print(f"Mean {name}: {score}")
        
        f1_score = np.mean([models[i]['scores']['f1-score'][-1] for i in range(len(models))])
        if f1_score > min_loss:
            last_improvement_epoch = epoch
            min_loss = f1_score

        print("Best F1 score:", min_loss, "\n--------------------")
        print("Best F1 score found:", epoch - last_improvement_epoch, "ago")

        # Early stopping if no improvement for 5 consecutive epochs
        if epoch - last_improvement_epoch > 5:
            break

    # Print message indicating the start of test evaluation
    print("\nTest evaluation")
    print("---------------------------------\n")
    
    predictions = []
    test_scores = []
    
    for model in models:
        # Evaluate model on test data
        r = model['model'].evaluate(test_gen, verbose=1)
        metric_names = [m.name for m in model['model'].metrics]
        
        for name, score in zip(metric_names, scores):
            model['test_scores'][name].append(score)

        # Predict on test data and store predictions
        p = model['model'].predict(test_gen)
        predictions.append(p)

    # Calculate mean predictions across all models
    predictions = np.array(predictions)
    y_pred = np.mean(predictions, axis=0)

    # Collect true labels
    y_true = []
    for x, y in test_gen:
        y_true.append(y)

    y_true = np.vstack(y_true)

    # Convert predictions and true labels to float32
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)

    # Calculate F1 score on test data
    metric = sm.metrics.FScore(threshold=0.3)
    test_f1score = metric(y_true, y_pred).numpy()  

    # Save results
    to_save = []
    for model in models:
        to_save.append({
            'train_cases': model['train_cases'],
            'val_cases': model['val_cases'],
            'scores': model['scores'],
            'test_scores': model['test_scores'],
        })
    result = {
        'models': to_save,
        'voting_score': test_f1score,
    }
    pk.dump(result, open(os.path.join(result_dir, 'result.pk'), 'wb'))

    print("Test F1 Score: ", test_f1score)

    return models

cases = sorted(os.listdir(PRO_DIR))
print(f"Found {len(cases)} cases")

test_cases = cases[-25:]
cases = cases[:-25]

print("Train cases: ", str(cases))
print("Test cases: ", str(test_cases))

BACKBONE = 'resnet50'
model_class = sm.FPN
MODEL = 'FPN'
LR = 0.0001
RESULT_DIR = 'results'

base_result_dir = RESULT_DIR

models = crossval(cases, test_cases, BACKBONE, MODEL, 
                  max_epochs=10000, folds=5, gpu=1, debug=False)
