# For running a Model on the local images in directories


#######################################################################################
# Imports
import os
import sys
import time
import keras
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


#######################################################################################
#Configuration

test_dir = './nsynth_files/nsynth_spec/nsynth_test/'
train_dir = './nsynth_files/nsynth_spec/nsynth_train/'
valid_dir = './nsynth_files/nsynth_spec/nsynth_valid/'

# tr_dir = pathlib.Path(train_dir)
# val_dir = pathlib.Path(valid_dir)

main_dir = './nsynth_files/nsynth_data'

IMG_WIDTH = 223
IMG_HEIGHT = 221
BATCH_SIZE = 128
OUTPUT_CLASSES = 12
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

#######################################################################################
# Dataset 

######################################### Attempt using tensorflow image dataset

print("Train Dataset... ")

train_ds =  tf.keras.utils.image_dataset_from_directory(
    main_dir,
    validation_split = 0.2,
    subset="training",
    seed=0,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    color_mode = 'rgb',
    shuffle = True,
    label_mode = 'categorical'

)

print("Valid Dataset...")

valid_ds =  tf.keras.utils.image_dataset_from_directory(
    main_dir,
    validation_split = 0.2,
    subset="validation",
    seed=0,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE,
    color_mode = 'rgb',
    shuffle = True,
    label_mode = 'categorical'

)

print("Normalizing Dataset...")

norm_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y:(norm_layer(x), y))

images, labels = next(iter(normalized_ds))

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size = AUTOTUNE)

np.savez_compressed('np_images', images)
np.savez_compressed('np_labels', labels)

# ntrain = tfds.as_numpy(train_ds)
# nvalid = tfds.as_numpy(valid_ds)
# nds = tfds.as_numpy(normalized_ds)

# np.savez_compressed('np_nds', nds)


# np.savez_compressed('np_images', ntrain)
# np.savez_compressed('np_labels', nvalid)

# np_labels = tf.keras.utils.to_categorical(labels, num_classes=OUTPUT_CLASSES)


#######################################################################################
# Model
print("Building Model...")
#################################### TF format friendly Model

tf_model = tf.keras.Sequential([
    # tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same',
        input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    ),
    
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same'
    ),
    
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),


     tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same'
    ),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),


    tf.keras.layers.Dense(OUTPUT_CLASSES, activation="softmax")


])

tf_model.compile(
    optimizer='adam',
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

history = tf_model.fit(
    train_ds,
    validation_data = valid_ds,
    verbose = 1,
    epochs = EPOCHS, 
    batch_size = BATCH_SIZE
)

# tf_model.save('tf_chromatic_classifier.h5')


plt.figure(0) 
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss values')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()


# model.save('chromatic_classifier.h5')
