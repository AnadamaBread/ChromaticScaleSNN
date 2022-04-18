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

main_dir = './nsynth_files/nsynth_spec/'

IMG_WIDTH = 250
IMG_HEIGHT = 250
BATCH_SIZE = 128
OUTPUT_CLASSES = 12
EPOCHS = 15
AUTOTUNE = tf.data.AUTOTUNE



#######################################################################################
# Dataset 

######################################### Attempt using flow_from Directory

# datagen=ImageDataGenerator(rescale=1.0/255) #GrayScale

# train_ds = datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_width,img_height),
#     batch_size = BATCH_SIZE,
#     class_mode='categorical'
# )

# valid_ds = datagen.flow_from_directory(
#     valid_dir,
#     target_size=(img_width,img_height),
#     batch_size = BATCH_SIZE,
#     class_mode='categorical'
# )

######################################### Attempt using re-classification
# print("Converting Datay to Numpy Arrays...\n")

# raw = []
# labels = []
# classes = 12 # 12 classes of the chromatic scale

# print("Starting Test Directory...\n")

# for i in range(classes):
#     for img in os.listdir(test_dir):
#         # print("Looking at image: " + img + "\n")
#         im = Image.open(test_dir + img)
#         im = im.resize((150,150))
#         im = np.array(im)
#         raw.append(im)
#         labels.append(i)


# print("Completed Test Directory.\n")

# print("Starting Train Directory...\n")

# for i in range(classes):
#     for img in os.listdir(train_dir):
#         im = Image.open(train_dir + img)
#         im = im.resize((150,150))
#         im = np.array(im)
#         raw.append(im)
#         labels.append(i)


# print("Completed Train Directory.\n")

# print("Starting Valid Directory...\n")


# v_raw = []
# v_labels=[]

# for img in(os.listdir(valid_dir)):
#     im = Image.open(valid_dir + img)
#     im = im.resize((150,150))
#     im = np.array(im)
#     v_raw.append(im)
#     label = 12
#     if '60' in img: 
#         label = 0
#     elif '61' in img: 
#         label = 1
#     elif '62' in img: 
#         label = 2
#     elif '63' in img: 
#         label = 3
#     elif '64' in img: 
#         label = 4
#     elif '65' in img: 
#         label = 5
#     elif '66' in img: 
#         label = 6
#     elif '67' in img: 
#         label = 7
#     elif '68' in img: 
#         label = 8
#     elif '69' in img: 
#         label = 9
#     elif '70' in img: 
#         label = 10
#     elif '71' in img: 
#         label = 11
#     v_labels.append(label)

# v_data = np.array(v_raw)
# v_labels = np.array(v_labels)
# x_val, y_val = v_data, v_labels

# data = np.array(raw)
# labels = np.array(labels)
# print("Normalizing Data...")

# norm_layer = tf.keras.layers.Rescaling(1./255) Images in Grayscale


# print("Data Converted to Numpy Arrays.")

# x_train, x_test, y_train, y_test = train_test_split(
#     data, 
#     labels, 
#     test_size=0.35, 
#     shuffle=True
#     # random_state=12
# )

# print("Training shape: ", x_train.shape, y_train.shape)
# print("/n")

# print("Testing shape: ", x_test.shape, y_test.shape)
# print("/n")

# print("Validating shape: ", x_val.shape, y_val.shape)
# print("/n")


# y_train = to_categorical(y_train, OUTPUT_CLASSES)
# y_test = to_categorical(y_test, OUTPUT_CLASSES)
# y_val = to_categorical(y_val, OUTPUT_CLASSES)

######################################### Attempt using tensorflow image dataset

print("Train Dataset... ")

train_ds =  tf.keras.utils.image_dataset_from_directory(
    main_dir,
    validation_split = 0.2,
    subset="training",
    seed=123,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE
)

print("Valid Dataset...")

valid_ds =  tf.keras.utils.image_dataset_from_directory(
    main_dir,
    validation_split = 0.2,
    subset="validation",
    seed=123,
    image_size = (IMG_HEIGHT, IMG_WIDTH),
    batch_size = BATCH_SIZE
)

#######################################################################################
# Model

print("Building Model...")
####################################
model = Sequential()
model.add(Conv2D(
    filters = 16,
    kernel_size=(3,3),
    activation='relu',
    padding="same",
    input_shape=x_train.shape[1:]
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Conv2D(
    filters = 32,
    kernel_size=(3,3),
    activation='relu',
    padding="same"
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
    padding="same"
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
    padding="same"
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Conv2D(
    filters = 128,
    kernel_size=(3,3),
    activation='relu',
    padding="same"
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(Conv2D(
    filters = 128,
    kernel_size=(3,3),
    activation='relu',
    padding="same"
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))

model.add(Dense(OUTPUT_CLASSES,activation="softmax"))

model.compile(
    optimizer='adam', 
    # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)


model.summary()

# history = model.fit_generator(
#     train_ds,
#     steps_per_epoch=len(train_ds),
#     epochs=5,
#     validation_data=valid_ds,
#     # validation_steps=len(valid_ds)
# )

# history = model.fit(
#     x_train, 
#     y_train, 
#     epochs=EPOCHS,
#     verbose=1,
#     batch_size=BATCH_SIZE, 
#     # validation_data=(x_test, y_test)
#     validation_data=(x_val, y_val)
# )

histoy = model.fit(
    train_ds,
    validation_data = valid_ds,
    verbose = 1,
    epochs=EPOCHS
)

#validation_data=(x_val, y_val)

model.save('chromatic_classifier.h5')


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
