# Written by Luis Baez and Cassidy Bradley 
# CNN model for generalized Audio Classification
# Following will be done afterwords:
#   Parse Dataset into Guitar, Piano, Trumpet
#   Classify only audio from the chromatic Scale notes of each instrument

# Notes of the Chromatic Scale: A, A#/Bb, B, C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, and G#/Ab. 



# Import Libraries 
import os
import sys
import time
import warnings
import librosa 
import librosa.display
import pickle 
import joblib

import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
import tensorflow_datsets as tfds
from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split


#########################################################

# Dataframe -- Need to parse from NSynth Here
# df = 'nsynth'
nsynth_classes = ['guitar', 'flute', 'brass', 'keyboard'] # Instruments 
nsynth_notes = [] # Notes 

#########################################################
# Config Settings -- General
REMOVE_MODEL = True
REMOVE_DATA = True
TRAIN_NEW_MODEL = True
USE_GPUS = False
N_GPUS = 4
#########################################################
# Global Variables -- General
DS_NAME = 'nsynth'
OUTPUT_CLASSES = 10 # Default number, will definitely need to change
BATCH_SIZE = 128
EPOCHS = 20
TRAIN_PERC = 128
VAL_PERC = 10
TEST_PERC = 10
TRAIN = 'test+train[:{}%]'.format(str(TRAIN_PERC))
VAL = 'test+train[{}%:{}%]'.format(str(TRAIN_PERC),str(TRAIN_PERC+VAL_PERC))
TEST = 'test+train[{}%:]'.format(str(TRAIN_PERC+VAL_PERC))
#########################################################
# Paths
MODEL_OUT_DIR = os.path.join(os.path.abspath('..'), 'models')
WORKING_DIR = os.path.join(os.path.abspath('..'),'data')
# Output filenames
MODEL_NAME = 'nsynth_cnn.h5' #Changed from lenet.h5 to VGG_9.h5
# Print the dirs
print('LOG --> MODEL_OUT_DIR: '+str(MODEL_OUT_DIR))
print('LOG --> DATASET_DIR: '+str(WORKING_DIR))
# Check that dirs exist if not create
if not os.path.exists(MODEL_OUT_DIR):
    os.mkdir(MODEL_OUT_DIR)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
# Cleanup last run
if os.path.exists(os.path.join(MODEL_OUT_DIR, MODEL_NAME)) and REMOVE_MODEL:
    os.remove(os.path.join(MODEL_OUT_DIR, MODEL_NAME))
if os.path.exists(os.path.join(WORKING_DIR, 'x_test.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'x_test.npz'))
if os.path.exists(os.path.join(WORKING_DIR, 'y_test.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'y_test.npz'))
if os.path.exists(os.path.join(WORKING_DIR, 'x_norm.npz')) and REMOVE_DATA:
    os.remove(os.path.join(WORKING_DIR, 'x_norm.npz'))
# Load an Audio File
sample_num = 1 # File to use
filename = df.recording_id[sample_num] + str('.flac')
time_start = df.t_min[sample_num]
time_end = df.t_max[sample_num]
#########################################################
# GPU USAGE
# Distribute on multiple GPUS

if USE_GPUS:
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    devices_names = [d.name.split('e:')[1] for d in devices]
    print(devices_names)
    strategy = tf.distribute.MirroredStrategy(devices=devices_names[:N_GPUS])
else:
    device_type = 'CPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    device_names = [d.name.split('e:')[1] for d in devices]
    strategy = tf.distribute.OneDeviceStrategy(device_names[0])

#########################################################

# Load file here
# sample_rate = librosa.load()
# librosa.display.waveplot(file,sample_rate,x_axis = 'time', color='blue')

# x = df.drop('species_id', axis=1) feature extraction
# y = df.species_id

#########################################################
# Functions -- General helper functions
# Pad images. Model looks for greyscale image of audio

def padding(arr, x, y):
    """ Pads array given the desired height and width"""
    height = arr.shape[0]
    width = arr.shape[1]

    max_i = max((x - height) // 2,0)
    max_i2 = max((0, x - max_a - height))

    max_j = max(0,(y - width) // 2)
    max_j2 = max( y - max_j - width, 0)

    return np.pad(arr, pad_width=((max_i, max_i2), (max_j, max_j2)), mode = 'constant')

def generalize_features(y_cut):
    max_size = 2000 # set auido file width

    stft = padding(np.abs(librosa.stft(y_cut, n_fft=255, hop_length = 512)), 128, max_size)
    
    mfccs = padding(librosa.feature.mfcc(y_cut, n_fft=n_fft,hop_length = hop_length, n_mfcc=128), 128, max_size)

    spec_centroid = librosa.feature.spectral_centroid(y=y_cut, sr=sample_rate)

    chroma_stft = librosa.feature.chrome._stft(y=y_cut, sr = sample_rate)

    spec_bw = libroas.feature.spectral_bandwidth(y=y_cut, sr = sample_rate)

    #Apply padding
    image = np.array([padding(normalize(spec_bw), 1 ,max_size)]).reshape(1, max_size)

    image = np.append(image, padding(normalize(spec_sentroid), 1, max_size), axis=0)

    # Continue padding -- mfcc and stft sized
    for i in range(0,9):
        image = np.append(image, padding(normalize(spec_bw), 1, max_size), axis=0)
        image = np.append(image, padding(normalize(spec_centroid), 1, max_size), axis = 0)
        image = np.append(image, padding(normalize(chroma_stft), 12, max_size), axis = 0)
        image = np.dstack(image, np.abs(stft))
        image = np.dstack((image, mfccs))
    return image

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def preprocess_ds(a_ds, info, a_split, eval_flag=False):
    """Normalize images, shuffle, and prep datasets."""
    ds = a_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if not eval_flag:
        ds = ds.shuffle(info.splits[a_split].num_examples)
    # ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def convert_to_numpy(ds):
    """Converts the tensors to numpy arrays"""
    global OUTPUT_CLASSES
    images = []
    labels = []
    ds_len = len(ds)
    print('LOG --> Processing dataset of length {}...'.format(ds_len))
    for image, label in tfds.as_numpy(ds):
        images.append(np.asarray(image))
        labels.append(int(label))
    images = np.asarray(images)
    labels = to_categorical(np.asarray(labels), OUTPUT_CLASSES)
    print(images.shape)
    print(labels.shape)
    return images, labels

def save_data_as_npz(images, labels, suffix, split):
    """Saves data as seperate image and label files."""
    global WORKING_DIR
    if split:
        np.savez_compressed(os.path.join(WORKING_DIR, 'x_{}'.format(suffix)),
                            images[::split])
    else:
        np.savez_compressed(os.path.join(WORKING_DIR, 'x_{}'.format(suffix)),
                            images)
        np.savez_compressed(os.path.join(WORKING_DIR, 'y_{}'.format(suffix)),
                            labels)
    return images, labels

#########################################################
# Dataset Parsing
(ds_train, ds_val, ds_test), ds_info = tfds.load(
    DS_NAME,
    as_supervised=True,
    split=[TRAIN, VAL, TEST],
    shuffle_files=False,
    with_info=True
) 

ds_train = preprocess_ds(ds_train, ds_info, TRAIN, False)
ds_val = preprocess_ds(ds_val, ds_info, VAL, False)
ds_test = preprocess_ds(ds_test, ds_info, TEST, True)
x_train, y_train = convert_to_numpy(ds_train)
x_test, y_test = convert_to_numpy(ds_test)
x_val, y_val = convert_to_numpy(ds_val)

# Splits
# Split once for test and training
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=123, stratify=y)
# print(x_train.shape, x_test.shape)


# Split twice to get the validation set
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=123, stratify=y)
# print(x_train.shape, x_test.shape, x_val.shape, len(y_train), len(y_test), len(y_val))

save_data_as_npz(x_test, y_test, 'test', None)
save_data_as_npz(x_train, y_train, 'norm', 10)

img_shape = x_train.shape[1:]

#########################################################
# Get features and keeps as labels
def get_features(data):
    features=[]
    labels=[]

    data = data.reset_index()
    for i in data.species_id.unique():
        print('species_id:', i)
        filelist = data.loc[data.species_id == i].index

        for j in range(0, len(filelist)):
            filename = data.iloc[filelist[j]].recording_id + str('.flac') # gets file name -- probably won't need this
            # define time signal capture
            time_start = data.iloc[filelist[j]].t_min
            time_end = data.iloc[fileslist[j]].t_max 
            recording_id = data.iloc[filelist[j]].recording_id
            species_id = i
            songtype_id = data.iloc[filelist[j]].songtype_id
            # Load file
            y, sample_rate = librosa.load(filename, sr=28000)
            # cut to start and end
            y_cut = y[int(round(time_start * sample_rate):int(round(time_end * sample_rate))]
            # generate features & output numpy array
            data = generate_features(y_cut)
            features.append(data[np.newaxis,...])
            labels.append(species_id)
    
    output = np.concatenate(features, axis = 0)
    return(np.array(output), labels)

# Calculate and store features
test_features, test_labels = get_features(pd.concat([x_test, y_test], axis=1))
train_features, train_labels = get_features(pd.concat([x_train, ytrain], axis=1))

# Normalize data into numpy array
x_train = np.array((x_train - np.min(x_train)) / np.max(x_train) - np.min(x_train))
x_test = np.array((x_test - np.min(x_test)) / np.max(x_test) - np.min(x_test))
x_train = x_train/np.std(x_train)
x_test = x_test/np.std(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#########################################################
# Convolutional Model
img_shape = (128, 1000, 3)

print('LOG --> Input Shape: '+str(img_shape))

with strategy.scope():
    # Input Layer
    inputs = tf.keras.Input(img_shape)
    # Begin Network -- Complex model
    layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(inputs)
    layer = tf.keras.layers.MaxPooling2D((2,2))(layer)
    layer = tf.keras.layers.Dropout(0.2)
    layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(layer)
    layer = tf.keras.layers.MaxPooling2D((2,2))(layer)
    layer = tf.keras.layers.Dropout(0.2)
    layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(layer)
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(units=64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.2)
    layer = tf.keras.layers.Dense(units=32, activation='relu')(layer)
    # layer = tf.keras.layers.Dense(units=24, activation='softmax')(layer)
    # Output Layer
    outputs = tf.keras.layers.Dense(units=OUTPUT_CLASSES, activation='softmax')(layer)

    # Instantiate the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

# Show the model summary
model.summary()

if TRAIN_NEW_MODEL:
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS, 
        verbose=2,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        batch_size=BATCH_SIZE
    )
else:
    model.load_weights(os.path.join(MODEL_OUT_DIR, MODEL_NAME))

# Evaluate the model on the test data
inference_start = time.time()
loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=2
)
inference_end = time.time()
total_inference_time = inference_end - inference_start
print('INFERENCE PERFORMED ON {} IMAGES IN BATCHES OF {}'.format(len(x_test), BATCH_SIZE))
print('EVALUATION LATENCY: {}'.format(total_inference_time))
print('EVALUATION LOSS: {}, EVALUATION ACC: {}'.format(loss,acc))

#########################################################
# Evaluate

history_dict = history.history
loss_values = history_dict['loss']
acc_values = history_dict['accuracy']
val_lost_values = history_dict['val_loss']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, 21)

fig, (ax1, ax2) = plt.subplot(1,2,figsize=(15,5))
ax1.plot(epochs, loss_values, 'bo', label='Training Loss')
ax1.plot(epochs, val_loss_values, 'orange', label= 'Validation Loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(epochs, acc_values, 'bo', label='Training Accuracy')
ax2.plot(epochs, val_acc_values, 'orange', label='Validation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()










