# Final appended version of the Chromatic Scale CNN 
# Resulting Model will be converted to SNN
# Written By Luis Baez and Cassidy Bradley
################################################################################

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

################################################################################
REMOVE_MODEL = True
REMOVE_DATA = True
TRAIN_NEW_MODEL = True
USE_GPUS = False
N_GPUS = 4
################################################################################
# Global Variables -- General
DS_NAME = 'nsynth'
OUTPUT_CLASSES = 12 # 12 Chromatic notes of the chromatic scale
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_PERC = 60
VAL_PERC = 10
TEST_PERC = 30

AUTOTUNE = tf.data.experimental.AUTOTUNE

TRAIN = 'test+train[:{}%]'.format(str(TRAIN_PERC))
VAL = 'test+train[{}%:{}%]'.format(str(TRAIN_PERC),str(TRAIN_PERC+VAL_PERC))
TEST = 'test+train[{}%:]'.format(str(TRAIN_PERC+VAL_PERC))
################################################################################
# Paths
MODEL_OUT_DIR = os.path.join(os.path.abspath('..'), 'models')
WORKING_DIR = os.path.join(os.path.abspath('..'),'data')
# Output filenames
MODEL_NAME = 'vgg_9.h5' #Changed from lenet.h5 to VGG_9.h5
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
################################################################################
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
################################################################################
# Helper Functions
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


def parse_tfr_elem(tfre):
    parse_dict = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    
    m = tf.io.parse_single_example(tfre, parse_dict)

    img = m['image']
    height = m['height']
    width = m['width']
    depth = m['depth']
    label = m['label']

    feature = tf.io.parse_tensor(img, out_type = tf.uint8)
    feature = tf.reshape(feature, shape=[height, width, depth])
    return(feature, label)

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
################################################################################
# Dataset Import

def get_dataset(filename, type):
    order_off = tf.data.Options()
    order_off.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filename)

    ds = ds.with_options(order_off)

    ds = ds.map(parse_tfr_elem, num_parallel_calls = AUTOTUNE)
    ds = ds.shuffle(2048, reshuffle_each_iteration = True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size = AUTOTUNE)
    ds = ds.repeat() if type == 'train' else ds

    return ds

ds_train = get_dataset('chromatic.tfrecords', "train")
ds_valid = get_dataset('chromatic.tfrecords', "valid")
ds_test = get_dataset('chromatic.tfrecords', "test")

# (ds_train, ds_val, ds_test), ds_info = tfds.load(
#     DS_NAME,
#     as_supervised=True,
#     split=[TRAIN, VAL, TEST],
#     shuffle_files=True,
#     with_info=True
# )

# ds_train = preprocess_ds(ds_train, ds_info, TRAIN, False)
# ds_val = preprocess_ds(ds_val, ds_info, VAL, False)
# ds_test = preprocess_ds(ds_test, ds_info, TEST, True)
x_train, y_train = convert_to_numpy(ds_train)
x_test, y_test = convert_to_numpy(ds_test)
x_val, y_val = convert_to_numpy(ds_val)

# # Save test set
# save_data_as_npz(x_test, y_test, 'test', None)
# save_data_as_npz(x_train, y_train, 'norm', 10)

# img_shape = x_train.shape[1:]
################################################################################
# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_OUT_DIR, MODEL_NAME),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=0,
    mode='min'
)
callbacks = [checkpoint]


################################################################################
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

################################################################################

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
