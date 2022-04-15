import os
import sys
import wave
import glob
import time
import pylab

import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
import IPython.display as display

from PIL import Image


################################################################################

# Convert Image data into TFRecords

# Numeric Values

def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# String/Char Values

def byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Create TFrecords files.
tfrecord_filename = 'chromatic.tfrecords'
writer = tf.io.TFRecordWriter(tfrecord_filename)


img_valid = glob.glob('nsynth_files/nsynth_spec/nsynth_valid/*.png')
img_test = glob.glob('nsynth_files/nsynth_spec/nsynth_test/*.png')
img_train = glob.glob('nsynth_files/nsynth_spec/nsynth_train/*.png')

for image in img_valid:
    img = Image.open(image)
    img = np.array(img.resize((32,32)))
    img_shape = img.shape

    label = 1
    if '60' in image: 
        label = 60
    elif '61' in image: 
        label = 61
    elif '62' in image: 
        label = 62
    elif '63' in image: 
        label = 63
    elif '64' in image: 
        label = 64
    elif '65' in image: 
        label = 65
    elif '66' in image: 
        label = 66
    elif '67' in image: 
        label = 67
    elif '68' in image: 
        label = 68
    elif '69' in image: 
        label = 69
    elif '70' in image: 
        label = 70
    elif '71' in image: 
        label = 71
    

    feature = { 
        'height': int_feature(img_shape[0]),
        'width': int_feature(img_shape[1]),
        'depth': int_feature(img_shape[2]),
        'label' : int_feature(label), 
        'image': byte_feature(img.tostring()) 
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

for image in img_test:
    img = Image.open(image)
    img = np.array(img.resize((32,32)))
    img_shape = img.shape


    label = 3

    feature = { 
        'height': int_feature(img_shape[0]),
        'width': int_feature(img_shape[1]),
        'depth': int_feature(img_shape[2]),
        'label' : int_feature(label), 
        'image': byte_feature(img.tostring()) 
    }

    tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

for image in img_train:
    img = Image.open(image)
    img = np.array(img.resize((32,32)))
    img_shape = img.shape



    label = 2

    feature = { 
        'height': int_feature(img_shape[0]),
        'width': int_feature(img_shape[1]),
        'depth': int_feature(img_shape[2]),
        'label' : int_feature(label), 
        'image': byte_feature(img.tostring()) 
    }

    tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()

################################################################################

# reader = tf.data.TFRecordReader()
# filenames = glob.glob('*.tfrecords')
# filename_queue = tf.train.string_input_producer(filenames)

# serialized_example = reader.read(filename_queue)
# feature_set = {'image': tf.FixedLenFeature([], tf.string), 'label' : tf.FixedLenFeature([], tf.int64) }

# features = tf.parse_single_example(serialized_example, features = feature_set)
# label = features['label']

# with tf.Session() as sess:
#     print(sess.run([image,label]))

# for record in raw_dataset.take(10):
#     example = tf.train.Example()
#     example.ParseFromString(record.numpy())
#     print(example)

################################################################################



filenames = glob.glob('*.tfrecords')

raw_dataset = tf.data.TFRecordDataset(filenames)



image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string)
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset.take(1):
    image_raw = image_features['image'].numpy()
    display.display(display.Image(data=image))