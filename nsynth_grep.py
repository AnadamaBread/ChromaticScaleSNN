# Get data from dataset
# Guitar, Flute, Piano, Brass
# Only chromatic scale notes (Maybe some bogus data)

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds

DS_NAME = 'nsynth'

