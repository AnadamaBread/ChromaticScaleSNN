# Get data from dataset
# Guitar, Flute, Piano, Brass
# Only chromatic scale notes (Maybe some bogus data)


import os
import sys
import wave
import glob
import time
import pylab

import numpy as np
import pandas as pd
import matplotlib as plt
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# import tensorflow_datasets as tfds

DS_NAME = 'nsynth'

NUM_SECS = 4
AUDIO_RATE = 16000 # 16khz
LOUD_RATE = 250 # 250 hz
FRAME_SIZE = 1024
FFT = 2048
RANGE = 120.0
# NOISE = 20.7

GROUPS = 5 # Run model on groups of GROUPS

TOTAL = 484 # Total number of chromatic notes in the Train matching instrument family 

Ins_family = ["brass", "guitar", "flute", "keyboard"]
Ins_Sources = ["acoustic", "electronic", "synthetic"] # Only going to use acoustic

source = "acoustic"
pitch_range = ["060", "061", "062", "063", "064", "065", "066", "067", "068", "069", "070", "071"] # second xxx digit in file name 
velocity = "050"

Qualities = ["bright", "dark", "multiphonic", "nonlinear_env", "percussive", "tempo-synced"]

SPLITS = ["train", "valid", "test"]

od_test = './nsynth_files/nsynth_spec/nsynth_test/' # Output spectrogram files
od_train = './nsynth_files/nsynth_spec/nsynth_train/'
od_valid = './nsynth_files/nsynth_spec/nsynth_valid/'

id_train = './nsynth_files/nsynth-train/audio/' # input .wav files
id_test = './nsynth_files/nsynth-test/audio/'
id_valid = './nsynth_files/nsynth-valid/audio/'


def parse_chromatic(in_dir, out_dir):
    """
        Function in charge of selecting the correct files to be converted from wav to spectrogram .png images.
        The .png images are parsed into different train, valid, and test directories. 
        The model is the run on these images. 

    """

    # TODO
    # Create a counter for labelling spectrogram files:
    # Files should be named something abstracted except for one from each chromatic note (maybe also instrument family)
    # Name labelled files 'instrumentfamily_pitchrange' 
    # 12 total labeled files

    count = 0
    for i, filename in enumerate(os.listdir(in_dir)):
       
        if '.wav' not in filename: 
            print('.wave not in the filename ' + filename + '\n')
            continue
        if i == GROUPS:
            print("Converted up to " + str(i) + " items.")
            # sys.exit()
            break
        for item in Ins_family:
            for pitch in pitch_range:
                if item in filename and source in filename and pitch in filename:
                # Remove catching swapped pitch and velocity values
                    if velocity in filename:
                        count = count + 1
                        print(" Classifyable FOUND IN " + filename + '\n')
                graph_spectrogram((in_dir + filename), i, item, pitch, out_dir)            


def graph_spectrogram(wav_file, index, instrument, pitch, out_dir):

    sound_info, frame_rate = get_wav_info(wav_file)
    # fig_name = str(instrument) + " #_" + str(index)

    file_name = str(instrument)[2] + '_' + str(index)

    if "valid" in out_dir:
        file_name = str(instrument) + '_' + str(pitch) + '_' + str(index)

    pylab.figure(num=None, figsize=(6, 4))
    pylab.subplot(111)
    # pylab.title('spectrogram of %r' % wav_file)
    # pylab.xlabel('Time (s)')
    # pylab.ylabel('Frequency (Hz)')
    pylab.specgram(sound_info, Fs=frame_rate)
    pylab.gray()
    pylab.axis('off')
    pylab.xticks([])
    pylab.yticks([])
    # os.chdir(out_dir)
    pylab.savefig(((out_dir + file_name) + '.png'), bbox_inches='tight', pad_inches=0)

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def main():
    """
        Main function control flow of seperating runs for each frame of data from the dataset.

    """

    print("Creating Train Spectrograms...\n")
    parse_chromatic(id_train, od_train)

    print("Creating Test Spectrograms...\n")
    parse_chromatic(id_test, od_test)

    print("Creating Valid Spectrograms...\n")
    parse_chromatic(id_valid, od_valid)


    print("Done!")



if __name__ == "__main__":
    main()