# ChromaticScaleSNN
Convolution Neural Network with SNN Toolbox integration for Neuromorphic Computing 

## Idea
### Top Down approach

1. Convert a working Convolutional Neural Network into a Spiking Neural Network.  
2. Build a Convolutional Neural Network to classify spectrogram images.  
3. Build spectrogram images from audio files of the instruments.  
4. Separate necessary audio from instruments by the Chromatic Scale.  

### Notes of the Chromatic Scale: A, A#/Bb, B, C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, and G#/Ab.

Chart of pitch frequencies. 

| Note | Octave 0 | Octave 1 | Octave 2|  ...  | Octave 4 |  ...  | Octave 8 |
| -----| -------- | -------- | ------- | ----- | -------- | ----- | -------- |
|  C   | 16.35 Hz | 32.70 Hz | 65.41 Hz |      | 261.63 Hz |      | 4186.01 Hz|
| C#/Db| 17.32 HZ | 34.65 Hz | 69.30 Hz |      | 277.18 Hz |      | 4434.92 Hz|
|  D   | 18.35 Hz | 36.71 Hz | 73.42 Hz |      | 293.66 Hz |      | 4698.63 Hz|
| D#/Eb| 19.45 HZ | 38.89 Hz | 77.78 Hz |      | 311.13 Hz |      | 4978.03 Hz|
|  E   | 20.60 Hz | 41.20 Hz | 82.41 Hz |      | 329.63 Hz |      | 5274.04 Hz|
|  F   | 21.83 HZ | 43.65 Hz | 87.31 Hz |      | 349.23 Hz |      | 5587.65 Hz|
| F#/Gb| 23.12 Hz | 46.25 Hz | 92.50 Hz |      | 369.99 Hz |      | 5919.91 Hz|
|  G   | 24.50 HZ | 49.00 Hz | 98.00 Hz |      | 392.00 Hz |      | 6271.93 Hz|
| G#/Ab| 25.96 Hz | 51.91 Hz | 103.83 Hz |     | 415.30 Hz |      | 6644.88 Hz|
|  A   | 27.50 Hz | 55.00 Hz | 110.00 Hz |     | 440.00 Hz |      | 7040.00 Hz|
| A#/Bb| 29.14 HZ | 58.27 Hz | 116.54 Hz |     | 466.16 Hz |      | 7458.62 Hz|
|  B   | 30.87 Hz | 61.74 Hz | 123.47 Hz |     | 493.88 Hz |      | 7902.13 Hz|

*Note: Range includes partially inaudible frequencies.  

Octave 4 contains average pitch measurements. The C note in octave 4 is the pitch for the middle key of a standard piano keyboard. Octave 4 will primarily be analyzed in our research. The audible human ear can here frequency pitch ranges between 20Hz and 20,000Hz. 




