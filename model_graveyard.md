# ANN Model Graveyard

### Purpose
  Here rest the model that did not preform all too well. 

## _Model 1_ 

```
model = Sequential()
model.add(Conv2D(
    filters = 32,
    kernel_size=(5,5),
    activation='relu',
    input_shape=x_train.shape[1:]
))

model.add(MaxPooling2D(2,2))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(
    filters = 32,
    kernel_size=(5,5),
    activation='relu',
))

model.add(MaxPooling2D(2,2))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(
    filters = 64,
    kernel_size=(5,5),
    activation='relu',
))

model.add(MaxPooling2D(2,2))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(
    filters = 64,
    kernel_size=(5,5),
    activation='relu',
))

model.add(MaxPooling2D(2,2))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(
    filters = 128,
    kernel_size=(5,5),
    activation='relu',
))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64,activation='relu'))

model.add(Dense(OUTPUT_CLASSES,activation="softmax"))
```

__Model 1 results after 15 epochs:__ 
Simple Sample model for nonsense. Ran on image sizes of 250x250  
Accuracy: 0.0842, loss: 2.4952, train time: 40.5 Minutes

## _Model 2_ 

```
model = Sequential()
model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
    input_shape=x_train.shape[1:]
))

model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))

model.add(Conv2D(
    filters = 128,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))

model.add(Conv2D(
    filters = 256,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))

model.add(Conv2D(
    filters = 256,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))

model.add(Conv2D(
    filters = 256,
    kernel_size=(3,3),
    activation='relu',
))

model.add(Flatten())

model.add(Dense(4096,activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(4096,activation='relu'))

model.add(Dense(OUTPUT_CLASSES,activation="softmax"))

```

__Model 2 results after 15 epochs:__   
Based on vgg-9 model from Assignment 2. Ran on image sizes of 80x80.  
*Err: only ran on valid ds.  
Accuracy: 0.0677, loss: 2.4899, train time: 1.25 Minutes  

## _Model 3_

```
model = Sequential()
model.add(Conv2D(
    filters = 32,
    kernel_size=(3,3),
    activation='relu',
    input_shape=x_train.shape[1:]
))

model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))
model.add(Dropout(rate=0.2))

model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))
model.add(Dropout(rate=0.2))


model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))

model.add(Dense(OUTPUT_CLASSES,activation="softmax"))
```
__Model 3 results after 15 epochs:__   
Based on known audio CNN model. Ran on image sizes of 150x150.  
*Err: only ran on valid ds.  
Accuracy: 0.0903, loss: 2.5308, train time: 1.5 Minutes   
_Second run with SparseCategorialCrossentropy:_  
Repeated Error:  

``` 
ValueError: `labels.shape` must equal `logits.shape` except for the last dimension. Received: labels.shape=(416,) and logits.shape=(32, 13) 
```

## _Model 4_

```
model = Sequential()
model.add(Conv2D(
    filters = 32,
    kernel_size=(3,3),
    activation='relu',
    input_shape=x_train.shape[1:]
))
model.add(Conv2D(
    filters = 32,
    kernel_size=(5,5),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
))
model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(
    filters = 128,
    kernel_size=(3,3),
    activation='relu',
))

model.add(Conv2D(
    filters = 128,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(
    filters = 256,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(
    filters = 512,
    kernel_size=(3,3),
    activation='relu',
))
model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))

model.add(Dense(OUTPUT_CLASSES,activation="softmax")

```
__Model 5 results after 15 epochs:__   
Based on known audio CNN model. Ran on image sizes of 150x150.   
*Err: only ran on valid ds for first run.  
Accuracy: 0.1094, loss: 2.5021, train time: 8.25 Minutes.     
_Second run with x_valid and y_valid:_  
Accuracy: 0.0720, lose: 2.4950, train time 11.25 Minutes.  
_Third run with test,train,valid in train-test-split():_   

## _Model 6_

This model is meant to allow overfitting...  
_starting build of final model_  

```
model = Sequential()
model.add(Conv2D(
    filters = 32,
    kernel_size=(3,3),
    activation='relu',
    input_shape=x_train.shape[1:]
))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Conv2D(
    filters = 64,
    kernel_size=(3,3),
    activation='relu',
))

model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))

model.add(Dense(OUTPUT_CLASSES,activation="softmax"))
```

__Model 6 results after 15 epochs:__   
Based on known audio CNN model. Ran on image sizes of 150x150. Batch Size now 128.   
Accuracy: 0.0940, loss: 2.5550, train time: 2.25 Minutes.  