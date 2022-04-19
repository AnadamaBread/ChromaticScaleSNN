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

## _Model 7_  

Formatting of Models have changed.  

```
tf_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters = 32,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same',
        input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    ),
    
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same'
    ),
    
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

     tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same'
    ),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense(4096, activation='relu'),

    tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Dense(OUTPUT_CLASSES, activation="softmax")


])
```

__Model 7 results after 20 epochs:__   
Accuracy: 0.9000, loss: 0.3206, train time: 7.15 Minutes.  

## _Stray Model_

```
# model = Sequential()
# model.add(Conv2D(
#     filters = 16,
#     kernel_size=(3,3),
#     activation='relu',
#     padding="same",
#     input_shape=x_train.shape[1:]
# ))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))
# model.add(Conv2D(
#     filters = 32,
#     kernel_size=(3,3),
#     activation='relu',
#     padding="same"
# ))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))
# model.add(Conv2D(
#     filters = 64,
#     kernel_size=(3,3),
#     activation='relu',
#     padding="same"
# ))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))
# model.add(Conv2D(
#     filters = 64,
#     kernel_size=(3,3),
#     activation='relu',
#     padding="same"
# ))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))
# model.add(Conv2D(
#     filters = 128,
#     kernel_size=(3,3),
#     activation='relu',
#     padding="same"
# ))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))

# model.add(Conv2D(
#     filters = 128,
#     kernel_size=(3,3),
#     activation='relu',
#     padding="same"
# ))
# model.add(tf.keras.layers.MaxPooling2D((2,2)))

# model.add(Flatten())
# model.add(Dense(128,activation='relu'))

# model.add(Dense(OUTPUT_CLASSES,activation="softmax"))

# model.compile(
#     optimizer='adam', 
#     # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=['accuracy']
# )


# model.summary()

# # history = model.fit_generator(
# #     train_ds,
# #     steps_per_epoch=len(train_ds),
# #     epochs=5,
# #     validation_data=valid_ds,
# #     # validation_steps=len(valid_ds)
# # )

# history = model.fit(
#     x_train, 
#     y_train, 
#     epochs=EPOCHS,
#     verbose=1,
#     batch_size=BATCH_SIZE, 
#     # validation_data=(x_test, y_test)
#     validation_data=(x_val, y_val)
# )

# # histoy = model.fit(
# #     train_ds,
# #     validation_data = valid_ds,
# #     verbose = 1,
# #     epochs=EPOCHS
# # )

# model.save('chromatic_classifier.h5')
```


## _Model 8_

Model build with BatchNormalizations 

```
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
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same'
    ),
    
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),



     tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = (3,3),
        activation = 'relu',
        padding='same'
    ),


    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Dense(OUTPUT_CLASSES, activation="softmax")


])

```

__Model 8 results after 20 epochs:__   
Model based on batch normalization
Images sizes are 223x221 with 2 nChannels. Batch Size of 128. 
Accuracy: 0.9600, loss: 0.1106, train time: 7.15 Minutes.  

## _Final Model_

```
tf_model = tf.keras.Sequential([
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
```
__Model Final results after 10 epochs:__   
Model based on dropout 0.2
Images sizes are 223x221 with 2 nChannels. Batch Size of 128. 
Accuracy: 0.8914, loss: 0.3255, train time: 2.30 Minutes. 