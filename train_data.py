import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
import os
import operator

def train_data():
    # creating the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(6,activation='softmax')
    ])
    
    model.summary()
    
    # compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # preparing the data and training the model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # stream train images from train directory
    training_set = train_datagen.flow_from_directory(
    'Images/Images/train',
    target_size=(150,150),
    batch_size=5,
    color_mode = 'grayscale',
    class_mode = 'categorical')
    
    # stream test images from test directory
    test_set = test_datagen.flow_from_directory(
    'Images/Images/test',
    target_size=(150,150),
    batch_size=5,
    color_mode = 'grayscale',
    class_mode = 'categorical')
    
    # fitting the training data to the model
    model.fit_generator(
    training_set,
    steps_per_epoch = 40,
    epochs=5,
    validation_data=test_set,
    #validation_steps=40,
    verbose =2
    )
    
    #saving the model
    model.save('model.h5')