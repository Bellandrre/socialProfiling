# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:11:07 2018

@author: karthik
"""

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout
from keras import optimizers

model_vgg16 = keras.applications.vgg16.VGG16()
model = Sequential()
for layer in model_vgg16.layers:
    model.add(layer)

model.layers.pop()
model.summary()
for layer in model.layers:
    layer.trainable = False

model.add(Dense(output_dim = 2, activation = 'softmax'))

model.compile(keras.optimizers.Adam(lr=.0001), loss= 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/karth/Documents/ML/Project/Dataset/Cropped_train',
        target_size=(224, 224),
        batch_size=16,
        classes=['Male','Female'])

test_set = test_datagen.flow_from_directory(
        'C:/Users/karth/Documents/ML/Project/Dataset/Cropped_test',
        target_size=(224, 224),
        batch_size=16,
        classes=['Male','Female'])

model.fit_generator(
        training_set,
        steps_per_epoch=6055,
        epochs=5,
        validation_data=test_set,
        validation_steps=1252)