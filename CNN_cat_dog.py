# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:39:37 2019

@author: Harshad
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Input

classifier = Sequential()
#conv layer
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#max pooling layer
classifier.add(MaxPool2D(pool_size = (2,2)))

#INCREASING ACCURACY
#conv layer
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
#max pooling layer
classifier.add(MaxPool2D(pool_size = (2, 2)))



#flatten
classifier.add(Flatten())
#classic ANN
#fully connected layer
classifier.add(Dense(units = 128, activation = 'relu'))
#output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))


#compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#data preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')
#
classifier.fit_generator(training_set, steps_per_epoch = #no of images)
                         8000, epochs = 25, validation_data = test_set, validation_steps = #no of test set images
                        2000)