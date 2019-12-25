# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:20:17 2019

@author: Harshad
"""

import numpy as np
import pandas as pd
import sys

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, [13]].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

#onehotencode indices 1 and 2 from the X dataset
label = LabelEncoder()
X[:, 1] = label.fit_transform(X[:, 1])

label = None
label = LabelEncoder()
X[:, 2] = label.fit_transform(X[:, 2])
ct = ColumnTransformer([('encoder', OneHotEncoder(categories = 'auto'), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)

X = X[:, 1:]

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

##feature scaling
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit(X_test)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D

classifier = Sequential()
#inserting layers
#hidden layer
classifier.add(Dense(activation = 'relu', kernel_initializer = 'uniform', units = 6, input_shape = (11,)))
#second hidden layer
classifier.add(Dense(activation = 'relu', kernel_initializer = 'uniform', units = 6))
#output layer
classifier.add(Dense(activation = 'sigmoid', kernel_initializer = 'uniform', units = 1))
#use softmax for activation incase your classifier has more than black and white classification

#compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  #accuracy?
#optimizer is the stochastic gradient descent algo - adam, use categorical_crossentropy for multiclassification
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

#final step
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
#comparison
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
