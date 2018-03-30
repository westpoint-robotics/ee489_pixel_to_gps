#!/home/rrc/anaconda3/bin/python3.6
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

train_path = 'set/train'
valid_path = 'set/valid'
test_path = 'set/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(100,100), classes=['s','l','r'], batch_size=100)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(100,100), classes=['s','l','r'], batch_size=50)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(100,100), classes=['s','l','r'], batch_size=50)

imgs,labels=next(train_batches)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(100,100,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.summary())

model.compile(Adam(lr=.0001),loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches,steps_per_epoch=10, validation_data = valid_batches, validation_steps=10, epochs=5, verbose=1)

model.save('latest.h5')
