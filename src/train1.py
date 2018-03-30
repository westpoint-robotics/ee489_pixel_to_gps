#!/home/rrc/anaconda3/bin/python3.6
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras import optimizers
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = 'set/train'
valid_path = 'set/valid'
test_path = 'set/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(100,100), classes=['l','r'], batch_size=150, shuffle=True)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(100,100), classes=['l','r'], batch_size=30,shuffle=True)


imgs,labels=next(train_batches)

imgs = imgs.astype('float32')
imgs /= 255

a= np.random.randint(0,high= 60000,size=10)
for i in range(9):
   plt.subplot(3,3,i+1)
   plt.imshow(imgs[i,0])
   plt.axis("off")
plt.show()

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(100,100,3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

print(model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches,steps_per_epoch=20, validation_data = valid_batches, validation_steps=10, epochs=50, verbose=1)

model.save('latest.h5')
