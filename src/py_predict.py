#!/usr/bin/python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import itertools
from keras.preprocessing.image import img_to_array, load_img



img=load_img("set/test/l/img_3527_l.png")
x = img_to_array(img)


vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(100,100,3),classes=3,pooling='max')

#print(vgg16_model.summary())

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.layers.pop()


for layer in model.layers:
    layer.trainable = False

model.add(Dense(3, activation='softmax'))
print(model.summary())


model.compile(Adam(lr=.0001),loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("/home/rrc/model_trial/model.h5")

print("starting predictions")
predictions= model.predict_classes(x[None,:,:,:],batch_size=1,verbose=2)
print("done.")
print(predictions)
