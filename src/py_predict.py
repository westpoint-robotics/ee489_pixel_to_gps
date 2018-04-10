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


train_path = 'set/train'
valid_path = 'set/valid'
test_path = 'set/test'

test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(100,100), classes=['l','s','r'], batch_size=5)

imgs,labels=next(test_batches)


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

model.load_weights("model.h5")


while True:
    labels_print= []
    for i in labels:
        if i[0]==1:
            labels_print.append(0)
        elif i[1]==1:
            labels_print.append(1)
        elif i[2]==1:
            labels_print.append(2)
    test_sample = np.array(imgs)
    print("starting predictions")
    predictions= model.predict_classes(test_sample,batch_size=5,verbose=2)
    print("done.")
    print(predictions)
    print(labels_print)

    imgs,labels=next(test_batches)
