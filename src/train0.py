#!/home/wborn/anaconda3/bin/python3.6
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
from keras.models import model_from_json
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

#plots images w/ labels
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] !=3):
                ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows +1
    for i in range(len(ims)):
        sp = f.add_subplot(rows,cols,i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i],fontsize=16)
        plt.imshow(ims[i],interpolation=None if interp else 'none')

gui = input("gui? [0/1] >")

train_path = 'set/train'
valid_path = 'set/valid'
test_path = 'set/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224), classes=['l','s','r'], batch_size=100)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224), classes=['l','s','r'], batch_size=50)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224), classes=['l','s','r'], batch_size=5)

imgs,labels=next(train_batches)

if gui==1:
    plots(imgs, titles=labels)
    plt.show()

vgg16_model = keras.applications.vgg16.VGG16()

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

model.fit_generator(train_batches,steps_per_epoch=100, validation_data = valid_batches, validation_steps=15, epochs=7, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
