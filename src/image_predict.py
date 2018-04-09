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
import matplotlib.pyplot as plt
from keras.models import load_model

test_path = 'set/test'

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

test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224), classes=['l','s','r'], batch_size=5)

imgs,labels=next(test_batches)

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

print("loading weights")
model.load_weights("latest.h5")
print("done.")
test_sample = np.array(imgs)
print("starting predictions")
predictions= model.predict(test_sample,batch_size=5,verbose=1)
print("done.")
print(predictions)

plots(imgs, titles=predictions)
plt.show()
