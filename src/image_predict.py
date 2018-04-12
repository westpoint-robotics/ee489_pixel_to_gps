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
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None, labels=None):
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
            if labels is not None:
                dt="Net: "
                if titles[i] == 0:
                    dt+='left '
                elif titles[i] == 1:
                    dt+='straight '
                elif titles[i] == 2:
                    dt+='right '
                dt+="Orig: "
                if labels[i] == 0:
                    dt+='left '
                elif labels[i] == 1:
                    dt+='straight '
                elif labels[i] == 2:
                    dt+='right '


                sp.set_title(dt,fontsize=10)
        plt.imshow(ims[i],interpolation=None if interp else 'none')

gui = input("gui? [0/1] >")

test_path = 'set/test'

test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(50,50), classes=['l','s','r'], batch_size=5)

imgs,labels=next(test_batches)

if gui==1:
    plots(imgs, titles=labels)
    plt.show()

vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(50,50,3),classes=3,pooling='max')

#print(vgg16_model.summary())

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.layers.pop()


for layer in model.layers:
    layer.trainable = False

model.add(Dense(3, activation='softmax'))
print(model.summary())


model.compile(Adam(lr=.001),loss='categorical_crossentropy', metrics=['accuracy'])

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

    #if gui == 1:
    plots(imgs, titles=predictions, labels=labels_print)
    plt.show()
    imgs,labels=next(test_batches)
