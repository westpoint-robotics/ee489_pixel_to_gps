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
from cv_bridge import CvBridge, CvBridgeError
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image

run = False
def callback(data):
    global img
    img = CvBridge().imgmsg_to_cv2(data, "bgr8")
    run=True

print("load model")
vgg16_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=(100,100,3),classes=3,pooling='max')
model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
model.layers.pop()
for layer in model.layers:
    layer.trainable = False
model.add(Dense(3, activation='softmax'))
model.compile(Adam(lr=.0001),loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights("model.h5")
print("loaded")

image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,callback)

while True:
    if run:
        print("starting predictions")
        predictions= model.predict_classes(img)
        print("done.")
        print(predictions)
