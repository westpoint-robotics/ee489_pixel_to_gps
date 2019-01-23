#!/usr/bin/env python
import cv2
import numpy as np
import rospy

for i in range(50):
    
    cap = cv2.VideoCapture(i)
    print(cap.isOpened())
    test, frame = cap.read()
    print(i,test)

