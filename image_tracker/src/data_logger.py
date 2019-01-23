#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
xPos = 0.0



def callback(data):
    global xPos
    xPos = data.data[1]

def listener():
    global xPos
    rospy.init_node('data_logger', anonymous = True)
    rospy.Subscriber('c1_pixelCoordinates', Float64MultiArray, callback)
    while not rospy.is_shutdown():
        print(str(xPos))


if __name__ == '__main__':
    listener()
    
