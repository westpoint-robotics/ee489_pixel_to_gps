#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

global bridge, cvImage
bridge = CvBridge()

def image_callback(data):
    global bridge, cvImage
    try:
        cvImage = bridge.imgmsg_to_cv2(data,"bgr8")
    except CvBridgeError as e:
        print(e)


# Capture the input frame from webcam


    # Resize the input frame
    #frame = cv2.resize(frame, None, fx=scaling_factor,fy=scaling_factor, interpolation=cv2.INTER_AREA)


if __name__=='__main__':
    global cvImage
    imageArray = Float64MultiArray()
    imageArray.data = [0,0,0,0]
    rospy.init_node('c1_pixelCoordinates', anonymous=True)
    rospy.Subscriber("/camera_array1/cam1/image_raw", Image, image_callback)
    pub = rospy.Publisher('c1_pixelCoordinates', Float64MultiArray, queue_size=1)
    rate = rospy.Rate(10)   

    # Iterate until the user presses ESC key
    frame_count = 0
    pixelCoords = np.zeros((1,2))
    scaling_factor = 0.5
    while not rospy.is_shutdown():
        try:
            frame = cvImage
        # Capture frame-by-frame
        #ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)
        # Convert the HSV colorspace
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
            mask = cv2.inRange(hsv,np.array([0,150,150]),np.array([30,255,255]))
        
        # Bitwise-AND mask and original image
            res = cv2.bitwise_and(frame, frame, mask=mask)
            res = cv2.medianBlur(res, 5)
       

            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
            areas = [cv2.contourArea(c) for c in contours]
            if len(contours) > 0:
                max_index = np.argmax(areas)
                myBox = contours[max_index]
                x,y,w,h = cv2.boundingRect(myBox)
            #print(cv2.boundingRect(myBox))
            
                cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255),1)
        

            #PIXEL COORDINATES OF OBJECT!!!
            
                pixelCoords = np.concatenate((pixelCoords,np.array([[x+0.5*w, y + 0.5*h]])))
            #print(pixelCoords[frame_count])
                x_Coord = x+0.5*w
                y_Coord = y+0.5*h
                area = w*h
                pixelTime = float(str(rospy.Time.now()))
            
            #print("X Pos: " + str(x_Coord) + " Y Pos: " + str(y_Coord) + " Area: " + str(area))

                imageArray.data = [pixelTime, x_Coord, y_Coord, area]
                pub.publish(imageArray)
            
                frame_count += 1


            cv2.imshow('Original image', frame)
            cv2.imshow('Color Detector', res)
        
            c = cv2.waitKey(5)
            if c == 27:
                break
        except:
            print("error..waiting on camera")
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
