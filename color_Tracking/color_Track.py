#https://thecodacus.com/opencv-object-tracking-colour-detection-python/#.W7ftmTB95D8

import cv2
import numpy as np

lowerBound=np.array([9,174,96]) #range of HSV values lower limit
upperBound=np.array([26,255,255]) #range of HSV values upper limit

cam= cv2.VideoCapture(0) # initializes camera object
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1) #creates font for printed text

while True:
    ret, img=cam.read() #read a frame from the camera
    img=cv2.resize(img,(340,220)) #resize it to make processing faster

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #convert image to HSV
    #******RFI: STILL NOT SURE WHAT THIS MEANS*******
    
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound) #creates a mask to cover all pixels not in the range
    
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)
