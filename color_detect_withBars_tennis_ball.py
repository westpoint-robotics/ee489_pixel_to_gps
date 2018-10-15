import cv2
import numpy as np

#'optional' argument is required for trackbar creation parameters
def nothing(x):
    pass

# Capture the input frame from webcam
def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame = cap.read()

    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.5
    

    cv2.namedWindow('Colorbars') #Create a window named 'Colorbars'
    
 
    #assign strings for ease of coding
    hh='Hue High'
    hl='Hue Low'
    sh='Saturation High'
    sl='Saturation Low'
    vh='Value High'
    vl='Value Low'
    wnd = 'Colorbars'
    #Begin Creating trackbars for each
    cv2.createTrackbar(hl, 'Colorbars',0,255,nothing)
    cv2.createTrackbar(hh, 'Colorbars',0,255,nothing)
    cv2.createTrackbar(sl, 'Colorbars',0,255,nothing)
    cv2.createTrackbar(sh, 'Colorbars',0,255,nothing)
    cv2.createTrackbar(vl, 'Colorbars',0,255,nothing)
    cv2.createTrackbar(vh, 'Colorbars',0,255,nothing)

    # Iterate until the user presses ESC key
    frame_count = 0
    pixelCoords = np.zeros((1,2))
    while True:
        frame = get_frame(cap, scaling_factor)
        # Capture frame-by-frame
        #ret, frame = cap.read()

        # Convert the HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #read trackbar positions for each trackbar
        hul=cv2.getTrackbarPos(hl, 'Colorbars')
        huh=cv2.getTrackbarPos(hh, 'Colorbars')
        sal=cv2.getTrackbarPos(sl, 'Colorbars')
        sah=cv2.getTrackbarPos(sh, 'Colorbars')
        val=cv2.getTrackbarPos(vl, 'Colorbars')
        vah=cv2.getTrackbarPos(vh, 'Colorbars')

        #make array for final values
        #HSVLOW=np.array([hul,sal,val])
        #HSVHIGH=np.array([huh,sah,vah])

        # Define 'blue' range in HSV colorspace
        #lower = np.array([60,100,100])
        #upper = np.array([180,255,255])

        # Threshold the HSV image to get only blue color
        #mask = cv2.inRange(hsv, lower, upper)

        #create a mask for that range
        #mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
        mask = cv2.inRange(hsv,np.array([49,40,131]),np.array([100,134,235]))
        #optimal values for tennis ball:
        #h = [19,107], s = [40,134], v = [126,255]

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        res = cv2.medianBlur(res, 5)
       

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #cv2.drawContours(frame,contours,-1,(255,0,0),3)
        #for i in range(len(contours)):
         #   x,y,w,h=cv2.boundingRect(contours[i])
          #  cv2.rectangle(res,(x,y),(x+w,y+h),(0,0,255), 2)
            #cv2.putText((frame), str(i+1),(x,y+h),font,(0,255,255))
        areas = [cv2.contourArea(c) for c in contours]
        if len(contours) > 0:
            max_index = np.argmax(areas)
            myBox = contours[max_index]
            x,y,w,h = cv2.boundingRect(myBox)
            #print(cv2.boundingRect(myBox))
            
            cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255),1)
        

            #PIXEL COORDINATES OF OBJECT!!!
            
            pixelCoords = np.concatenate((pixelCoords,np.array([[x+0.5*w, y + 0.5*h]])))
            print(pixelCoords[frame_count])
            
            frame_count += 1


        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector', res)
        print(np.shape(pixelCoords))
        #cv2.imshow('Object',hsv)

        # Check if the user pressed ESC key
        c = cv2.waitKey(5)
        if c == 27:
            break
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    
