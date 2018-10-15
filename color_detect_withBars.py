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
        HSVLOW=np.array([hul,sal,val])
        HSVHIGH=np.array([huh,sah,vah])

        # Define 'blue' range in HSV colorspace
        #lower = np.array([60,100,100])
        #upper = np.array([180,255,255])

        # Threshold the HSV image to get only blue color
        #mask = cv2.inRange(hsv, lower, upper)

        #create a mask for that range
        mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        #res = cv2.medianBlur(res, 5)
	res = cv2.GaussianBlur(res, (3, 3), 0)
		
	(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
	#contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	if len(cnts) > 0:
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
            rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)
            #cv2.circle(frame, (rect[3][0], rect[3][1]), 7, (255, 255, 255), -1)
            centerX = abs((rect[0][0] - rect[3][0]) / 2 + rect[0][0])
            centerY = abs((rect[1][1] - rect[0][1]) / 2 + rect[0][1])
            
		
	'''
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        cv2.drawContours(frame,contours,-1,(255,0,0),3)
        for i in range(len(contours)):
            x,y,w,h=cv2.boundingRect(contours[i])
            cv2.rectangle(res,(x,y),(x+w,y+h),(0,0,255), 2)
            #cv2.cv.PutText(cv2.cv.fromarray(frame), str(i+1),(x,y+h),font,(0,255,255))
	'''
		
			
        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector', res)
        #cv2.imshow('Object',hsv)

        # Check if the user pressed ESC key
        c = cv2.waitKey(5)
        if c == 27:
            break
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
