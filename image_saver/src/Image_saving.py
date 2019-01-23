#!/usr/bin/env python
import rospy 
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2

global state, num, trial_num, data_set, bridge
bridge = CvBridge()
num=0
data_set="test"
current = False
state= False
trial_num = 0

def image_callback(data):
    global num, trial_num, data_set, bridge
    try:
        trial_num+=1
        if trial_num % 40 == 0:
            cv_image = bridge.imgmsg_to_cv2(data,"bgr8")
            pub_string = "/home/user1/Pictures/Goose_Pictures/"+str(data_set)+str(trial_num)+".png"
            cv2.imwrite( pub_string , cv_image )
            print trial_num
    except CvBridgeError as e:
        print(e)

if __name__ == '__main__':
    rospy.init_node('drive_node', anonymous=True)
    rospy.Subscriber("/camera_array/cam0/image_raw", Image, image_callback)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pass
    rate.sleep()
    
