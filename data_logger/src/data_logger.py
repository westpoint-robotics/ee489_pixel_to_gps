#!/usr/bin/env python
import cv2
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry

c1_x = 0.0
c1_y = 0.0
c1_area = 0.0
c2_x = 0.0
c2_y = 0.0
c2_area = 0.0
c3_x = 0.0
c3_y = 0.0
c3_area = 0.0
c4_x = 0.0
c4_y = 0.0
c4_area = 0.0

def listener():
	rospy.init_node('data_logger', anonymous = True)
	#rospy.Subscriber('c1_pixelCoordinates', Float64MultiArray, c1_data)
	rospy.Subscriber('c2_pixelCoordinates', Float64MultiArray, c2_data)
	rospy.Subscriber('c3_pixelCoordinates', Float64MultiArray, c3_data)
	#rospy.Subscriber('c4_pixelCoordinates', Float64MultiArray, c4_data)
	rospy.Subscriber('/odom', Odometry, gps_data)
	"""
        f = open("/home/user1/Desktop/logged_data.csv", "w")
	while not rospy.is_shutdown():
		if lastGPS_x != UTM_x or lastGPS_y != UTM_y:
                        dataString = str(c1_x) + "," + str(c1_y) + "," + str(c1_area) + "," + str(c2_x) + "," + str(c2_y) + "," + str(c2_area) + "," + str(UTM_x) + "," + str(UTM_y) + "\n" 
            
                        f.write(dataString)
                        print(c1_x)
                        lastGPS_x = UTM_x
                        lastGPS_y = UTM_y
        f.close()
        """



def c1_data(data1):
	global c1_x
	global c1_y
	global c1_area
	c1_x = data1.data[1]
	c1_y = data1.data[2]
	c1_area = data1.data[3]
        
def c2_data(data2):
	global c2_x
	global c2_y
	global c2_area
	c2_x = data2.data[1]
	c2_y = data2.data[2]
	c2_area = data2.data[3]

def c3_data(data3):
	global c3_x
	global c3_y
	global c3_area
	c3_x = data3.data[1]
	c3_y = data3.data[2]
	c3_area = data3.data[3]

def c4_data(data4):
	global c4_x
	global c4_y
	global c4_area
	c4_x = data4.data[1]
	c4_y = data4.data[2]
	c4_area = data4.data[3]

def gps_data(data):
	global c1_x
	global c1_y
	global c1_area
	global c2_x
	global c2_y
	global c2_area
	global c3_x
	global c3_y
	global c3_area
	global c4_x
	global c4_y
	global c4_area
	
    	camera1_x = "0"
    	camera1_y = "0"
    	camera1_z = "0"
    	camera1_angle = "0"
    	camera2_x = "0"
    	camera2_y = "0"
    	camera2_z = "0"
        camera2_angle = "0"
    	camera3_x = "0"
    	camera3_y = "0"
    	camera3_z = "0"
        camera3_angle = "0"
    	camera4_x = "0"
    	camera4_y = "0"
    	camera4_z = "0"
        camera4_angle = "0"

	UTM_x = data.pose.pose.position.x
	UTM_y = data.pose.pose.position.y

    	dataString = str(UTM_x) + "," + str(UTM_y) + "," + camera2_x + "," + camera2_y + "," + camera2_z + "," + camera2_angle + "," + str(c2_x) + "," + str(c2_y) + "," + str(c2_area) + "," + camera3_x + "," + camera3_y + "," + camera3_z + "," + camera3_angle + "," + str(c3_x) + "," + str(c3_y) + "," + str(c3_area) + "\n" 
        print(dataString)
	with open("/home/user1/Desktop/logged_data_trial2.csv",'a') as file:
                file.write(dataString)
                
	

if __name__ == '__main__':
	listener()
	
        rospy.spin() 
