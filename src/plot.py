#!/usr/bin/env python

'''
Odom.py
CDT Born and CDT Zahin

This node subscribes to /odom and to the ticks and heading topics.
It then displays a graph that tracks the location of the robot
using both UTM gps tracking and odometry.

This node was run remotely.
'''

import rospy, time, math
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as pp

global x,y
global points
global trial_num
points = []


def pose_callback(data):
    global x,y
    x= data.pose.position.x
    y= data.pose.position.y

def plot():
    #rotary code here
    global x, y

    global points

    points.append([x,y])

    pp.plot(x,y, 'bo')
    pp.draw()
    pp.pause(0.000001)



    #f.write(str(seconds)+","+str(x_coord)+","+str(y_coord))

def shutdown():
    global points
    write= ""
    for p in points:
        write+= str(p[0])+','+str(p[1])+";"
    with open('/home/wborn/trials/trial_'+str(rospy.get_param('/trial_num')), 'wb') as fp:
        fp.write(write)
        fp.close()
    pass


def init():
    rospy.init_node('plot',anonymous=True)
    pp.show(block=False)

    global x,y
    x=0
    y=0

    rospy.on_shutdown(shutdown)

    rospy.Subscriber("/vrpn_client_node/RigidBody1/pose", PoseStamped, pose_callback)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        global seconds
        seconds = rospy.get_time()
        plot()
        r.sleep()

init()
