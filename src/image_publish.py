#!/usr/bin/python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("/turtle_follow/output/image_raw",Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/turtle_follow/usb_cam_node/image_raw",Image,self.callback)
    print("done.")

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (100, 100))
    resized_image = resized_image[50:100, 0:100]
    resized_image = cv2.resize(resized_image, (50, 50))

    cv2.waitKey(3)
    #rospy.loginfo(current)
    try:
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(resized_image, "mono8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
