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

current="x"
num=0
class image_converter:

  def __init__(self):
    global num
    trial_num = rospy.get_param('/trial_num')
    rospy.loginfo("Started trial number: "+str(trial_num))
    self.image_pub = rospy.Publisher("/output/image_raw",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

    self.drive_sub= rospy.Subscriber("/output/drive_out",String,self.callback1)
    print("done.")


  def callback1(self,data):
    global current
    current = data

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    resized_image = cv2.resize(cv_image, (50, 50))

    cv2.waitKey(3)
    #rospy.loginfo(current)
    try:
      global num

      rospy.loginfo("Published image.")
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(resized_image, "bgr8"))
      if str(current)[7] != 'x':
          num+=1
          pub_string = str(current)[7]+"/img_"+str(num)+"_"+str(current)[7]+".png"
          rospy.loginfo("Wrote image: "+pub_string)
          cv2.imwrite( pub_string , resized_image );
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
    print("number last was "+num)
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
