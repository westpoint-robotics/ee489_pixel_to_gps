#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import String

buttons = [0,0,0,0,0,0,0,0,0,0,0]

data = []

current='x'

def callback(data):
    global move_cmd
    global cmd_vel
    print (str(data)[7])
    if str(data)[7] == 's':
        move_cmd.linear.x = .25
        move_cmd.angular.z = 0
    elif str(data)[7] == 'r':
        move_cmd.linear.x = 0.25
        move_cmd.angular.z = -1
    elif str(data)[7] == 'l':
        move_cmd.linear.x = 0.25
        move_cmd.angular.z = 1
    else:
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0
    cmd_vel.publish(move_cmd)

class GoForward():


    def __init__(self):
        # initiliaze
        rospy.init_node('GoForward', anonymous=False)
        rospy.Subscriber("/turtle_follow/output/drive_out", String, callback)
        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        global cmd_vel
        cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)




        #TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(60);

        # Twist is a datatype for velocity
        global move_cmd
        move_cmd = Twist()

        global current
        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
            pass



    def shutdown(self):
        print(data)
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        self.drive_pub.publish("x")
        # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
        # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)

if __name__ == '__main__':
    #try:
    GoForward()
    #except:
        #rospy.loginfo("GoForward node terminated.")
