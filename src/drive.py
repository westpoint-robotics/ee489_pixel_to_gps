#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import String

buttons = [0,0,0,0,0,0,0,0,0,0,0]

data = []

current='x'

def callback(data):
    global buttons
    buttons = data.buttons

class GoForward():


    def __init__(self):
        # initiliaze
        rospy.init_node('GoForward', anonymous=False)
        self.drive_pub = rospy.Publisher("/output/drive_out",String)

        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)




        #TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(60);

        # Twist is a datatype for velocity
        move_cmd = Twist()

        global current
        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
            # let's go forward at 0.2 m/s
            # subscribe to joy nod
            rospy.Subscriber("joy", Joy, callback)
            rospy.loginfo(buttons[0])
            if buttons[0] == 1:
                data.append('x')
                current = 'x'
                rospy.logcinfo("Stop")
                move_cmd.linear.x = 0
                # turn at 0 radians/s
                move_cmd.angular.z = 0
            elif buttons[1] == 1:
                data.append('r')
                current = 'r'
                rospy.loginfo("right")
                move_cmd.linear.x = 0.1
                # turn at -1 radians/s
                move_cmd.angular.z = -.4
            elif buttons[2] == 1:
                data.append('l')
                current = 'l'
                rospy.loginfo("left")
                move_cmd.linear.x = 0.1
                # turn at 1 radians/s
                move_cmd.angular.z = .4
            else:
                data.append('s')
                current = 's'
                rospy.loginfo("Straight")
                move_cmd.linear.x = 0.1
                # let's turn at 0 radians/s
                move_cmd.angular.z = 0
            if buttons[3] == 1:
                 self.shutdown();
            # publish the velocity
            self.cmd_vel.publish(move_cmd)
            self.drive_pub.publish(current)
            # wait for 0.1 seconds (10 HZ) and publish again
            r.sleep()



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
