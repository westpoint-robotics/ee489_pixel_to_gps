# Instructions for teleoperation of the GVR-Bot 
Istructions for using an Xbox controller to command the GVR-Bot

### Steps for configuring tele-operation control
1.  Install the ROS Joy package
    + `sudo apt-get install ros-kinetic-joy` 
2.  Install the ROS Teleop-Twist package 
    + `sudo apt-get install ros-kinetic-teleop-twist-joy` 
3.  Copy the ROS package called 'gvr_teleop' that is inside this repo into the /src directory of your catkin workspace 
    + From your catkin_ws directory, perform a `catkin_make`
    + Then, perform a `rospack profile` command
4.  Execute the launch file
    + `roslaunch gvr_teleop drive_joy.launch`
    + If necessary, modify the launch file parameters to change the behavior of the joystick commands to the robot.
   
    
   
            
    
              


