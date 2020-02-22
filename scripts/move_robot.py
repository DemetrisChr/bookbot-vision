#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from time import time

def laser_scan_callback(data):
    print data.ranges

def read_laser_scan_data():
    rospy.Subscriber('scan',LaserScan,laser_scan_callback)

def move_motor(fwd,ang):
    pub = rospy.Publisher('cmd_vel',Twist,queue_size = 10)
    mc = Twist()
    mc.linear.x = fwd
    mc.angular.z = ang
    pub.publish(mc)

# Adjustment value ranges from -100 to 100
def adjust_robot_position(adjustment_value):
    rospy.init_node('example_script',anonymous=True)

    start_time = time()
    duration = 1 #in seconds

    forward_speed = adjustment_value * 0.65 / 100
    turn_speed = 0

    while time()<start_time+duration:
        try:
            read_laser_scan_data()
            move_motor(forward_speed,turn_speed)
        except rospy.ROSInterruptException:
            pass
    else:
        move_motor(0,0)


if __name__ == '__main__':
    rospy.init_node('example_script',anonymous=True)

    start_time = time()
    duration = 5 #in seconds

    forward_speed = 1
    turn_speed = 0

    while time()<start_time+duration:
        try:
            read_laser_scan_data()
            move_motor(forward_speed,turn_speed)
        except rospy.ROSInterruptException:
            pass
    else:
        move_motor(0,0)
