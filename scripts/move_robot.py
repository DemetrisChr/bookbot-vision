import rospy
import sys

sys.path.insert(1, '/home/pi/vision/msgs')
from geometry_msgs.msg import Twist

from time import time, sleep

class MoveRobot():
    def __init__(self):
        rospy.init_node('move_robot', anonymous=True)
        self.pub = rospy.Publisher('om_with_tb3/cmd_vel', Twist, queue_size=10)
        self.mc = Twist()

    def setSpeed(self, speed):
        rospy.Rate(1).sleep()
        self.mc.linear.x = speed
        self.pub.publish(self.mc)

    def adjustSpeed(self, adjustment_value):
        # speed = - adjustment_value * 0.65 / 1000
        if abs(adjustment_value) < 5:
            speed = 0
        else:
            speed = -0.01 * (adjustment_value / 100)
        self.setSpeed(speed)

    def shutDown(self):
        self.setSpeed(0)
        rospy.signal_shutdown('SHUTDOWN')


if __name__ == '__main__':
    """
    adjustment_values = list(range(-100, 101, 2))
    mv = MoveRobot()
    for adj_value in adjustment_values:
        mv.adjustSpeed(adj_value)
    """
    start_time = time()
    mv = MoveRobot()
    mv.adjustSpeed(50)
    sleep(2)
    mv.adjustSpeed(20)
    sleep(2)
    mv.adjustSpeed(0)
    sleep(2)
    mv.adjustSpeed(-25)
    sleep(2)
    mv.adjustSpeed(0)
    mv.shutDown()
