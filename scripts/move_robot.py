import rospy
import sys

sys.path_insert('/home/pi/vision/msgs')
from geometry_msgs.msg import Twist


class MoveRobot():
    def __init__(self):
        rospy.init_node('move_robot', anonymous=True)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.mc = Twist()

    def setSpeed(self, speed):
        self.mc.linear.x = speed
        self.pub.publish(self.mc)

    def adjustSpeed(self, adjustment_value):
        speed = adjustment_value * 0.65 / 100
        self.setSpeed(speed)


if __name__ == '__main__':
    adjustment_values = list(range(-100, 101, 2))
    mv = MoveRobot()
    for adj_value in adjustment_values:
        mv.adjustSpeed(adj_value)
