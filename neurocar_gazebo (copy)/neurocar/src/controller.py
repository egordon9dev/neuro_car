#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin
import numpy as np
from getkey import getkey, keys

bridge = CvBridge()

class Attributes:
    def __init__(self):
        self.acceleration = 1
        self.rotation = 1
        self.speed = 0
        self.angle = 0

class Movement:
    def __init__(self, attributes):
        #rospy.init_node('move_robot_node', anonymous=False)
        #self.pub_move = rospy.Publisher("/cmd_vel",Twist,queue_size=10)
        self.move = Twist()
        self.attr = attributes

    def publish_vel(self):
        self.pub_move.publish(self.move)

    def moving(self):
        self.move.linear.x = self.attr.speed
        self.move.angular.z = self.attr.angle

    def move_forward(self):
        self.attr.speed += self.attr.acceleration
        self.moving()

    def move_backward(self):
        self.attr.speed -= self.attr.acceleration
        self.moving()

    def rotate_right(self):
        self.attr.angle += self.attr.rotation
        self.moving()

    def rotate_left(self):
        self.attr.angle -= self.attr.rotation
        self.moving()

    def stop(self):        
        self.move.linear.x=0
        self.move.angular.z=0.0
    

def callback(img):
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)

    ndarray = np.asarray(cv_image)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

def main():
    sub = rospy.Subscriber('neurocar/camera/image_raw', Image, callback)
    pub = rospy.Publisher('neurocar/cmd_vel', Twist, queue_size=10)
    rospy.init_node('controller', anonymous=True)

    attributes = Attributes()
    mov = Movement(attributes)
    rate = rospy.Rate(2) # 10hz

    while not rospy.is_shutdown():
        #movement = raw_input('Enter desired movement: ')
        key = getkey()
        if key == 'w':
            mov.move_forward()
        elif key == 's':
            mov.move_backward()
        elif key == 'a':
            mov.rotate_left()
        elif key == 'd':
            mov.rotate_right()

        pub.publish(mov.move)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

