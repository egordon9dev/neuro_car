#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from math import sin
from math import log
import numpy as np
from getkey import getkey, keys

bridge = CvBridge()

class Attributes:
    def __init__(self):
        self.maxAngle = 5
        self.maxSpeed = 5
        self.acceleration = 0.5
        self.rotation = 2.5
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

    def speedFunc(self, speed):
        return 3*(speed**(1/3))

    def execute(self):
        self.move.linear.x = self.attr.speed**2 #self.speedFunc(self.attr.speed)
        self.move.angular.z = self.attr.angle
        #print(self.move.linear.x)

    def move_car(self, speed):
        if (speed > 0 and self.attr.speed + speed * self.attr.acceleration <= self.attr.maxSpeed):
            self.attr.speed += speed * self.attr.acceleration
        elif (speed < 0 and self.attr.speed + speed * self.attr.acceleration >= 0):
            self.attr.speed += speed * self.attr.acceleration

        if (self.attr.speed < 0):
            self.attr.speed = 0
        self.execute()

    def move_none(self):
        self.execute()

    def rotate(self, angle):
        if (angle > 0 and self.attr.angle + self.attr.rotation <= self.attr.maxAngle):
            self.attr.angle += angle * self.attr.rotation
        elif (angle < 0 and self.attr.angle + self.attr.rotation >= -self.attr.maxAngle):
            self.attr.angle += angle * self.attr.rotation
        self.execute()

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
            mov.move_car(1)
        elif key == 's':
            mov.move_car(-1)
        
        if key == 'a':
            mov.rotate(-1)
        elif key == 'd':
            mov.rotate(1)

        pub.publish(mov.move)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


