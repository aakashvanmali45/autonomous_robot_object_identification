#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleAvoidance:
    def __init__(self):
        rospy.init_node('obstacle_avoidance_node', anonymous=True)

        # Publisher to the robot's velocity command topic
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber to the LIDAR scan topic
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # Velocity command message
        self.twist = Twist()

        # Parameters
        self.min_distance = 0.7  # Minimum distance to obstacle

        rospy.loginfo("Obstacle avoidance node started")

    def scan_callback(self, data):
        # Process LIDAR data to find the minimum distance
        min_distance = min(data.ranges)

        # Obstacle avoidance logic
        if min_distance < self.min_distance:
            # Obstacle detected, turn the robot
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.7
            
        elif min_distance < self.min_distance:
            # Obstacle detected, turn the robot
            self.twist.linear.x = 0.0
            self.twist.angular.z = -0.7
            
        else:
            # Path is clear, move forward
            self.twist.linear.x = 0.7
            self.twist.angular.z = 0.0

        # Publish the velocity command
        self.cmd_pub.publish(self.twist)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        obstacle_avoidance = ObstacleAvoidance()
        obstacle_avoidance.run()
    except rospy.ROSInterruptException:
        pass
