#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan

class LidarDataPrinter:
    def __init__(self):
        rospy.init_node('lidar_data_printer_node', anonymous=True)
        
        # Updated to subscribe to the '/scan' topic
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        rospy.loginfo("Lidar data printer node started and listening to /scan topic")

    def scan_callback(self, data):
        # Print LIDAR data to the terminal
        rospy.loginfo("Received LIDAR data:")
        rospy.loginfo("Ranges: %s", data.ranges)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        lidar_data_printer = LidarDataPrinter()
        lidar_data_printer.run()
    except rospy.ROSInterruptException:
        pass

