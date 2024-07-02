#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import os

class ObjectIdentificationNode:
    def __init__(self):
        rospy.init_node('object_identification_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.image_callback)

        # Paths to the YOLO files
        yolo_path = os.path.expanduser("~/catkin_ws/src/Fusion_description/yolo")
        weights_path = os.path.join(yolo_path, "yolov3.weights")
        config_path = os.path.join(yolo_path, "yolov3.cfg")
        names_path = os.path.join(yolo_path, "coco.names")

        # Print paths for debugging
        print("Weights path:", weights_path)
        print("Config path:", config_path)
        print("Names path:", names_path)

        # Load YOLO model
        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            return

        # Load class names
        try:
            with open(names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        except Exception as e:
            rospy.logerr(f"Failed to load class names: {e}")
            return

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.detect_objects(frame)
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def detect_objects(self, img):
        try:
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    color = self.colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Object Identification", img)
            cv2.waitKey(3)
        except Exception as e:
            rospy.logerr(f"Error in detect_objects: {e}")

if __name__ == '__main__':
    try:
        ObjectIdentificationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


