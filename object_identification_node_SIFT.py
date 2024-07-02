import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import os

class ObjectIdentificationNodeSIFT:
    def __init__(self):
        rospy.init_node('object_identification_node_SIFT', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.image_callback)

        # Directory with reference images
        images_dir = "~/catkin_ws/src/Fusion_description/images"
        images_dir = os.path.expanduser(images_dir)

        self.reference_images = []
        self.reference_keypoints = []
        self.reference_descriptors = []
        self.reference_names = []

        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Load reference images and compute their keypoints and descriptors
        for filename in os.listdir(images_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(images_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    keypoints, descriptors = self.sift.detectAndCompute(img, None)
                    self.reference_images.append(img)
                    self.reference_keypoints.append(keypoints)
                    self.reference_descriptors.append(descriptors)
                    self.reference_names.append(filename.split('.')[0])
                    rospy.loginfo(f"Loaded reference image: {filename}")

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.detect_objects(frame)
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def detect_objects(self, img):
        try:
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.sift.detectAndCompute(gray_frame, None)

            if descriptors is None:
                rospy.logwarn("No descriptors found for current frame.")
                return

            for ref_img, ref_kp, ref_desc, ref_name in zip(self.reference_images, self.reference_keypoints, self.reference_descriptors, self.reference_names):
                if ref_desc is None:
                    rospy.logwarn(f"No descriptors found for reference image {ref_name}.")
                    continue
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=10) 
                
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(ref_desc, descriptors, k=2)
 

                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                if len(good_matches) > 10:
                    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    if len(src_pts) < 4 or len(dst_pts) < 4:
                    	continue

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    h, w = ref_img.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    dst = np.int32(dst)
                    rospy.loginfo(f"Detected object '{ref_name}' at position: {dst}")
        
        except Exception as e:
            rospy.logerr(f"Error in detect_objects: {e}")


if __name__ == '__main__':
    try:
        ObjectIdentificationNodeSIFT()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()

