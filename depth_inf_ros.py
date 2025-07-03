#!/usr/bin/env python3
import rospy
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthPoseDetector:
    def __init__(self):
        rospy.init_node('yolov8_from_depth_image', anonymous=True)

        # Parameters
        self.model_path = rospy.get_param('~model_path', '/proj/pan_ws/src/posture_recognition/scripts/runs/pose/train4/weights/best.pt')
        self.image_topic = rospy.get_param('~image_topic', '/depth_image')
        self.visualize = rospy.get_param('~visualize', True)

        # Load model and bridge
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        # ROS
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/pose/inference_result", Image, queue_size=1)

        rospy.loginfo("YOLOv8 custom depth model loaded.")

        rospy.spin()

    def image_callback(self, msg):
        try:
            # Read 16UC1 depth image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # shape: HxW, dtype: uint16

            # Normalize to 0–255 and convert to uint8
            depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_gray = depth_norm.astype(np.uint8)  # shape: HxW

            # Convert to 3 channels (1-channel replicated)
            input_image = cv2.merge([depth_gray, depth_gray, depth_gray])  # shape: HxWx3

            # Run YOLOv8 inference
            results = self.model(input_image)[0]
            annotated = results.plot()

            keypoints = results.keypoints

            if keypoints.conf is not None:
                for i, confs in enumerate(keypoints.conf):
                    avg_conf = confs.mean()
                    rospy.loginfo(f"Detection {i}: Avg keypoint confidence = {avg_conf:.3f}")
            else:
                rospy.logwarn("No keypoints or confidence found.")

            # Publish result
            ros_image = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            ros_image.header = msg.header
            self.image_pub.publish(ros_image)

        except Exception as e:
            rospy.logerr(f"Inference error: {e}")

if __name__ == "__main__":
    try:
        DepthPoseDetector()
    except rospy.ROSInterruptException:
        pass
