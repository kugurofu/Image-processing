import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error('Error converting ROS Image to OpenCV image: %s' % str(e))
            return

        # Get RGB value at specified coordinates (x, y)
        x, y = 100, 100  # Example coordinates, you can change this
        b, g, r = cv_image[y, x]

        # Compare R and G values
        if r > g:
            # If red is higher than green, set the pixel to red
            cv_image[y, x] = (0, 0, 255)  # Set pixel to red (BGR format)
        else:
            # If green is higher than red, set the pixel to green
            cv_image[y, x] = (0, 255, 0)  # Set pixel to green (BGR format)

        # Display the modified image
        cv2.imshow('Processed Image', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

