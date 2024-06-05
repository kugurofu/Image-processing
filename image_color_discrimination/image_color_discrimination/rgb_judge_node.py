import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Int32, 'rgb_judge', 10)
        self.bridge = CvBridge()
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Extract the specified regions
        region1 = cv_image[86:108, 334:357]
        region2 = cv_image[126:149, 334:357]
        
        # Calculate the average R value for region1 and region2
        mean_r1 = np.mean(region1[:, :, 2])
        mean_r2 = np.mean(region2[:, :, 2])

        # Publish the judgment
        judge_msg = Int32()
        if mean_r1 >= 150 and mean_r2 <= 200:
            judge_msg.data = 1
        else:
            judge_msg.data = 0
        self.publisher.publish(judge_msg)

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

