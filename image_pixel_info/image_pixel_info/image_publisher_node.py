import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ament_index_python.packages import get_package_share_directory  # 追加

class ImagePublisherNode(Node):

    def __init__(self):
        super().__init__('image_publisher_node')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bridge = CvBridge()
        # パッケージの共有ディレクトリから画像ファイルのパスを取得
        package_share_directory = get_package_share_directory('image_pixel_info')
        self.image_path = os.path.join(package_share_directory, 'images', 'test3.jpg')
        self.cv_image = cv2.imread(self.image_path)

        if self.cv_image is None:
            self.get_logger().error(f"Could not open or find the image: {self.image_path}")
        else:
            self.get_logger().info(f"Loaded image: {self.image_path}")

    def timer_callback(self):
        if self.cv_image is not None:
            # OpenCVの画像をROS2の画像メッセージに変換して配信
            msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing image')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

