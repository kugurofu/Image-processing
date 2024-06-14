import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MJPGCameraPublisher(Node):
    def __init__(self):
        super().__init__('mjpg_camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.cap = cv2.VideoCapture('/dev/webcam')
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.timer_callback)

        # MJPG フォーマットの設定
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.cap.isOpened():
            self.get_logger().error('カメラデバイスを開けません')
            rclpy.shutdown()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # フレームを RGB に変換してパブリッシュ
            image_message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(image_message)
        else:
            self.get_logger().warning('フレームを取得できません')

def main(args=None):
    rclpy.init(args=args)
    node = MJPGCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

