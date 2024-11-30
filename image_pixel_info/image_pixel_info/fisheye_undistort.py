import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class FisheyeUndistortNode(Node):
    def __init__(self):
        super().__init__('fisheye_undistort')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, '/camera/undistorted_image', 10)
        self.br = CvBridge()

        # 魚眼カメラのキャリブレーションパラメータを設定
        # これらのパラメータはキャリブレーションから得られたものです
        self.K = np.array([[230.3124073007367, 0.8902268984725723, 366.0547101660642], [0, 230.3903534611963, 367.3569954767625], [0.0,0.0,1.0]])
        self.D = np.array([[-0.008173216167374261, -0.05337521624866415, 0.06211795804586437, -0.02542501550578094]])
        self.R = np.eye(3)  # 回転行列 R（ここでは単位行列を仮に使用）
        self.P = np.array([[115.1562036503684, 0.8902268984725723, 366.0547101660642, 0], [0, 115.1951767305981, 367.3569954767625, 0], [0, 0, 1, 0]]) # 投影行列 P（ここでは仮に使用）
        self.dim = (736, 736)  # 画像のサイズ

        # 歪み補正と再投影用の変換マップを生成
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, self.R, self.P, self.dim, cv2.CV_16SC2)

    def listener_callback(self, msg):
        # ROS2 の Image メッセージを OpenCV 画像に変換
        cv_image = self.br.imgmsg_to_cv2(msg, 'bgr8')

        # 歪み補正
        undistorted_image = cv2.remap(cv_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 歪み補正後の画像を ROS2 の Image メッセージに変換してパブリッシュ
        undistorted_msg = self.br.cv2_to_imgmsg(undistorted_image, 'bgr8')
        self.publisher.publish(undistorted_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FisheyeUndistortNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

