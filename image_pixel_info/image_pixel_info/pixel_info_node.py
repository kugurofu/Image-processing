import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np  # numpy をインポート
import cv2  # OpenCV をインポート
import colorsys

class PixelInfoNode(Node):

    def __init__(self):
        super().__init__('pixel_info_node')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.x = 100  # 調査したいピクセルのx座標
        self.y = 100  # 調査したいピクセルのy座標

    def image_callback(self, msg):
        # 画像メッセージをOpenCVの画像に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 画像のサイズを取得
        height, width, _ = cv_image.shape

        # 座標が画像サイズ内にあるか確認
        if self.x < 0 or self.x >= width or self.y < 0 or self.y >= height:
            self.get_logger().error('Coordinates are out of bounds')
            return

        # 指定した座標の画素値を取得
        b, g, r = cv_image[self.y, self.x]
        rgb = np.array([[[r, g, b]]], dtype=np.uint8)  # numpy配列に変換

        # RGB値をログに出力
        self.get_logger().info(f'RGB: ({r}, {g}, {b})')

        # RGBをHSVに変換
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = hsv[0, 0]

        # HSV値をログに出力
        self.get_logger().info(f'Hue: {hue}')
        self.get_logger().info(f'Saturation: {saturation}')
        self.get_logger().info(f'Value (Brightness): {value}')

        # RGBをHSLに変換
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)

        # HSL値を0-1から0-255にスケール変換してログに出力
        self.get_logger().info(f'Lightness: {int(l * 255)}')

def main(args=None):
    rclpy.init(args=args)
    node = PixelInfoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

