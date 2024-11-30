#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class XYcolor_Image_Node(Node):
    def __init__(self):
        super().__init__('XYcolor_Image_Node')
        
        # サブスクライバの設定
        self.subscription = self.create_subscription(
            Image,
            '/retinex_image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # パブリッシャの設定
        self.publisher_XYcolor = self.create_publisher(Image, '/XYcolor_image', 10)
        
        # cv_bridge のインスタンス化
        self.bridge = CvBridge()

        self.get_logger().info('XYcolor Image Node has been started.')
    
    def listener_callback(self, msg):
        try:
            # ROS イメージメッセージを OpenCV 画像に変換（元の画像 I(x,y)）
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
        
        # 輝度計算
        XYcolor = self.calculate_XYcolor(img)
            
        # 輝度率計算
        luminance_ratio_scaled = self.calculate_XYcolor(XYcolor)
        
        # publish image
        self.publish_images(XYcolor)
        
    def calculate_luminance(self, img):    
        # BGRから輝度を計算（より効率的な方法）
        luminance = np.dot(img[...,:3], [0.114, 0.587, 0.299])
    
        # 輝度を8ビットに変換し、値をクリップ
        luminance = np.clip(luminance, 0, 255).astype(np.uint8)
    
        return luminance

    def calculate_luminance_ratio(self, luminance):
        # 最大・最小輝度値を取得
        max_luminance = np.max(luminance)
        min_luminance = np.min(luminance)
    
        # 各ピクセルの輝度を最大・最小輝度で割り、割合を計算
        luminance_ratio = (luminance.astype(np.float32) - min_luminance) / (max_luminance - min_luminance)
    
        # 見やすくするために0～255にスケーリング
        luminance_ratio_scaled = (luminance_ratio * 255).astype(np.uint8)
    
        return luminance_ratio_scaled    
    
    def publish_images(self, img_XYcolor):
        try:
            #XYを公開（カラー画像）
            XYcolor_msg = self.bridge.cv2_to_imgmsg(img_XYcolor, encoding='bgr8')
            self.publisher_XYcolor.publish(XYcolor_msg)
            self.get_logger().debug('Published luminance image.')
        except Exception as e:
            self.get_logger().error(f'Could not convert image to ROS message: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = XYcolor_Image_Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
