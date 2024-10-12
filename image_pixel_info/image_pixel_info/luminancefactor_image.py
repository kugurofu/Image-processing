#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Luminance_Image_Node(Node):
    def __init__(self):
        super().__init__('Luminance_Image_Node')
        
        # パラメータの宣言
        self.declare_parameter('variance', 300)
        
        # パラメータの取得
        self.variance = self.get_parameter('variance').get_parameter_value().integer_value
        
        # サブスクライバの設定
        self.subscription = self.create_subscription(
            Image,
            '/retinex_image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # パブリッシャの設定
        self.publisher_retinex = self.create_publisher(Image, '/luminancefactor_image', 10)
        
        # cv_bridge のインスタンス化
        self.bridge = CvBridge()

        self.get_logger().info('Luminance Image Node has been started.')
    
    def listener_callback(self, msg):
        try:
            # ROS イメージメッセージを OpenCV 画像に変換（元の画像 I(x,y)）
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
            
        # retinex処理
        img_luminance = self.SSR(img, self.variance)
        
        # publish image
        self.publish_images(img_retinex)
