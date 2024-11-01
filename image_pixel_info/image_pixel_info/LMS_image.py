#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import skimage.exposure

class ChromaticAdaptationNode(Node):
    def __init__(self):
        super().__init__('ChromaticAdaptationNode')

        # パラメータの宣言
        self.declare_parameter('adaptation_factor', 1.0)
        
        # パラメータの取得
        self.adaptation_factor = self.get_parameter('adaptation_factor').get_parameter_value().double_value

        # サブスクライバの設定
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # パブリッシャの設定
        self.publisher_adapted = self.create_publisher(Image, '/LMS_image', 10)
        
        # cv_bridge のインスタンス化
        self.bridge = CvBridge()

        self.get_logger().info('Chromatic Adaptation Node has been started.')

    def listener_callback(self, msg):
        try:
            # ROS イメージメッセージを OpenCV 画像に変換
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') / 255.0
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # 白色点の推定とクロマチック適応処理
        src_white_point = self.ave_srgb(img)
        dst_white_point = np.array([1.0, 1.0, 1.0])
        adapted_img = self.chromatic_adaptation(src_white_point, dst_white_point, img, self.adaptation_factor)
        
        # 公開用画像の変換と送信
        self.publish_images(adapted_img)

    def srgb_to_lrgb(self, srgb): 
        return skimage.exposure.adjust_gamma(srgb, 2.2)
    
    def lrgb_to_srgb(self, linear): 
        return skimage.exposure.adjust_gamma(np.clip(linear, 0.0, 1.0), 1 / 2.2)
        
    def srgb_to_xyz(self, srgb): 
        RGB_TO_XYZ = np.array(
            [[0.41245, 0.35758, 0.18042],
             [0.21267, 0.71516, 0.07217],
             [0.01933, 0.11919, 0.95023]]
        )
        return self.srgb_to_lrgb(srgb) @ RGB_TO_XYZ.T    

    def xyz_to_srgb(self, xyz): 
        XYZ_TO_RGB = np.array(
            [[3.24048, -1.53715, -0.49854],
             [-0.96926,  1.71516,  0.07217],
             [0.01933,  0.11919,  0.95023]]
        )
        return self.lrgb_to_srgb(xyz @ XYZ_TO_RGB.T)

    def xyz_to_lms(self, xyz, M): 
        return xyz @ M.T

    def normalize_xyz(self, xyz): 
        return xyz / xyz[1]
    
    def ave_srgb(self, img): 
        return self.lrgb_to_srgb(self.srgb_to_lrgb(img).mean((0, 1)))

    def chromatic_adaptation(self, src_white_point, dst_white_point, src_img, adapt): 
        src_img_xyz = self.srgb_to_xyz(src_img)
        xyz_src = self.normalize_xyz(self.srgb_to_xyz(src_white_point))
        xyz_dst = self.normalize_xyz(self.srgb_to_xyz(dst_white_point))
        XYZ_TO_LMS = np.array(
            [[ 0.733,  0.430, -0.162],
             [-0.704,  1.698,  0.006],
             [ 0.003,  0.014,  0.983]]
        )
        lms_src = self.xyz_to_lms(xyz_src, XYZ_TO_LMS)
        lms_dst = self.xyz_to_lms(xyz_dst, XYZ_TO_LMS)
        g = (adapt * lms_dst + (1.0 - adapt) * lms_src) / lms_src
        adapt_mat = np.linalg.inv(XYZ_TO_LMS) @ np.diag(g) @ XYZ_TO_LMS
        adapt_xyz = src_img_xyz @ adapt_mat.T
        return self.xyz_to_srgb(adapt_xyz)
    
    def publish_images(self, img_adapted):
        try:
            adapted_msg = self.bridge.cv2_to_imgmsg((img_adapted * 255).astype(np.uint8), encoding='rgb8')
            self.publisher_adapted.publish(adapted_msg)
            self.get_logger().debug('Published chromatic adapted image.')
        except Exception as e:
            self.get_logger().error(f'Could not convert image to ROS message: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ChromaticAdaptationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

