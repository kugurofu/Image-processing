#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Retinex_Image_Node(Node):
    def __init__(self):
        super().__init__('Retinex_Image_Node')
        
        # パラメータの宣言
        self.declare_parameter('variance', 300)
        
        # パラメータの取得
        self.variance = self.get_parameter('variance').get_parameter_value().integer_value
        
        # サブスクライバの設定
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # パブリッシャの設定
        self.publisher_retinex = self.create_publisher(Image, '/retinex_image', 10)
        
        # cv_bridge のインスタンス化
        self.bridge = CvBridge()

        self.get_logger().info('Retinex Image Node has been started.')
    
    def listener_callback(self, msg):
        try:
            # ROS イメージメッセージを OpenCV 画像に変換（元の画像 I(x,y)）
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return
            
        # retinex処理
        img_retinex = self.SSR(img, self.variance)
        
        # publish image
        self.publish_images(img_retinex)
        
    def singleScaleRetinex(self, img, variance):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
        return retinex
    
    def SSR(self, img, variance):
        img = np.float64(img) + 1.0
        # 各チャネルに対して Retinex 処理を適用
        img_retinex = self.singleScaleRetinex(img, variance)
        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            for u, c in zip(unique, count):
                if u == 0:
                    zero_count = c
                    break            
            # 低値と高値を決定
            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break            
            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
            img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                   (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                   * 255
        # export uint8
        img_retinex = np.uint8(img_retinex)        
        return img_retinex
        
    def publish_images(self, img_retinex):
        try:
            # I'(x,y) を公開（カラー画像）
            # I_prime_uint8 = np.clip(img_retinex, 0, 255).astype(np.uint8)
            retinex_msg = self.bridge.cv2_to_imgmsg(img_retinex, encoding='bgr8')
            self.publisher_retinex.publish(retinex_msg)
            self.get_logger().debug('Published retinex image.')
        except Exception as e:
            self.get_logger().error(f'Could not convert image to ROS message: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = Retinex_Image_Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



## otehon
"""      
def singleScaleRetinex(img,variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex

def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex

   

def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex



def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break            
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break            
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
        
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)        
    return img_retinex


variance_list=[15, 80, 30]
variance=300
    
img = cv2.imread('/home/ubuntu/ros2_ws/src/Image-processing/image_pixel_info/images/test3.jpg')
img_msr=MSR(img,variance_list)
img_ssr=SSR(img, variance)

cv2.imshow('Original', img)
cv2.imshow('MSR', img_msr)
cv2.imshow('SSR', img_ssr)
cv2.imwrite('SSR.jpg', img_ssr)
cv2.imwrite('MSR.jpg',img_msr)


cv2.waitKey(0)
cv2.destroyAllWindows()
"""
