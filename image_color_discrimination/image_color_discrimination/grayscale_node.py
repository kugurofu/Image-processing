import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',  # 画像トピック名を適切なものに変更してください
            self.image_callback,
            10)
        self.subscription  # メッセージが届かないように、購読オブジェクトを保持しておく必要があります
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # ROSのImageメッセージをOpenCVの画像に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 抽出する四角形の頂点を定義
        pts1 = np.array([[337, 86], [337, 108], [357, 108], [357, 86]], np.int32)
        pts2 = np.array([[337, 126], [337, 149], [357, 149], [357, 126]], np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        pts2 = pts2.reshape((-1, 1, 2))

        # 四角形の領域を抽出
        mask1 = np.zeros_like(cv_image)
        mask2 = np.zeros_like(cv_image)
        cv2.fillPoly(mask1, [pts1], (255, 255, 255))
        cv2.fillPoly(mask2, [pts2], (255, 255, 255))
        masked_image1 = cv2.bitwise_and(cv_image, mask1)
        masked_image2 = cv2.bitwise_and(cv_image, mask2)

        # グレースケールに変換
        gray_image1 = cv2.cvtColor(masked_image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(masked_image2, cv2.COLOR_BGR2GRAY)

        # 黒い領域をカットして座標(0,0)に移動
        x1, y1, w1, h1 = cv2.boundingRect(pts1)
        x2, y2, w2, h2 = cv2.boundingRect(pts2)
        gray_image1 = gray_image1[y1:y1+h1, x1:x1+w1]
        gray_image2 = gray_image2[y2:y2+h2, x2:x2+w2]

        # グレースケール画像を表示
        cv2.imshow('Masked Gray Image 1', gray_image1)
        cv2.imshow('Masked Gray Image 2', gray_image2)
        cv2.waitKey(1)  # キー入力待ち（1ミリ秒）
        
        # 四角形の範囲内の輝度の平均を計算
        mean1 = np.mean(gray_image1)
        mean2 = np.mean(gray_image2)
        print("1つ目の四角形の輝度平均:", mean1)
        print("2つ目の四角形の輝度平均:", mean2)
        
        # メッシュでグレースケール画像をプロット
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        x1, y1 = np.meshgrid(range(gray_image1.shape[1]), range(gray_image1.shape[0]))
        x2, y2 = np.meshgrid(range(gray_image2.shape[1]), range(gray_image2.shape[0]))

        ax1.plot_surface(x1, y1, gray_image1, cmap='gray', vmin = 0, vmax=255)
        ax1.set_title('Gray Image 1')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Luminance')
        ax1.set_zlim(0,255)

        ax2.plot_surface(x2, y2, gray_image2, cmap='gray', vmin = 0, vmax=255)
        ax2.set_title('Gray Image 2')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Luminance')
        ax2.set_zlim(0,255)

        plt.show()
        

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

