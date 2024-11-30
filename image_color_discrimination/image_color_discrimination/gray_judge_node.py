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
            'image_raw',  # 画像トピック名を適切なものに変更してください
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(Int32, '/gray_judge', 10)
        self.subscription  # メッセージが届かないように、購読オブジェクトを保持しておく必要があります
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # ROSのImageメッセージをOpenCVの画像に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 抽出する四角形の頂点を定義
        pts1 = np.array([[334, 86], [334, 108], [357, 108], [357, 86]], np.int32)
        pts2 = np.array([[334, 126], [334, 149], [357, 149], [357, 126]], np.int32)
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

        # 輝度の平均値を計算
        mean1 = np.mean(gray_image1)
        mean2 = np.mean(gray_image2)

        # 判定結果をPublish
        msg = Int32()
        if mean1 > mean2:
            msg.data = 1
        else:
            msg.data = 0
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

