import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge

# YOLOv5 モデルの読み込み
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
model.classes = [9]  # トラフィックライトのみを検出する

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # トピック名を指定
            self.image_callback,
            10)
        self.publisher = self.create_publisher(String, 'traffic_light_status', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.get_logger().info('Received image')
        
        # ROS Image メッセージを OpenCV 画像に変換
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # トラフィックライトを検出する
        status, pedestrian_light_img = self.detect_traffic_light(cv_image)
        
        if status:
            self.publisher.publish(String(data=status))
            self.get_logger().info(f'Published traffic light status: {status}')
            
            if pedestrian_light_img is not None:
                # 歩行者用信号機の画像を表示
                cv2.imshow('Pedestrian Traffic Light', pedestrian_light_img)
                cv2.waitKey(1)  # ウィンドウの更新
        else:
            self.get_logger().info('No pedestrian traffic light detected')

    def detect_traffic_light(self, image):
        results = model(image)
        img = self.return_pedestrian_traffic_light(results)
        
        if isinstance(img, np.ndarray):
            red_blue_imgs = self.extract_red_blue_area(img)
            return self.return_traffic_light_signal(red_blue_imgs), img
        else:
            return None, None

    def return_pedestrian_traffic_light(self, results):
        if len(results.crop()) == 0:
            return None

        for traffic_light in results.crop():
            img = traffic_light['im']
            img_shape = img.shape

            if img_shape[0] > img_shape[1]:  # 縦長の画像なら
                return img
        return None

    def extract_red_blue_area(self, img):
        img_shape = img.shape
        w_c = int(img_shape[1] / 2)
        s = int(img_shape[1] / 6)
        upper_h_c = int(img_shape[0] / 4)
        lower_h_c = int(img_shape[0] * 3 / 4)

        return [
            img[upper_h_c - s:upper_h_c + s, w_c - s:w_c + s, :],
            img[lower_h_c - s:lower_h_c + s, w_c - s:w_c + s, :]
        ]

    def return_traffic_light_signal(self, img_list):
        upper_img = img_list[0]
        lower_img = img_list[1]

        upper_red_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:, :, 0].mean()
        upper_blue_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:, :, 2].mean()
        upper_delta = abs(upper_red_nums - upper_blue_nums)

        lower_red_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:, :, 0].mean()
        lower_blue_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:, :, 2].mean()
        lower_delta = abs(lower_red_nums - lower_blue_nums)

        if upper_delta >= lower_delta:
            return 'red'
        else:
            return 'blue'

def main(args=None):
    rclpy.init(args=args)
    traffic_light_detector = TrafficLightDetector()
    rclpy.spin(traffic_light_detector)
    traffic_light_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # ウィンドウを閉じる

if __name__ == '__main__':
    main()

