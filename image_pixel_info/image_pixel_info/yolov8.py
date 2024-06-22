import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

# YOLOv8 モデルの読み込み
model = YOLO('yolov8x.pt')  # 小さなモデルの例
model.classes = [9]  # クラス9（トラフィックライト）のみを検出する

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
        results = model.predict(image, stream=True)  # 推論を実行
        img = self.return_pedestrian_traffic_light(results)
        
        if isinstance(img, np.ndarray):
            red_blue_imgs = self.extract_red_blue_area(img)
            return self.return_traffic_light_signal(red_blue_imgs), img
        else:
            return None, None

    def return_pedestrian_traffic_light(self, results):
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # バウンディングボックス
            confidences = result.boxes.conf.cpu().numpy()  # 信頼度
            classes = result.boxes.cls.cpu().numpy()  # クラス

            for box, cls, conf in zip(boxes, classes, confidences):
                if int(cls) == 9:  # クラス9（トラフィックライト）
                    x1, y1, x2, y2 = map(int, box)
                    cropped_img = result.orig_img[y1:y2, x1:x2]
                    if cropped_img.shape[0] > cropped_img.shape[1]:  # 縦長の画像なら
                        return cropped_img
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

        # RGBからHSVへ変換
        upper_hsv = cv2.cvtColor(upper_img, cv2.COLOR_BGR2HSV)
        lower_hsv = cv2.cvtColor(lower_img, cv2.COLOR_BGR2HSV)

        # 赤色の範囲を定義 (信号機の赤)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])

        # 青色の範囲を定義 (信号機の青)
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([140, 255, 255])

        # 上部画像の赤色マスク
        upper_red_mask1 = cv2.inRange(upper_hsv, red_lower1, red_upper1)
        upper_red_mask2 = cv2.inRange(upper_hsv, red_lower2, red_upper2)
        upper_red_mask = upper_red_mask1 | upper_red_mask2

        # 下部画像の青色マスク
        lower_blue_mask = cv2.inRange(lower_hsv, blue_lower, blue_upper)

        # 赤色ピクセルの割合を計算
        upper_red_pixels = cv2.countNonZero(upper_red_mask)
        lower_blue_pixels = cv2.countNonZero(lower_blue_mask)

        self.get_logger().info(f'Upper red pixels: {upper_red_pixels}, Lower blue pixels: {lower_blue_pixels}')

        # ピクセルのカウントに基づいて信号の色を決定
        if upper_red_pixels > lower_blue_pixels:
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

