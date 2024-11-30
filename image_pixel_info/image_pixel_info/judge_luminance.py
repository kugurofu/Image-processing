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
            '/LMS_image',  # トピック名を指定
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
            self.publisher.publish(String(data="None"))
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

        # 上部と下部の画像をグレースケールに変換し、輝度を計算
        upper_gray = cv2.cvtColor(upper_img, cv2.COLOR_BGR2GRAY)
        lower_gray = cv2.cvtColor(lower_img, cv2.COLOR_BGR2GRAY)

        # 輝度の合計を計算
        upper_brightness = np.sum(upper_gray)
        lower_brightness = np.sum(lower_gray)

        self.get_logger().info(f'Upper brightness: {upper_brightness}, Lower brightness: {lower_brightness}')

        # 明るさに基づいて信号の色を決定
        if upper_brightness > lower_brightness:
            return 'red'  # 上部が明るい場合は赤信号
        else:
            return 'blue'  # 下部が明るい場合は青信号

def main(args=None):
    rclpy.init(args=args)
    traffic_light_detector = TrafficLightDetector()
    rclpy.spin(traffic_light_detector)
    traffic_light_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # ウィンドウを閉じる

if __name__ == '__main__':
    main()

