#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')

        # パラメータの宣言
        self.declare_parameter('kernel_size', 5)
        self.declare_parameter('sigma', 0.0)
        self.declare_parameter('clip_min', -1.0)
        self.declare_parameter('clip_max', 1.0)
        self.declare_parameter('epsilon', 1e-6)

        # パラメータの取得
        self.kernel_size = self.get_parameter('kernel_size').get_parameter_value().integer_value
        self.sigma = self.get_parameter('sigma').get_parameter_value().double_value
        self.clip_min = self.get_parameter('clip_min').get_parameter_value().double_value
        self.clip_max = self.get_parameter('clip_max').get_parameter_value().double_value
        self.epsilon = self.get_parameter('epsilon').get_parameter_value().double_value

        # サブスクライバの設定
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # パブリッシャの設定
        self.publisher_reflection = self.create_publisher(Image, '/reflection_image', 10)
        self.publisher_reflection_component = self.create_publisher(Image, '/reflection_component_image', 10)
        self.publisher_luminance = self.create_publisher(Image, '/luminance_image', 10)
        self.publisher_beta = self.create_publisher(Image, '/beta_image', 10)
        self.publisher_xyz = self.create_publisher(Image, '/xyz_image', 10)  # 新規追加
        self.publisher_x = self.create_publisher(Image, '/x_image', 10)      # 新規追加
        self.publisher_y = self.create_publisher(Image, '/y_image', 10)      # 新規追加

        # cv_bridge のインスタンス化
        self.bridge = CvBridge()

        self.get_logger().info('Image Processor Node has been started.')

    def listener_callback(self, msg):
        try:
            # ROS イメージメッセージを OpenCV 画像に変換（元の画像 I(x,y)）
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # ガウシアンフィルタの適用（I'(x,y)）
        blurred_image = self.apply_gaussian_filter(cv_image)

        if blurred_image is None:
            return  # エラーが発生した場合は処理を中断

        # 反射率成分の生成（P(x,y) = log((I + epsilon) / (I' + epsilon))）
        P_normalized = self.compute_reflection_component(cv_image, blurred_image)

        if P_normalized is None:
            return  # エラーが発生した場合は処理を中断

        # 輝度 L の計算
        L_normalized = self.compute_luminance(P_normalized)

        if L_normalized is None:
            return  # エラーが発生した場合は処理を中断

        # 輝度率 β の計算
        beta_normalized = self.compute_beta(L_normalized)

        if beta_normalized is None:
            return  # エラーが発生した場合は処理を中断

        # XYZ色空間への変換
        XYZ, XYZ_normalized = self.compute_xyz(P_normalized)

        if XYZ is None or XYZ_normalized is None:
            return  # エラーが発生した場合は処理を中断

        # xy色度座標の計算
        x_normalized, y_normalized = self.compute_xy(XYZ)

        if x_normalized is None or y_normalized is None:
            return  # エラーが発生した場合は処理を中断

        # ROS イメージメッセージに変換して公開
        self.publish_images(blurred_image, P_normalized, L_normalized, beta_normalized, XYZ_normalized, x_normalized, y_normalized)
        

    def apply_gaussian_filter(self, image, kernel_size=None, sigma=None):
        """
        ガウシアンフィルタを画像に適用する関数

        :param image: 入力画像（numpy.ndarray）
        :param kernel_size: カーネルサイズ（奇数の整数）。デフォルトはself.kernel_size
        :param sigma: ガウシアン関数の標準偏差。デフォルトはself.sigma
        :return: フィルタ適用後の画像（numpy.ndarray）
        """
        try:
            if kernel_size is None:
                kernel_size = (self.kernel_size, self.kernel_size)
            else:
                kernel_size = (kernel_size, kernel_size)

            blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
            return blurred_image
        except Exception as e:
            self.get_logger().error(f'Error applying Gaussian filter: {e}')
            return None

    def compute_reflection_component(self, original_image, blurred_image):
        """
        反射率成分 P(x, y) = log((I + epsilon) / (I' + epsilon)) を計算する関数

        :param original_image: 元の画像 I(x, y)（numpy.ndarray, uint8）
        :param blurred_image: ガウシアンフィルタ適用後の画像 I'(x, y)（numpy.ndarray, uint8）
        :return: 正規化された反射率成分画像 P_normalized（numpy.ndarray, uint8）
        """
        try:
            # 画像をfloat32に変換
            I = original_image.astype(np.float32)
            I_prime = blurred_image.astype(np.float32)

            # ゼロ除算を避けるためにIとI'に小さな値を加える
            I_safe = I + self.epsilon
            I_prime_safe = I_prime + self.epsilon

            # P(x,y) の計算（各チャネルごとに計算）
            P = np.log(I_safe / I_prime_safe)

            # 無限大や NaN を処理
            P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

            # デバッグ: P の範囲をログ出力
            min_P = np.min(P)
            max_P = np.max(P)
            self.get_logger().debug(f'P range before normalization: min={min_P}, max={max_P}')

            # 必要に応じて P をクリップ
            P_clipped = np.clip(P, self.clip_min, self.clip_max)  # クリップ範囲をパラメータで調整

            # Pの値を正規化して可視化可能な範囲にスケーリング
            # 各チャネルごとに正規化を適用
            P_normalized = np.zeros_like(P)
            for c in range(3):  # B, G, R チャネル
                P_normalized[:, :, c] = cv2.normalize(
                    P_clipped[:, :, c], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # uint8 にキャスト
            P_normalized = np.uint8(P_normalized)

            return P_normalized

        except Exception as e:
            self.get_logger().error(f'Error computing reflection component: {e}')
            return None

    def compute_luminance(self, P_normalized):
        """
        輝度 L = 0.299 * PR + 0.586 * PG + 0.114 * PB を計算する関数

        :param P_normalized: 正規化された反射率成分画像 P (numpy.ndarray, uint8)
        :return: 正規化された輝度画像 L_normalized (numpy.ndarray, uint8)
        """
        try:
            # OpenCVはBGR順でデータを扱うため、チャネルの順番に注意します
            P_B = P_normalized[:, :, 0].astype(np.float32)
            P_G = P_normalized[:, :, 1].astype(np.float32)
            P_R = P_normalized[:, :, 2].astype(np.float32)

            # 輝度 L の計算
            L = 0.299 * P_R + 0.586 * P_G + 0.114 * P_B

            # L の正規化
            L_normalized = cv2.normalize(L, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            L_normalized = np.uint8(L_normalized)

            return L_normalized

        except Exception as e:
            self.get_logger().error(f'Error computing luminance: {e}')
            return None

    def compute_beta(self, L_normalized):
        """
        輝度率 β = (L - L_min) / (L_max - L_min) を計算する関数

        :param L_normalized: 正規化された輝度画像 L_normalized (numpy.ndarray, uint8)
        :return: 正規化された輝度率画像 beta_normalized (numpy.ndarray, uint8)
        """
        try:
            L_float = L_normalized.astype(np.float32)

            L_min = np.min(L_float)
            L_max = np.max(L_float)

            self.get_logger().debug(f'L range: min={L_min}, max={L_max}')

            if L_max - L_min == 0:
                self.get_logger().warning('L_max equals L_min, setting beta to 0.')
                beta = np.zeros_like(L_float, dtype=np.float32)
            else:
                beta = (L_float - L_min) / (L_max - L_min)

            # β の正規化（0〜255）
            beta_normalized = cv2.normalize(beta, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            beta_normalized = np.uint8(beta_normalized)

            return beta_normalized

        except Exception as e:
            self.get_logger().error(f'Error computing beta: {e}')
            return None

    def compute_xyz(self, P_normalized):
        """
        反射率成分 P のRGB値をXYZ色空間に変換する関数

        :param P_normalized: 正規化された反射率成分画像 P (numpy.ndarray, uint8)
        :return: 生のXYZ色空間画像 XYZ (numpy.ndarray, float32) および 正規化されたXYZ色空間画像 XYZ_normalized (numpy.ndarray, uint8)
        """
        try:
            # P_normalized を float32 に変換し、0-1の範囲にスケーリング
            P_float = P_normalized.astype(np.float32) / 255.0

            # 変換行列
            conversion_matrix = np.array([
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505]
            ], dtype=np.float32)

            # OpenCVはBGR順でデータを扱うため、チャネル順をBGRからRGBに変更
            P_rgb = cv2.cvtColor(P_normalized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # 0-1の範囲

            # RGBからXYZへの変換（ベクトル化された計算）
            XYZ = np.dot(P_rgb, conversion_matrix.T)

            # XYZの範囲を確認
            min_XYZ = np.min(XYZ)
            max_XYZ = np.max(XYZ)
            self.get_logger().debug(f'XYZ range before normalization: min={min_XYZ}, max={max_XYZ}')

            # 正規化（0-255）
            XYZ_normalized = cv2.normalize(XYZ, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            XYZ_normalized = np.uint8(XYZ_normalized)

            # 再びBGR順に変換（OpenCVはBGR順で表示するため）
            XYZ_bgr = cv2.cvtColor(XYZ_normalized, cv2.COLOR_RGB2BGR)

            return XYZ, XYZ_bgr

        except Exception as e:
            self.get_logger().error(f'Error computing XYZ: {e}')
            return None, None

    def compute_xy(self, XYZ):
        """
        生のXYZ色空間画像からxy色度座標を計算する関数

        :param XYZ: 生のXYZ色空間画像 (numpy.ndarray, float32)
        :return: 正規化されたx色度座標画像 x_normalized (numpy.ndarray, uint8) と
                 正規化されたy色度座標画像 y_normalized (numpy.ndarray, uint8)
        """
        try:
            # XYZの合計を計算
            sum_XYZ = XYZ[:, :, 0] + XYZ[:, :, 1] + XYZ[:, :, 2]
            # ゼロ除算を避けるために小さな値を加える
            epsilon = 1e-6
            sum_XYZ_safe = sum_XYZ + epsilon

            # x と y の計算
            x = XYZ[:, :, 0] / sum_XYZ_safe
            y = XYZ[:, :, 1] / sum_XYZ_safe

            # x と y を 0-255 にスケーリング
            x_normalized = cv2.normalize(x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            y_normalized = cv2.normalize(y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # uint8 にキャスト
            x_normalized = np.uint8(x_normalized)
            y_normalized = np.uint8(y_normalized)

            return x_normalized, y_normalized

        except Exception as e:
            self.get_logger().error(f'Error computing xy coordinates: {e}')
            return None, None

    def publish_images(self, blurred_image, P_normalized, L_normalized, beta_normalized, XYZ_normalized, x_normalized, y_normalized):
        """
        画像をROSメッセージに変換し、各パブリッシャを通じて公開する関数

        :param blurred_image: ガウシアンフィルタ適用後の画像 I'(x, y)（numpy.ndarray, uint8）
        :param P_normalized: 正規化された反射率成分画像 P_normalized（numpy.ndarray, uint8）
        :param L_normalized: 正規化された輝度画像 L_normalized（numpy.ndarray, uint8）
        :param beta_normalized: 正規化された輝度率画像 beta_normalized（numpy.ndarray, uint8）
        :param XYZ_normalized: 正規化されたXYZ色空間画像 XYZ_normalized (numpy.ndarray, uint8)
        :param x_normalized: 正規化されたx色度座標画像 x_normalized (numpy.ndarray, uint8)
        :param y_normalized: 正規化されたy色度座標画像 y_normalized (numpy.ndarray, uint8)
        """
        try:
            # I'(x,y) を公開（カラー画像）
            I_prime_uint8 = np.clip(blurred_image, 0, 255).astype(np.uint8)
            reflection_msg = self.bridge.cv2_to_imgmsg(I_prime_uint8, encoding='bgr8')
            self.publisher_reflection.publish(reflection_msg)
            self.get_logger().debug('Published reflection image.')

            # P(x,y) を公開（カラー画像）
            reflection_component_msg = self.bridge.cv2_to_imgmsg(P_normalized, encoding='bgr8')
            self.publisher_reflection_component.publish(reflection_component_msg)
            self.get_logger().debug('Published reflection component image.')

            # 輝度 L を公開（グレースケール画像）
            luminance_msg = self.bridge.cv2_to_imgmsg(L_normalized, encoding='mono8')
            self.publisher_luminance.publish(luminance_msg)
            self.get_logger().debug('Published luminance image.')

            # 輝度率 β を公開（グレースケール画像）
            beta_msg = self.bridge.cv2_to_imgmsg(beta_normalized, encoding='mono8')
            self.publisher_beta.publish(beta_msg)
            self.get_logger().debug('Published beta image.')

            # XYZ色空間画像を公開（カラー画像）
            if XYZ_normalized is not None:
                xyz_msg = self.bridge.cv2_to_imgmsg(XYZ_normalized, encoding='bgr8')
                self.publisher_xyz.publish(xyz_msg)
                self.get_logger().debug('Published XYZ image.')
            else:
                self.get_logger().warning('XYZ_normalized is None, skipping XYZ image publication.')

            # x色度座標を公開（グレースケール画像）
            if x_normalized is not None:
                x_msg = self.bridge.cv2_to_imgmsg(x_normalized, encoding='mono8')
                self.publisher_x.publish(x_msg)
                self.get_logger().debug('Published x chromaticity image.')
            else:
                self.get_logger().warning('x_normalized is None, skipping x image publication.')

            # y色度座標を公開（グレースケール画像）
            if y_normalized is not None:
                y_msg = self.bridge.cv2_to_imgmsg(y_normalized, encoding='mono8')
                self.publisher_y.publish(y_msg)
                self.get_logger().debug('Published y chromaticity image.')
            else:
                self.get_logger().warning('y_normalized is None, skipping y image publication.')

        except Exception as e:
            self.get_logger().error(f'Could not convert image to ROS message: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

