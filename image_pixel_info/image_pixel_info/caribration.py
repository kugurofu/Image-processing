import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import glob

class FisheyeCalibrationNode(Node):
    def __init__(self):
        super().__init__('fisheye_calibration')
        self.declare_parameter('calibration_images_path', '/home/ubuntu/ros2_ws/src/Image-processing/image_pixel_info/calibration_images/*.jpg')
        self.declare_parameter('checkerboard_shape', [6, 8])  # 6x9のチェスボードなら [5, 8]
        self.run_calibration()

    def run_calibration(self):
        # パラメータの読み込み
        images_path = self.get_parameter('calibration_images_path').get_parameter_value().string_value
        self.get_logger().info(f"Using images path: {images_path}")
        checkerboard_shape = self.get_parameter('checkerboard_shape').get_parameter_value().integer_array_value
        CHECKERBOARD = tuple(checkerboard_shape)

        # チェスボードの角点を保存するリスト
        objpoints = []  # 3Dの世界座標
        imgpoints = []  # 2Dの画像平面座標

        # チェスボードの3D座標（固定された実際のサイズ）
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        # キャリブレーション用の画像が保存されているディレクトリ
        images = glob.glob(images_path)
        self.get_logger().info(f"Found images: {images}")

        if not images:
            self.get_logger().error("No calibration images found. Please check the path and file pattern.")
            return

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                self.get_logger().error(f"Failed to load image: {fname}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # チェスボードのコーナーを検出
            ret, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                self.get_logger().info(f"Chessboard corners found in image: {fname}")
                objpoints.append(objp)
                imgpoints.append(corners)

                # コーナーを描画
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            else:
                self.get_logger().warn(f"Failed to find chessboard corners in image: {fname}")
                # 検出失敗時の画像を表示
                cv2.imshow('Failed to find corners', img)
                cv2.waitKey(1000)  # 1秒間表示

        cv2.destroyAllWindows()

        if not objpoints or not imgpoints:
            self.get_logger().error("Failed to find chessboard corners in any of the images. Please ensure the images contain a visible chessboard.")
            return

        # キャリブレーションを実行
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs, tvecs = [], []
        R = np.eye(3)  # 回転行列の初期化
        P = np.zeros((3, 4))  # 投影行列の初期化

        ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + 
            # cv2.fisheye.CALIB_CHECK_COND + 
            cv2.fisheye.CALIB_FIX_SKEW,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

        if ret:
            self.get_logger().info(f"Calibration succeeded with RMS error: {ret}")

            # R と P を求める
            # R は恒等行列、P は K を使用して拡張される
            R = np.eye(3)  # 単位行列
            P[:3, :3] = K  # カメラ行列を P にコピー
            P[:3, 3] = 0   # 4列目をゼロにする

            self.get_logger().info(f"K (Camera Matrix):\n{K}")
            self.get_logger().info(f"D (Distortion Coefficients):\n{D}")
            self.get_logger().info(f"R (Rotation Matrix):\n{R}")
            self.get_logger().info(f"P (Projection Matrix):\n{P}")
        else:
            self.get_logger().error("Calibration failed.")

def main(args=None):
    rclpy.init(args=args)
    node = FisheyeCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

