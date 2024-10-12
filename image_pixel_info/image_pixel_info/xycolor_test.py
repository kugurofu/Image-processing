import cv2
import numpy as np

def convert_xyz(filename):
    # 画像をカラーで読み込む
    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    # RGB色空間からXYZ色空間に変換
    cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    
    # XYZチャンネルを分離
    X, Y, Z = cvt_img[:, :, 0], cvt_img[:, :, 1], cvt_img[:, :, 2]

    # XYZ値を正規化
    total = X + Y + Z
    total[total == 0] = 1e-10  # ゼロ除算を避けるための微小値を追加
    x_normalized = X / total
    y_normalized = Y / total
    z_normalized = Z / total

    # 正規化されたXYZ値を新しい画像にまとめる
    normalized_img = cv2.merge((x_normalized, y_normalized, z_normalized))

    return normalized_img

cvt_img = convert_xyz("/home/ubuntu/ros2_ws/src/Image-processing/image_pixel_info/images/test2.jpg")

# 画像を0-255の範囲にスケーリング
cvt_img_scaled = np.clip(cvt_img * 255, 0, 255).astype(np.uint8)

cv2.imshow("Normalized XYZ", cvt_img_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

