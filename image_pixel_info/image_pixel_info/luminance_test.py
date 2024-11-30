import cv2
import numpy as np

def calculate_luminance(filename):
    # 画像をカラーで読み込む
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"画像を読み込めませんでした: {filename}")
        sys.exit(1)
    
    # BGRから輝度を計算（より効率的な方法）
    luminance = np.dot(img[...,:3], [0.114, 0.587, 0.299])
    
    # 輝度を8ビットに変換し、値をクリップ
    luminance = np.clip(luminance, 0, 255).astype(np.uint8)
    
    return luminance

def calculate_luminance_ratio(luminance):
    # 最大・最小輝度値を取得
    max_luminance = np.max(luminance)
    min_luminance = np.min(luminance)
    
    # 各ピクセルの輝度を最大・最小輝度で割り、割合を計算
    luminance_ratio = (luminance.astype(np.float32) - min_luminance) / (max_luminance - min_luminance)
    
    # 見やすくするために0～255にスケーリング
    luminance_ratio_scaled = (luminance_ratio * 255).astype(np.uint8)
    
    return luminance_ratio_scaled

luminance = calculate_luminance("/home/ubuntu/ros2_ws/src/Image-processing/image_pixel_info/images/test3.jpg")
Luminance_ratio = calculate_luminance_ratio(luminance)
cv2.imshow('Luminance', luminance)
cv2.imshow('Luminance_ratio', Luminance_ratio)

cv2.waitKey(0)
cv2.destroyAllWindows()
