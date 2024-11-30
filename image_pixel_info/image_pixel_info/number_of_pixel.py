import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 画像を読み込む
image_path = "waypoint_map_003.jpeg"  # 画像ファイルのパス
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. 対象の色範囲を指定 (例: 赤色)
lower_bound = np.array([200, 0, 0])  # 赤色の下限（BGR）
upper_bound = np.array([255, 50, 50])  # 赤色の上限（BGR）

# マスクを作成
mask = cv2.inRange(image, lower_bound, upper_bound)

# 3. ピクセル数をカウント
pixel_count = cv2.countNonZero(mask)
print(f"矢印のピクセル数: {pixel_count}")

# 4. 結果を表示（オプション）
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Mask Image")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()

