from ultralytics import YOLO

# ベースモデルとして軽量な yolov8n を使用
model = YOLO('yolov8n.pt')

# 学習
results = model.train(
    data='/home/ubuntu/ros2_ws/src/Image-processing/learning_yolo/datasets/traffic_light_1/data.yaml',
    epochs=3,
    imgsz=640,
    batch=4,        # バッチサイズを小さく
    workers=2,      # ワーカースレッド数を削減
    amp=False,      # 自動混合精度を無効化
    device='cpu'    # CPU を使用
)

