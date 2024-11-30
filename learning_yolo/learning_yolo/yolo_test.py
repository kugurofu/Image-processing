from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/last.pt')

# Predict the model
model.predict('/home/ubuntu/ros2_ws/src/Image-processing/image_pixel_info/images/202406141157_1.jpg', save=True, conf=0.1)

