import os
import json
from ultralytics import YOLO

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
data_path = os.path.join(project_path, 'invoice_information_extractor', 'dataset', 'data.yaml')
weights_path = r'runs\detect\train38\weights\best.pt' #17 va para 18

def train_yolov8 (data_path, epochs = 100, batch_size = 16, img_size=640, model_name = 'yolov8s.pt', weights_path = None):
    
    if not os.path.exists(data_path):
        print(f'El {data_path} dont exists')
        return
    

    
    model_name = weights_path if weights_path and os.path.exists(weights_path) else "yolov8s.pt"
    model = YOLO(model_name)
    
    model.train(
    data=data_path,
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    device=0,
    workers=4,
    amp=False,
    optimizer="AdamW",
    cls = 0.6,
    box = 8.5,
    iou = 0.6,
    kobj = 1.5,
    mosaic=0.0,
    mixup=0.0,
    weight_decay=0.0009, 
    lr0= 0.0001,
    lrf= 0.00001, 
    flipud=0.0,
    fliplr=0.3,
    hsv_h=0.013,
    hsv_s=0.3,
    hsv_v=0.2,
    degrees=3.0, 
    translate=0.01, 
    scale=0.05,
    shear=1.0,
    perspective=0.0,
    copy_paste=0.0,
    auto_augment="none",
    erasing=0.0,
    crop_fraction=1.0,
    freeze=10
    )
    
if __name__ == '__main__':
    train_yolov8(data_path, epochs=600, batch_size=8, img_size=640, weights_path= weights_path)