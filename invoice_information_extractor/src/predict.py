import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO(r"runs\detect\train20\weights\best.pt") 
img_path = r'invoice_information_extractor\dataset\test\resized\X51005337867.jpg'
predict = model(img_path, conf = 0.1, iou=0.6, imgsz = 800)
image = cv2.imread(img_path)

def get_color(idx):
    np.random.seed(idx)
    return tuple(np.random.randint(0, 255, 3).tolist())
h, w, _ = image.shape


box_thickness = max(1, int(min(w, h) / 300))  

font_scale = max(0.3, min(w, h) / 5000)  
font_thickness = max(1, int(min(w, h) / 5000))

for result in predict:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        color = get_color(class_id)


        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)


        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x, text_y = x1, y1 - 5
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), color, -1)
        cv2.putText(image, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

scale_percent = 80  
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

cv2.imshow('Predictions', image_resized)
cv2.imwrite("output.jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.waitKey(0)
cv2.destroyAllWindows()

