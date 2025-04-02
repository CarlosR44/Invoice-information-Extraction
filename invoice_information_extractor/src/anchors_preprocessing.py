import numpy as np
import os
import json
from sklearn.cluster import KMeans

path_labels = r'invoice_information_extractor\dataset\train\labels'

def load_yolo_boxes(path):
    
    boxes = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            for l in f.readlines():
                values = list(map(float, l.split()))
                if len(values) != 5:
                   print(f"Advertencia: {file} tiene una línea con {len(values)} valores en lugar de 5: {l}")
                   continue
                w, h = values[3], values[4]
                boxes.append([float(w), float(h)])
    
    return boxes

boxes_wh = load_yolo_boxes(path_labels)

def find_anchors(boxes, clusters = 9):
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(boxes)
    anchors = kmeans.cluster_centers_
    return anchors

anchors = find_anchors(boxes_wh)
print("Nuevos anchors:\n", anchors)

img_sze = 640
scaled_anchors = (anchors * img_sze).astype(int).tolist()
print("Anchors escalados para YOLO:", scaled_anchors)

with open('scaled_anchors.json', 'w') as f:
    json.dump(scaled_anchors, f)
