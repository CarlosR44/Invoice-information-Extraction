import cv2 as cv 
import numpy as np
import os 

import os
import cv2

# Directorios de entrada y salida
input_folder = r"invoice_information_extractor\dataset\test\img"
output_folder = r"invoice_information_extractor/dataset/test/resized"


os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error cargando {filename}, se omitirá.")
            continue
        h, w = img.shape[:2]
        new_w = 640
        new_h = int((new_w / w) * h)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_img)
        
        print(f"Guardada: {output_path}")