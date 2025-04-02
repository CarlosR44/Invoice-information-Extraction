import os
import cv2
import pandas as pd
from src.ocr_process import ocr_trocr
from ultralytics import YOLO

trocr = ocr_trocr
model_path = r"runs\detect\train20\weights\best.pt"
model = YOLO(model_path)
image_path = r"invoice_information_extractor\dataset\test\package"
excel_path = r"C:\Users\cbeff\ProyectosIA\invoice_information_extractor\facturas_extraidas.xlsx"

def process_image(image_path):
   image = cv2.imread(image_path)
   if image is None:
       print(f'image could not be read')   
   else:
       print(f'Image processing')
   return image

def process_images(image_path):
    
    if os.path.isfile(image_path):
       return [image_path]
        
    elif os.path.isdir(image_path):
        return [os.path.join(image_path, file_name) for file_name in os.listdir(image_path)
                 if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print(f'dir is incorrect')
        return []

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def ocr_extraction(model, image_path):
    
    images = process_images(image_path)
    extracted_data = []
    
    for img_path in images:
        image = process_image(img_path)
        if image is None:
            continue

        results = model(img_path, conf = 0.1, imgsz = 800)
        data_dict = {
            "Company": "",
            "Address": "",
            "Date": "",
            "Cashier": "",
            "Document_number": "",
            "Product_code": [],
            "Product_description": [],
            "Amount-RM-": [],
            "Units": [],
            "Unit_price": [],
            "Total": [],
            "SubTotal": [],
            "Cash": [],
            "Change": [],
            "Discount": [],
        }
        
        for result in results:
           for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            
            roi = image[y1:y2, x1:x2]
            roi = preprocess_image(roi)
            text = trocr(roi).strip()
            
            if label in ["Company", "Address", "Date", "Cashier", "Document_number", "Cash", "Change", "Discount", "SubTotal", "Total"]:
               data_dict[label] = text

            elif label in ["Product_code", "Product_description", "Amount-RM-", "Units", "Unit_price"]:
               data_dict[label].append(text)
        
        max_length = max(
            len(data_dict[key]) for key in ["Product_code", "Product_description", "Amount-RM-", 
                                            "Units", "Unit_price"])
        for key in ["Product_code", "Product_description", "Amount-RM-", "Units", "Unit_price"]:
             while len(data_dict[key]) < max_length:
               data_dict[key].append("")
            
        extracted_data.append(data_dict)
             
    df = pd.DataFrame(extracted_data)
    df.to_excel(excel_path, index=False)
    print(f"Excel file saved in: {excel_path}")    
    return df

df_results = ocr_extraction(model, image_path) 
print(df_results)  

