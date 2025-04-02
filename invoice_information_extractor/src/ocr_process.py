import torch
import numpy
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(device)

def ocr_trocr(roi):
    
    roi_pill = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    pixel_values = processor(roi_pill, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generate_ids = ocr_model.generate(pixel_values)
    
    text = processor.batch_decode(generate_ids, skip_special_tokens = True)[0]
    
    return text.strip()

