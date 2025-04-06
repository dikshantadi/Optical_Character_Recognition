import pytesseract
import cv2
import numpy as np 

def perform_ocr(image_path, output_txt="output.txt"):
    img = cv2.imread(image_path)
    text = pytesseract.image_to_string(img, config='--psm 6')
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)
    return text

