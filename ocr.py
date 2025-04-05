import pytesseract
import cv2
import numpy as np 

preprocessed_img = cv2.imread("preprocessed.jpg", cv2.IMREAD_COLOR_BGR)

text = pytesseract.image_to_string(preprocessed_img)
print(text)
