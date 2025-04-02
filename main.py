import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread("resources/AI.jpg", cv2.IMREAD_COLOR_BGR)

target_width = 600
h, w = image.shape[:2]
target_height = int((target_width / w) * h)

image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA) 
image1 = cv2.GaussianBlur(image, (3, 3), 1)

cv2.imshow("resize for now", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()