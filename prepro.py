import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("resources/sh.jpg", cv2.IMREAD_COLOR)

# to resize the image
target_width = 700
h, w = image.shape[:2]
target_height = int((target_width / w) * h)
image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0) #reduces noise
#blurred_image = cv2.fastNlMeansDenoising(blurred_image, h=10)

#adaptive thresholding
img_adaptive_mean = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5
)
img_edge = cv2.Canny(img_adaptive_mean, 100, 200)

cv2.imshow("Adaptive Thresholding", img_edge)
cv2.waitKey(0)
cv2.destroyAllWindows()