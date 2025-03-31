import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread("resources/AI.jpg", cv2.IMREAD_COLOR_BGR)

plt.imshow(image, cmap= "gray")
plt.show()