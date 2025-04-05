import cv2
import numpy as np

#resize
image = cv2.imread("resources/dsp.jpg")
target_width = 700
h, w = image.shape[:2]
target_height = int((target_width / w) * h)
image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3, 3), 0)

# Sharpen before thresholding (so edges are stronger in binary image)
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

# Adaptive Thresholding
adaptive = cv2.adaptiveThreshold(
    sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

cv2.imwrite("preprocessed.jpg", adaptive)