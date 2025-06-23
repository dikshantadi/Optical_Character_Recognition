import cv2
import numpy as np

# Create clean test image
img = np.zeros((100, 300), dtype=np.uint8)
cv2.putText(img, "TEST123", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
cv2.imwrite("test_image.png", img)
print("Created test_image.png with text 'TEST123'")