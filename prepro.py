import cv2
import numpy as np

def preprocessing(path):
    # Read image
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not read image file")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,  # Increased block size
        11   # Increased constant
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    output_path = "preprocessed.jpg"
    cv2.imwrite(output_path, cleaned)
    return output_path