import cv2
import numpy as np

def preprocessing(path):
    #resizing
    image = cv2.imread(path)
    target_width = 1000
    h, w = image.shape[:2]
    target_height = int((target_width / w) * h)
    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Sharpen before thresholding (so edges are stronger in binary image)
    sharpen_kernel = np.array([[0, -1, 0],
                            [-1, 5.5, -1],
                            [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        sharpened, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,  
        21,  
        10   
    )
    #_, adaptive = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    output_path = "preprocessed.jpg"
    cv2.imwrite(output_path, adaptive)
    return output_path

