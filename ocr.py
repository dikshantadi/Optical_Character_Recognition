import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Load model
model = tf.keras.models.load_model("ocr_model.h5")

# Class names
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']

def debug_show(image, title="Debug", delay=500):
    cv2.imshow(title, image)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def preprocess_digit(char_img):
    """Enhanced character preprocessing"""
    # Add border padding
    char_img = cv2.copyMakeBorder(char_img, 15, 15, 15, 15, 
                                 cv2.BORDER_CONSTANT, value=0)
    
    # Resize preserving aspect ratio
    h, w = char_img.shape
    if h > w:
        char_img = cv2.copyMakeBorder(char_img, 0, 0, (h-w)//2, (h-w)//2, 
                                    cv2.BORDER_CONSTANT, value=0)
    else:
        char_img = cv2.copyMakeBorder(char_img, (w-h)//2, (w-h)//2, 0, 0,
                                    cv2.BORDER_CONSTANT, value=0)
    
    # Resize and normalize
    char_img = cv2.resize(char_img, (28, 28))
    char_img = char_img.astype(np.float32) / 255.0
    
    # Debug visualization
    debug_show((char_img * 255).astype(np.uint8), "Preprocessed Character")
    
    return char_img.reshape(1, 28, 28, 1)

def perform_ocr(image_path="test_image.png"):
    print("\n=== Starting OCR Processing ===")
    
    # 1. Load and verify image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image")
    debug_show(img, "Original Image")
    
    # 2. Advanced binarization
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )
    debug_show(binary, "After Thresholding")
    
    # 3. Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    debug_show(cleaned, "After Noise Removal")
    
    # 4. Improved contour detection
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Found {len(contours)} potential characters")
    
    # 5. Character processing
    characters = []
    min_height = 20  # Minimum character height in pixels
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Validate character size and aspect ratio
        if h < min_height or w/h > 2.5 or h/w > 2.5:
            print(f"Skipping contour {i}: Invalid size {w}x{h}")
            continue
            
        # Extract and pad character
        char_img = cleaned[y:y+h, x:x+w]
        padded = cv2.copyMakeBorder(char_img, 10, 10, 10, 10,
                                  cv2.BORDER_CONSTANT, value=0)
        debug_show(padded, f"Character {i+1}")
        
        # Preprocess and predict
        processed = preprocess_digit(padded)
        prediction = model.predict(processed, verbose=0)
        
        # Get top prediction
        pred_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        char = class_names[pred_idx]
        
        print(f"Character {i+1}: Predicted '{char}' with {confidence:.1%} confidence")
        
        if confidence > 0.7:  # Only accept confident predictions
            characters.append((x, char))
    
    # 6. Generate output
    text = ' '.join([char for _, char in sorted(characters)])
    print("\n=== Final OCR Result ===")
    print(text)
    return text

if __name__ == "__main__":
    perform_ocr()