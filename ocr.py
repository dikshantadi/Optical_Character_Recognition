import cv2
import numpy as np 
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("ocr_model.h5")

# Define the label map: index -> character
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']

def preprocess_digit(cropped_img):
    resized = cv2.resize(cropped_img, (28, 28))
    normalized = resized / 255.0
    return normalized.reshape(1, 28, 28, 1)

def perform_ocr(image_path="preprocessed.jpg", output_txt="output.txt"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read preprocessed image")
    
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours with stricter parameters
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    characters = []
    min_height = 15  # Minimum character height to consider
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter small contours and very wide contours (likely not characters)
        if h > min_height and w/h < 3:
            char_img = img[y:y+h, x:x+w]
            
            # Add white border to maintain aspect ratio
            border_size = 5
            char_img = cv2.copyMakeBorder(
                char_img,
                border_size,
                border_size,
                border_size,
                border_size,
                cv2.BORDER_CONSTANT,
                value=0
            )
            
            # Resize to 28x28 preserving aspect ratio
            rows, cols = char_img.shape
            if rows > cols:
                factor = 28.0/rows
                cols = int(cols*factor)
                rows = 28
            else:
                factor = 28.0/cols
                rows = int(rows*factor)
                cols = 28
                
            char_img = cv2.resize(char_img, (cols, rows))
            
            # Center in 28x28 canvas
            delta_w = 28 - cols
            delta_h = 28 - rows
            top = delta_h//2
            bottom = delta_h - top
            left = delta_w//2
            right = delta_w - left
            char_img = cv2.copyMakeBorder(
                char_img,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=0
            )
            
            # Normalize and predict
            char_img = char_img.astype(np.float32)/255.0
            char_img = np.expand_dims(char_img, axis=(0, -1))
            
            prediction = model.predict(char_img, verbose=0)
            predicted_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Only accept predictions with sufficient confidence
            if confidence > 0.7:  # Adjust threshold as needed
                predicted_char = class_names[predicted_idx]
                characters.append((x, y, w, h, predicted_char))
    
    # Sort characters left-to-right, top-to-bottom
    characters.sort(key=lambda c: (c[1], c[0]))
    
    # Reconstruct text
    text = ""
    prev_y = -1
    for x, y, w, h, char in characters:
        if prev_y != -1 and abs(y - prev_y) > h/2:
            text += "\n"
        text += char
        prev_y = y
    
    with open(output_txt, "w") as f:
        f.write(text)
    
    return text

if __name__ == "__main__":
    result = perform_ocr("preprocessed.jpg", "output.txt")
    print("Detected text:\n", result)