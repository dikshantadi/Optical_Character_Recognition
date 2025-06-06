import cv2
import numpy as np 
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("cnn_ocr.h5")

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
    # Read and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours of characters
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    characters = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10:
            char_img = img[y:y+h, x:x+w]
            input_char = preprocess_digit(char_img)
            prediction = model.predict(input_char, verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_char = class_names[predicted_index]
            characters.append((x, y, predicted_char))

    # Sort detected characters row-wise, then left to right
    characters.sort(key=lambda tup: (tup[1] // 50, tup[0]))  # group by row, then column

    # Reconstruct text
    text = ""
    current_line = -1
    for x, y, char in characters:
        line = y // 50
        if line != current_line:
            text += "\n"
            current_line = line
        text += char + " "

    # Write to output file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text.strip())

    return text.strip()

# Example usage
if __name__ == "__main__":
    result = perform_ocr("preprocessed.jpg", "output.txt")
    print("Detected text:\n", result)