import pytesseract
import cv2
import numpy as np 
import tensorflow as tf

model = tf.keras.models.load_model("cnn_ocr.h5")

def preprocess_digit(cropped_img):
    resized = cv2.resize(cropped_img, (28, 28))
    normalized = resized / 255.0
    return normalized.reshape(1, 28, 28, 1)

def perform_ocr(image_path="preprocessed.jpg", output_txt="output.txt"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10:
            digit_img = img[y:y+h, x:x+w]
            input_digit = preprocess_digit(digit_img)
            prediction = model.predict(input_digit, verbose=0)
            digit = np.argmax(prediction)
            digits.append((x, y, digit))

    # Sort by y (rows), then x (left-to-right)
    digits.sort(key=lambda tup: (tup[1]//50, tup[0]))  # group by ~50px vertical spacing

    text = ""
    current_line = -1
    for x, y, digit in digits:
        line = y // 50
        if line != current_line:
            text += "\n"  # new row
            current_line = line
        text += str(digit) + " "

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)

    return text
