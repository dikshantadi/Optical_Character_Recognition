import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import json

# Load model and class names
model = tf.keras.models.load_model("ocr_model.h5")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Helper: Sort by lines (top-down) and left-right within line
def sort_by_lines(boxes, line_threshold=15):
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # sort by y then x
    lines = []
    current_line = []
    for x, y, img in boxes:
        if not current_line:
            current_line.append((x, y, img))
        elif abs(y - current_line[-1][1]) < line_threshold:
            current_line.append((x, y, img))
        else:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [(x, y, img)]
    if current_line:
        lines.append(sorted(current_line, key=lambda b: b[0]))
    return [img for line in lines for _, _, img in line]

# Preprocessing + character extraction
def extract_characters(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_data = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out noise
        if w < 5 or h < 5:
            continue

        char_crop = thresh[y:y+h, x:x+w]
        resized = cv2.resize(char_crop, (28, 28), interpolation=cv2.INTER_AREA)
        norm = resized.astype("float32") / 255.0
        norm = np.expand_dims(norm, axis=-1)
        char_data.append((x, y, norm))

    # Sort characters
    sorted_chars = sort_by_lines(char_data)
    return np.array(sorted_chars)

# Prediction
def predict_from_image(image_path):
    char_images = extract_characters(image_path)
    if len(char_images) == 0:
        return "No characters found"

    predictions = model.predict(char_images)
    predicted_labels = [class_names[np.argmax(p)] for p in predictions]
    return ''.join(predicted_labels)

# GUI App
class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR with CNN")
        self.root.geometry("400x600")

        self.label = tk.Label(root, text="Select an image with text", font=("Helvetica", 12))
        self.label.pack(pady=10)

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

        self.button = tk.Button(root, text="Choose Image", command=self.open_image)
        self.button.pack(pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk

        result = predict_from_image(file_path)
        self.result_label.configure(text=f"Predicted: {result}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()