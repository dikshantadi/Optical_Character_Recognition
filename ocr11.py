import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import pickle

# Loading the model and label encoder
model = tf.keras.models.load_model("ocr_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_character(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Invalid image"
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(1, 64, 64, 1)
    prediction = model.predict(img_input)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_label

def select_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = Image.open(file_path).resize((200, 200))

    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    result = predict_character(file_path)
    result_label.config(text=f"Predicted: {result}")

root = tk.Tk()
root.title("OCR Predictor")
root.geometry("500x500")

btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="Predicted: ", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()