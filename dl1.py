import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# loading the dataset form my computer (downloaded)
DATASET_PATH = "resources/dataset/Dataset/data/training_data"
all_images = []
all_labels = []

for label in sorted(os.listdir(DATASET_PATH)): 
    label_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (28, 28))
            all_images.append(img_resized)
            all_labels.append(label)

# Converting to an arrays
X = np.array(all_images)
X = X / 255.0  # Normalize
X = X.reshape(-1, 28, 28, 1)

y = np.array(all_labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

datagen = ImageDataGenerator(
    rotation_range=10,         
    zoom_range=0.1,            
    width_shift_range=0.1,     
    height_shift_range=0.1,    
    shear_range=0.1,          
    brightness_range=[0.8, 1.2] 

)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'), 
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),  
    Dropout(0.4), 
    Dense(len(np.unique(y)), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, y_categorical, epochs=50, validation_split=0.2)

TEST_PATH = "resources/dataset/Dataset/data/testing_data"
all_images = []
all_labels = []

for label in sorted(os.listdir(TEST_PATH)):
    label_path = os.path.join(TEST_PATH, label)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (28, 28))
            all_images.append(img_resized)
            all_labels.append(label)

X_test = np.array(all_images)
X_test = X_test / 255.0  # Normalize
X_test = X_test.reshape(-1, 28, 28, 1)

y_test = np.array(all_labels)

# Load the label encoder used during training
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print("Labels used in model:", list(label_encoder.classes_))

# Transform test labels using same mapping
y_encoded_test = label_encoder.transform(y_test)
y_categorical_test = to_categorical(y_encoded_test)

y_pred = model.predict(X_test)  # Probabilities
y_pred_classes = np.argmax(y_pred, axis=1)  # Class indices
y_true_classes = np.argmax(y_categorical_test, axis=1)

from sklearn.metrics import accuracy_score

test_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save("ocr_model.h5")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation='vertical', cmap='viridis')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.tight_layout()
plt.show()