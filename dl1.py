import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1. Loading the data (its in my computer)
# -----------------------------
DATASET_PATH = "resources/dataset/Dataset2/data/training_data"
all_images = []
all_labels = []

for label in sorted(os.listdir(DATASET_PATH)):
    label_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f" Could not read image: {img_path}")
                continue
            img_resized = cv2.resize(img, (64, 64))  # Use 64x64
            all_images.append(img_resized)
            all_labels.append(label)

X = np.array(all_images)
X = X / 255.0  
X = X.reshape(-1, 64, 64, 1)  

y = np.array(all_labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Saving the label encoder for testing phase
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# -----------------------------
# 2. Data augmentation
# -----------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    brightness_range=[0.6, 1.3],
    horizontal_flip=False,  
    fill_mode='nearest'
)

datagen.fit(X_train)

# -----------------------------
# 3. Building the CNN model
# -----------------------------
model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 4. Training the model
# -----------------------------
history = model.fit(
    X, y_categorical,
    epochs=30,

)

# -----------------------------
# 5. Loading the test data (again in my comp)
# -----------------------------
TEST_PATH = "resources/dataset/Dataset2/data/testing_data"
test_images = []
test_labels = []

for label in sorted(os.listdir(TEST_PATH)):
    label_path = os.path.join(TEST_PATH, label)
    if os.path.isdir(label_path):
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read test image: {img_path}")
                continue
            img_resized = cv2.resize(img, (64, 64))
            test_images.append(img_resized)
            test_labels.append(label)

X_test = np.array(test_images)
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 64, 64, 1)

y_test = np.array(test_labels)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print(" Labels used:", list(label_encoder.classes_))

# Encode test labels
y_encoded_test = label_encoder.transform(y_test)
y_categorical_test = to_categorical(y_encoded_test)

# -----------------------------
# 6. Evaluating the model
# -----------------------------
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_categorical_test, axis=1)

test_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


model.save("ocr_model.h5")

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation='vertical', cmap='viridis')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()