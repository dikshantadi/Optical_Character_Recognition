import tensorflow as tf
import numpy as np

train_dir = "resources/dataset/Dataset/data/training_data"
test_dir = "resources/dataset/Dataset/data/testing_data"

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    label_mode="int"
)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(28, 28),
    color_mode="grayscale",
    batch_size=32,
    label_mode="int"
)

# Normalize pixel values (0-255 to 0-1)
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(normalize_img)
test_ds = test_ds.map(normalize_img)

# Then define your CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(36, activation='softmax')  # 36 classes: 0-9 + A-Z
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=20, validation_data=test_ds)

model.save('cnn_ocr.h5')

