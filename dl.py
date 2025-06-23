import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Dataset paths
train_dir = "resources/dataset/Dataset/data/training_data"
test_dir = "resources/dataset/Dataset/data/testing_data"

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(28, 28),
    color_mode="grayscale",
    batch_size=64,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(28, 28),
    color_mode="grayscale",
    batch_size=64,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(28, 28),
    color_mode="grayscale",
    batch_size=64,
    label_mode="int"
)

# Verify class names
print("Class names:", train_ds.class_names)
num_classes = len(train_ds.class_names)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05, fill_mode='constant'),
    layers.RandomZoom(0.05, fill_mode='constant'),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1, fill_mode='constant')
])

# Enhanced model architecture
def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        # Data augmentation
        data_augmentation,
        
        # Normalization
        layers.Rescaling(1./255),
        
        # Conv Block 1
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Build model
model = build_model()

# Learning rate schedule (fixed implementation)
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Compile model with the schedule
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

# Evaluation
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.2%}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save final model
model.save('ocr_model.h5')
print("Model saved as ocr_model.h5")