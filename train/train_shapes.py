"""Train CNN model for geometric shape recognition"""

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
DATASET_DIR = "shapes_dataset"
CLASSES = ["circle", "rectangle", "triangle"]
NUM_CLASSES = len(CLASSES)


# -------------------------
# Load dataset
# -------------------------
def load_dataset(dataset_dir, img_size):
    X_train, y_train, X_test, y_test = [], [], [], []

    for idx, shape in enumerate(CLASSES):
        # Train
        for f in os.listdir(os.path.join(dataset_dir, "train", shape)):
            if f.endswith(".png"):
                img = cv2.imread(os.path.join(dataset_dir, "train", shape, f), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                X_train.append(img)
                y_train.append(idx)

        # Test
        for f in os.listdir(os.path.join(dataset_dir, "test", shape)):
            if f.endswith(".png"):
                img = cv2.imread(os.path.join(dataset_dir, "test", shape, f), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                X_test.append(img)
                y_test.append(idx)

    X_train = np.array(X_train).astype("float32") / 255.0
    X_test = np.array(X_test).astype("float32") / 255.0

    X_train = X_train.reshape(-1, img_size, img_size, 1)
    X_test = X_test.reshape(-1, img_size, img_size, 1)

    y_train = to_categorical(np.array(y_train), NUM_CLASSES)
    y_test = to_categorical(np.array(y_test), NUM_CLASSES)

    print(f"Loaded dataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    return X_train, y_train, X_test, y_test


if not os.path.exists(DATASET_DIR):
    print("Dataset not found! Please generate it first.")
    exit()

X_train, y_train, X_test, y_test = load_dataset(DATASET_DIR, IMG_SIZE)


# -------------------------
# Data Augmentation
# -------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest"
)
datagen.fit(X_train)


# -------------------------
# Build CNN
# -------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print(model.summary())


# -------------------------
# Callbacks
# -------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ModelCheckpoint("shapes_cnn_model_best.h5", monitor="val_accuracy", save_best_only=True)
]


# -------------------------
# Training
# -------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)


# -------------------------
# Evaluation
# -------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%")

model.save("shapes_cnn_model.h5")
print("Saved: shapes_cnn_model.h5 (final), shapes_cnn_model_best.h5 (best)")


# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("shapes_training_history.png")
print("Training plot saved.")
