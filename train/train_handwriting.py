import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATASET_DIR = "custom_handwriting_dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_SAVE_PATH = "custom_letters_model.h5"
PRETRAINED_MODEL = "emnist_letters_pretrained.h5"

LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]


def load_dataset(data_dir):
    images, labels = [], []

    for label_idx, letter in enumerate(LETTERS):
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.exists(letter_dir):
            continue

        for filename in os.listdir(letter_dir):
            if not filename.endswith(".png"):
                continue

            img = cv2.imread(os.path.join(letter_dir, filename), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (28, 28))
            if np.mean(img) > 127:
                img = 255 - img

            images.append(img)
            labels.append(label_idx)

    if not images:
        return None, None

    return np.array(images), np.array(labels)


def preprocess_data(X, y):
    X = X.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y = to_categorical(y, 26)
    return X, y


def create_model(use_pretrained=True):
    if use_pretrained and os.path.exists(PRETRAINED_MODEL):
        model = load_model(PRETRAINED_MODEL)
        for layer in model.layers[:-4]:
            layer.trainable = False
        return model

    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        # Fully Connected
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(26, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("=== TRAIN CUSTOM HANDWRITING MODEL ===")

    if not os.path.exists(TRAIN_DIR):
        print("Dataset not found. Please collect data first.")
        return

    # Load data
    X_train, y_train = load_dataset(TRAIN_DIR)
    if X_train is None:
        print("No training data found.")
        return

    X_test, y_test = load_dataset(TEST_DIR) if os.path.exists(TEST_DIR) else (None, None)

    # If no separate test set → split
    if X_test is None or len(X_test) < 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

    # Preprocess
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest"
    )
    datagen.fit(X_train)

    # Model
    use_transfer = os.path.exists(PRETRAINED_MODEL)
    model = create_model(use_pretrained=use_transfer)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True)
    ]

    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # Save final model
    final_path = MODEL_SAVE_PATH.replace(".h5", "_final.h5")
    model.save(final_path)

    # Training plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss")

    plot_path = "training_history.png"
    plt.tight_layout()
    plt.savefig(plot_path)

    print("Training completed.")
    print(f"Saved model: {MODEL_SAVE_PATH}")
    print(f"Saved final model: {final_path}")
    print(f"History plot: {plot_path}")


if __name__ == "__main__":
    main()