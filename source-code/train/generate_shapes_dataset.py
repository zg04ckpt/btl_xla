import numpy as np
import cv2
import os
from tqdm import tqdm

# ==============================
# Configuration
# ==============================
IMG_SIZE = 64
NUM_PER_CLASS = 2000
DATASET_DIR = "shapes_dataset"
CLASSES = ["circle", "rectangle", "triangle"]

# Create dataset folders
for split in ["train", "test"]:
    for shape in CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, split, shape), exist_ok=True)


# ==============================
# Shape Generators
# ==============================
def generate_circle(size):
    img = np.zeros((size, size), dtype=np.uint8)
    center = np.random.randint(size // 4, 3 * size // 4, 2)
    radius = np.random.randint(size // 6, size // 3)
    thickness = np.random.choice([2, 3, 4, -1])
    cv2.circle(img, tuple(center), radius, 255, thickness)
    return img


def generate_rectangle(size):
    img = np.zeros((size, size), dtype=np.uint8)
    w = np.random.randint(size // 4, size // 2)
    h = np.random.randint(size // 4, size // 2)
    x = np.random.randint(5, size - w - 5)
    y = np.random.randint(5, size - h - 5)
    thickness = np.random.choice([2, 3, 4, -1])
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, thickness)
    return img


def generate_triangle(size):
    img = np.zeros((size, size), dtype=np.uint8)
    margin = size // 6
    pts = np.random.randint(margin, size - margin, (3, 2))
    pts = pts.reshape((-1, 1, 2))
    thickness = np.random.choice([2, 3, 4, -1])
    if thickness == -1:
        cv2.fillPoly(img, [pts], 255)
    else:
        cv2.polylines(img, [pts], True, 255, thickness)
    return img


# ==============================
# Noise / Augmentation
# ==============================
def add_noise(img):
    # Rotation
    if np.random.random() > 0.5:
        angle = np.random.randint(-30, 30)
        M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))

    # Noise
    if np.random.random() > 0.7:
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

    # Blur
    if np.random.random() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ==============================
# Dataset Generation
# ==============================
generators = {
    "circle": generate_circle,
    "rectangle": generate_rectangle,
    "triangle": generate_triangle
}

print("\nGenerating dataset...\n")

train_n = int(NUM_PER_CLASS * 0.8)
test_n = NUM_PER_CLASS - train_n

for shape in CLASSES:
    gen = generators[shape]

    # Train
    for i in tqdm(range(train_n), desc=f"Train {shape}"):
        img = add_noise(gen(IMG_SIZE))
        path = os.path.join(DATASET_DIR, "train", shape, f"{shape}_{i}.png")
        cv2.imwrite(path, img)

    # Test
    for i in tqdm(range(test_n), desc=f"Test {shape}"):
        img = add_noise(gen(IMG_SIZE))
        path = os.path.join(DATASET_DIR, "test", shape, f"{shape}_{i}.png")
        cv2.imwrite(path, img)

print("\nDataset generated successfully!")
print(f"Location: {DATASET_DIR}")
print(f"Train per class: {train_n}")
print(f"Test per class: {test_n}")
