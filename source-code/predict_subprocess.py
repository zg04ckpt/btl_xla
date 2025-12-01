#!/usr/bin/env python
"""Standalone prediction script - avoids GUI/TensorFlow conflicts"""
import sys
import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

model_path = sys.argv[1]
data_file = sys.argv[2]

# Load digit images
with open(data_file, "rb") as f:
    digit_images = pickle.load(f)

# Load model and predict
import keras
import numpy as np
import cv2

model = keras.models.load_model(model_path, compile=False)

results = []
for digit_img in digit_images:
    if digit_img.shape != (28, 28):
        digit_img = cv2.resize(digit_img, (28, 28))
    
    # Preprocessor đã normalize rồi, không cần chia 255 nữa
    digit_img = digit_img.astype("float32")
    digit_img = digit_img.reshape(1, 28, 28, 1)
    
    prediction = model.predict(digit_img, verbose=0)
    digit = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][digit])
    
    results.append((digit, confidence))

# Save results
with open(data_file + ".result", "wb") as f:
    pickle.dump(results, f)
