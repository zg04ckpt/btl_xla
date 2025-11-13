"""Handwritten digit recognition from image using CNN model trained on MNIST"""

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import os

# Load model
try:
    model = load_model("mnist_cnn_model.h5")
    print("Model loaded successfully.\n")
except IOError:
    print("Error: Cannot find mnist_cnn_model.h5")
    exit()


def preprocess_digit(digit_image):
    """Convert digit image to MNIST format (28x28, black background, white digit)"""
    try:
        digit_np = np.array(digit_image)
        
        # Find bounding box and crop
        coords = cv2.findNonZero(digit_np)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit_np = digit_np[y:y+h, x:x+w]
        
        # Add padding
        digit_np = cv2.copyMakeBorder(digit_np, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        
        # Resize maintaining aspect ratio
        height, width = digit_np.shape
        if height == 0 or width == 0:
            return None
        
        if height > width:
            new_height = 20
            new_width = max(1, int(20 * width / height))
        else:
            new_width = 20
            new_height = max(1, int(20 * height / width))
        
        new_width = min(new_width, 20)
        new_height = min(new_height, 20)
        
        digit_resized = cv2.resize(digit_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create 28x28 canvas
        img_padded = np.zeros((28, 28), dtype=np.uint8)
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        img_padded[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = digit_resized
        
        # Normalize
        img_array = img_padded.astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def recognize_from_file():
    """Recognize digit sequence from image file"""
    # File dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    file_path = filedialog.askopenfilename(
        title="Select image with digit sequence",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    root.destroy()
    
    if not file_path:
        print("No file selected.")
        return
    
    # Read image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot read image.")
        return
    
    # Preprocessing
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find and filter contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_bboxes = []
    image_height, image_width = image.shape
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0
        area = cv2.contourArea(cnt)
        
        if (h > image_height * 0.25 and 
            0.8 < aspect_ratio < 3.5 and 
            area > 50 and
            5 < w < image_width * 0.5):
            digit_bboxes.append((x, y, w, h))
    
    digit_bboxes.sort(key=lambda b: b[0])
    print(f"Detected {len(digit_bboxes)} digits.\n")
    
    # Recognition
    recognized_string = ""
    pil_image = Image.fromarray(thresh)
    
    for index, (x, y, w, h) in enumerate(digit_bboxes):
        padding = 5
        digit_pil = pil_image.crop((
            max(0, x - padding), 
            max(0, y - padding), 
            min(thresh.shape[1], x + w + padding), 
            min(thresh.shape[0], y + h + padding)
        ))
        
        processed_digit = preprocess_digit(digit_pil)
        
        if processed_digit is not None:
            prediction = model.predict(processed_digit, verbose=0)
            digit = np.argmax(prediction[0])
            confidence = prediction[0][digit]
            
            print(f"  [{index}] -> {digit} ({confidence:.1%})")
            recognized_string += str(digit)
        else:
            recognized_string += "?" 
    
    # Result
    print(f"\n{'='*50}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"Result: {recognized_string}")
    print(f"Success: {len([c for c in recognized_string if c != '?'])}/{len(digit_bboxes)} digits")
    print(f"{'='*50}")


if __name__ == "__main__":
    recognize_from_file()
