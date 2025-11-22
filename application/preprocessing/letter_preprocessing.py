"""Letter preprocessing module for handwritten letter recognition."""

import os
import cv2
import numpy as np
from scipy import ndimage
import math


def get_best_shift(img):
    """Calculate optimal shift to center the letter based on center of mass."""
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift_img(img, sx, sy):
    """Apply shift transformation to center the letter."""
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def segment_and_preprocess_letters(image_path, output_path=None):
    """
    Segment and preprocess letters from an image to 28x28 format.
    
    Args:
        image_path: Path to input image
        output_path: Optional directory to save individual letter images
        
    Returns:
        List of preprocessed 28x28 letter images (normalized 0-1)
    """
    # Đọc ảnh grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn.")

    # Invert: chữ đen nền trắng -> chữ trắng nền đen
    img_inv = 255 - img

    # Binary threshold (sử dụng Otsu để tự động)
    _, binary = cv2.threshold(img_inv, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Morphological operations cho letters: closing + opening
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Tìm contours (các vùng chữ cái)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp contours từ trái sang phải (dựa trên x)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    letters = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Bỏ qua contours nhỏ (noise)
        if w < 10 or h < 20:
            continue
        
        # Kiểm tra aspect ratio và area
        aspect_ratio = h / float(w) if w > 0 else 0
        area = cv2.contourArea(cnt)
        
        if aspect_ratio < 0.3 or aspect_ratio > 5.0 or area < 200:
            continue

        # Crop chữ cái với padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(binary.shape[1], x + w + padding)
        y2 = min(binary.shape[0], y + h + padding)
        
        letter = binary[y1:y2, x1:x2]

        # Make square để giữ tỷ lệ
        rows, cols = letter.shape
        size = max(rows, cols)
        square = np.zeros((size, size), dtype=np.uint8)
        
        y_offset = (size - rows) // 2
        x_offset = (size - cols) // 2
        square[y_offset:y_offset+rows, x_offset:x_offset+cols] = letter

        # Resize to 28x28
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

        # Làm nét chữ đậm hơn bằng dilation nhẹ
        kernel_dilate = np.ones((2, 2), np.uint8)
        resized = cv2.dilate(resized, kernel_dilate, iterations=1)

        # Center bằng center of mass
        shiftx, shifty = get_best_shift(resized)
        letter_shifted = shift_img(resized, shiftx, shifty)

        # Normalize về 0-1
        letter_norm = letter_shifted / 255.0

        letters.append(letter_norm)

        # Lưu ảnh nếu có output_path
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, f'letter_{i}.png')
            cv2.imwrite(save_path, (letter_norm * 255).astype(np.uint8))
            print(f"Đã lưu ảnh tại: {save_path}")

    return letters


def preprocess_single_letter(letter_image):
    """
    Preprocess a single letter image to 28x28 format.
    
    Args:
        letter_image: numpy array of letter image (already inverted - white on black)
        
    Returns:
        28x28 normalized letter image (0-1)
    """
    # Crop to content
    coords = cv2.findNonZero(letter_image)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    padding = 20
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(letter_image.shape[1], x + w + padding)
    y2 = min(letter_image.shape[0], y + h + padding)
    
    cropped = letter_image[y1:y2, x1:x2]
    
    # Make square
    rows, cols = cropped.shape
    size = max(rows, cols)
    square = np.zeros((size, size), dtype=np.uint8)
    
    y_offset = (size - rows) // 2
    x_offset = (size - cols) // 2
    square[y_offset:y_offset+rows, x_offset:x_offset+cols] = cropped
    
    # Resize to 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Dilation để làm nét
    kernel_dilate = np.ones((2, 2), np.uint8)
    resized = cv2.dilate(resized, kernel_dilate, iterations=1)
    
    # Center of mass alignment
    shiftx, shifty = get_best_shift(resized)
    aligned = shift_img(resized, shiftx, shifty)
    
    # Normalize
    return aligned / 255.0


# Example usage
if __name__ == "__main__":
    # Test với một ảnh mẫu
    pass
