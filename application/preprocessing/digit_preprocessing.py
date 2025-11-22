import os
import cv2
import numpy as np
from scipy import ndimage
import math

def get_best_shift(img):
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty

def shift_img(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

def segment_and_preprocess_to_mnist(image_path, output_path):
    # Đọc ảnh grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn.")

    # Invert: chữ đen nền trắng -> chữ trắng nền đen
    img_inv = 255 - img

    # Binary threshold (sử dụng Otsu để tự động)
    _, binary = cv2.threshold(img_inv, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Tìm contours (các vùng chữ số)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp contours từ trái sang phải (dựa trên x)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    digits = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        # Bỏ qua contours nhỏ (noise)
        if w < 10 or h < 10 or w * h < 100:
            continue

        # Crop chữ số
        digit = binary[y:y+h, x:x+w]

        # Loại bỏ viền thừa (rows/cols hoàn toàn đen)
        while np.sum(digit[0]) == 0:
            digit = digit[1:]
        while np.sum(digit[-1]) == 0:
            digit = digit[:-1]
        while np.sum(digit[:, 0]) == 0:
            digit = np.delete(digit, 0, 1)
        while np.sum(digit[:, -1]) == 0:
            digit = np.delete(digit, -1, 1)

        rows, cols = digit.shape

        # Resize để fit vào 20x20 (giữ aspect ratio)
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            digit = cv2.resize(digit, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            digit = cv2.resize(digit, (cols, rows))

        # Pad về 28x28 (thêm padding đều)
        cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
        rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
        digit = np.pad(digit, (rows_padding, cols_padding), 'constant')

        # Center bằng center of mass
        shiftx, shifty = get_best_shift(digit)
        digit_shifted = shift_img(digit, shiftx, shifty)

        # Normalize về 0-1
        digit_norm = digit_shifted / 255.0

        digits.append(digit_norm)

        # Lưu ảnh (nhân lại 255 để lưu grayscale uint8)
        save_path = os.path.join(output_path, f'{i}.png')
        cv2.imwrite(save_path, (digit_norm * 255).astype(np.uint8))
        print(f"Đã lưu ảnh tại: {save_path}")

    return digits

digits = segment_and_preprocess_to_mnist(
    r'tieuluan\case_study_1\image.png',
    r'tieuluan\case_study_1\my_dataset'
)