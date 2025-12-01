"""
Các hàm tiện ích dùng chung cho preprocessing
"""
import cv2
import numpy as np
from scipy import ndimage


def get_best_shift(img):
    """
    Tính toán độ dịch chuyển để căn giữa ảnh theo center of mass
    """
    cy, cx = ndimage.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty


def shift_img(img, sx, sy):
    """
    Dịch chuyển ảnh theo vector (sx, sy)
    """
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def remove_border(img):
    """
    Loại bỏ viền đen thừa xung quanh ảnh
    """
    # Loại bỏ rows trống (toàn 0)
    while img.shape[0] > 0 and np.sum(img[0]) == 0:
        img = img[1:]
    
    while img.shape[0] > 0 and np.sum(img[-1]) == 0:
        img = img[:-1]
    
    # Loại bỏ cols trống (toàn 0)
    while img.shape[1] > 0 and np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)
    
    while img.shape[1] > 0 and np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)
    
    return img


def apply_threshold(img, method='otsu'):
    """
    Áp dụng threshold để chuyển ảnh về nhị phân
    """
    if method == 'otsu':
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:  # simple
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    return binary
