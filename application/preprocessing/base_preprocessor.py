"""
Base Preprocessor - Class cha chứa logic preprocessing chung
"""
import cv2
import numpy as np
import math
from .utils import get_best_shift, shift_img, remove_border, apply_threshold


class BasePreprocessor:
    def __init__(self, target_size=(28, 28), inner_size=20):
        """
            target_size: Tuple (height, width) kích thước đầu ra
            inner_size: Kích thước vùng chứa ký tự (int hoặc tuple)
        """
        self.target_size = target_size
        self.inner_size = inner_size if isinstance(inner_size, tuple) else (inner_size, inner_size)
    
    def load_image(self, image_path):

        # Đọc ảnh grayscale từ đường dẫn
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return img
    
    def invert_image(self, img):

        # Đảo màu ảnh: đen->trắng, trắng->đen

        return 255 - img
    
    def binarize(self, img, method='otsu'):
        # Chuyển ảnh về nhị phân (đen/trắng)

        return apply_threshold(img, method)
    
    def find_contours(self, binary):
        # Tìm contours (các vùng ký tự) và sắp xếp từ trái sang phải
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sắp xếp theo tọa độ x (trái -> phải)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        return contours
    
    def filter_contour(self, cnt, min_w=10, min_h=10, min_area=100):
        """
        Kiểm tra xem contour có hợp lệ không (lọc bỏ noise)
        """
        x, y, w, h = cv2.boundingRect(cnt)
        return w >= min_w and h >= min_h and w * h >= min_area
    
    def resize_keep_aspect(self, img):
        """
        Resize ảnh giữ nguyên aspect ratio để fit vào inner_size
        """
        rows, cols = img.shape
        inner_h, inner_w = self.inner_size
        
        # Tính toán kích thước mới dựa trên cạnh lớn hơn
        if rows > cols:
            factor = inner_h / rows
            new_rows = inner_h
            new_cols = int(round(cols * factor))
            # Đảm bảo không vượt quá inner_w
            if new_cols > inner_w:
                new_cols = inner_w
                new_rows = int(round(rows * inner_w / cols))
        else:
            factor = inner_w / cols
            new_cols = inner_w
            new_rows = int(round(rows * factor))
            # Đảm bảo không vượt quá inner_h
            if new_rows > inner_h:
                new_rows = inner_h
                new_cols = int(round(cols * inner_h / rows))
        
        return cv2.resize(img, (new_cols, new_rows))
    
    def pad_to_target(self, img):
        # Pad ảnh về target_size với viền đen (giá trị 0)

        rows, cols = img.shape
        target_h, target_w = self.target_size
        
        # Tính padding cho rows (trên/dưới)
        rows_padding = (
            int(math.ceil((target_h - rows) / 2.0)),
            int(math.floor((target_h - rows) / 2.0))
        )
        
        # Tính padding cho cols (trái/phải)
        cols_padding = (
            int(math.ceil((target_w - cols) / 2.0)),
            int(math.floor((target_w - cols) / 2.0))
        )
        
        return np.pad(img, (rows_padding, cols_padding), 'constant', constant_values=0)
    
    def center_image(self, img):
        # Căn giữa ảnh theo center of mass (trọng tâm
        shiftx, shifty = get_best_shift(img)
        return shift_img(img, shiftx, shifty)
    
    def normalize(self, img):
        # Normalize giá trị pixel về khoảng [0, 1]

        return img.astype('float32') / 255.0
    
    def preprocess_single(self, character_img):
        # Pipeline preprocessing cho 1 ký tự

        # 1. Loại bỏ viền thừa
        character_img = remove_border(character_img)
        
        # 2. Resize giữ aspect ratio
        character_img = self.resize_keep_aspect(character_img)
        
        # 3. Pad về target_size
        character_img = self.pad_to_target(character_img)
        
        # 4. Căn giữa theo center of mass
        character_img = self.center_image(character_img)
        
        # 5. Normalize về [0, 1]
        character_img = self.normalize(character_img)
        
        return character_img
    
    def segment_and_preprocess(self, image_path, output_path=None):

        # Segment và preprocess ảnh chứa nhiều ký tự

        raise NotImplementedError("Phương thức này cần được implement bởi class con")
