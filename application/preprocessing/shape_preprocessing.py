import os
import cv2
import numpy as np
from .base_preprocessor import BasePreprocessor


class ShapePreprocessor(BasePreprocessor):
    
    def __init__(self, target_size=(64, 64), inner_size=56):
        """
            target_size: Kích thước output (mặc định 64x64 cho shapes)
            inner_size: Kích thước vùng chứa shape trước khi pad (mặc định 56x56)
        """
        super().__init__(target_size, inner_size)
    
    def filter_contour(self, cnt, min_w=20, min_h=20, min_area=400):
        """
        Override filter_contour với threshold phù hợp cho shapes
        """
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Kiểm tra kích thước cơ bản
        if w < min_w or h < min_h or w * h < min_area:
            return False
        
        return True
    
    def preprocess_shape_with_morph(self, img):
        """
        Áp dụng morphological operations để làm sạch shapes
        """
        # Morphological closing: đóng các lỗ nhỏ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Morphological opening: loại bỏ nhiễu nhỏ
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened
    
    def segment_and_preprocess(self, image_path, output_path=None, save_images=True, apply_morph=True, return_steps=False):
        """
        Segment và preprocess ảnh chứa nhiều shapes
        """
        # 1. Load ảnh
        img = self.load_image(image_path)
        
        # Load ảnh gốc màu để hiển thị contours
        img_color = cv2.imread(image_path)
        if img_color is None:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 2. Invert màu (shape đen nền trắng -> shape trắng nền đen)
        img_inv = self.invert_image(img)
        
        # 3. Gaussian Blur để giảm nhiễu
        blurred = cv2.GaussianBlur(img_inv, (5, 5), 0)
        
        # 4. Binarize
        binary = self.binarize(blurred, method='otsu')
        
        # 5. Apply morphological operations (nếu cần)
        if apply_morph:
            morph = self.preprocess_shape_with_morph(binary)
        else:
            morph = binary
        
        # 6. Tìm contours
        contours = self.find_contours(morph)
        
        # Tạo thư mục output nếu cần
        if save_images and output_path:
            os.makedirs(output_path, exist_ok=True)
        
        shapes = []
        shape_count = 0
        
        # Lưu preprocessing steps nếu cần
        preprocessing_steps = {}
        if return_steps:
            # Bước 1: Ảnh gốc màu
            preprocessing_steps['1_original'] = img_color
            
            # Bước 2: Grayscale
            preprocessing_steps['2_grayscale'] = img
            
            # Bước 3: Blurred
            preprocessing_steps['3_blurred'] = blurred
            
            # Bước 4: Threshold (Binary)
            preprocessing_steps['4_threshold'] = binary
            
            # Bước 5: Morphology
            preprocessing_steps['5_morphology'] = morph
        
        # 7. Xử lý từng contour
        filtered_contours = []
        cropped_images = []
        
        # Tạo ảnh để vẽ bounding boxes
        img_with_boxes = img_color.copy()
        
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Kiểm tra và vẽ bounding box
            is_valid = self.filter_contour(cnt)
            
            if is_valid:
                # Vẽ khung xanh cho contour hợp lệ
                cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                filtered_contours.append(cnt)
                
                # Crop shape từ ảnh đã morphology
                shape = morph[y:y+h, x:x+w]
            cropped_images.append(shape)
            
            # Preprocess shape
            shape_processed = self.preprocess_single(shape)
            shapes.append(shape_processed)
            
            # Lưu preprocessing steps cho từng shape
            if return_steps:
                preprocessing_steps[f'7_shape_{shape_count}'] = (shape_processed * 255).astype('uint8')
            
            # Lưu ảnh ra file (nếu cần)
            if save_images and output_path:
                save_path = os.path.join(output_path, f'shape_{shape_count}.png')
                # Chuyển về uint8 để lưu (nhân 255)
                shape_to_save = (shape_processed * 255).astype('uint8')
                cv2.imwrite(save_path, shape_to_save)
            
            shape_count += 1
        
        # Thêm bước 6: Bounding boxes trên ảnh màu gốc
        if return_steps:
            preprocessing_steps['6_contours'] = img_with_boxes
        
        if return_steps:
            return preprocessing_steps, shapes
        return shapes


# Ví dụ sử dụng (có thể comment lại khi không cần)
if __name__ == "__main__":
    # Khởi tạo preprocessor
    preprocessor = ShapePreprocessor(target_size=(64, 64), inner_size=56)
    
    # Xử lý ảnh
    shapes = preprocessor.segment_and_preprocess(
        image_path='path/to/shapes_image.png',
        output_path='output/shapes',
        save_images=True,
        apply_morph=True
    )
