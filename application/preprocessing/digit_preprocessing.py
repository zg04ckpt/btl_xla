import os
import cv2
import numpy as np
from .base_preprocessor import BasePreprocessor


class DigitPreprocessor(BasePreprocessor):
    
    def __init__(self, target_size=(28, 28), inner_size=20):
        """
            target_size: Kích thước output (mặc định 28x28 cho MNIST)
            inner_size: Kích thước vùng chứa chữ số trước khi pad (mặc định 20x20)
        """
        super().__init__(target_size, inner_size)
    
    def segment_and_preprocess(self, image_path, output_path=None, save_images=True, return_steps=False):
        """
        Segment và preprocess ảnh chứa nhiều chữ số
        """
        # 1. Load ảnh (grayscale)
        img = self.load_image(image_path)
        
        # Load ảnh gốc màu để hiển thị contours (nền trắng, dễ nhìn)
        img_color = cv2.imread(image_path)
        if img_color is None:
            # Nếu load màu thất bại, convert từ grayscale
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 2. Invert màu (chữ đen nền trắng -> chữ trắng nền đen)
        img_inv = self.invert_image(img)
        
        # 3. Gaussian Blur để giảm nhiễu trước khi threshold
        blurred = cv2.GaussianBlur(img_inv, (5, 5), 0)
        
        # 4. Binarize (Otsu threshold)
        binary = self.binarize(blurred, method='otsu')
        
        # 5. Morphological operations để làm sạch
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 6. Tìm contours
        contours = self.find_contours(morph)
        
        # Tạo thư mục output nếu cần
        if save_images and output_path:
            os.makedirs(output_path, exist_ok=True)
        
        digits = []
        digit_count = 0
        
        # Lưu preprocessing steps nếu cần
        preprocessing_steps = {}
        if return_steps:
            # Bước 1: Ảnh gốc màu
            preprocessing_steps['1_original'] = img_color
            
            # Bước 2: Grayscale
            preprocessing_steps['2_grayscale'] = img
            
            # Bước 3: Blurred (Gaussian blur sau khi invert)
            preprocessing_steps['3_blurred'] = blurred
            
            # Bước 4: Threshold (Binary)
            preprocessing_steps['4_threshold'] = binary
            
            # Bước 5: Morphology (sau closing + opening)
            preprocessing_steps['5_morphology'] = morph
        
        # 7. Xử lý từng contour
        filtered_contours = []
        cropped_images = []
        
        # Tạo ảnh để vẽ contours (trên ảnh màu gốc - nền trắng)
        img_with_boxes = img_color.copy()
        
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Kiểm tra và vẽ bounding box
            is_valid = self.filter_contour(cnt, min_w=10, min_h=10, min_area=100)
            
            if is_valid:
                # Vẽ khung xanh cho contour hợp lệ
                cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                filtered_contours.append(cnt)
                
                # Crop chữ số từ ảnh đã morphology
                digit = morph[y:y+h, x:x+w]
            cropped_images.append(digit)
            
            # Preprocess chữ số
            digit_processed = self.preprocess_single(digit)
            digits.append(digit_processed)
            
            # Lưu preprocessing steps cho từng digit
            if return_steps:
                preprocessing_steps[f'7_digit_{digit_count}'] = (digit_processed * 255).astype('uint8')
            
            # Lưu ảnh ra file (nếu cần)
            if save_images and output_path:
                save_path = os.path.join(output_path, f'digit_{digit_count}.png')
                # Chuyển về uint8 để lưu (nhân 255)
                digit_to_save = (digit_processed * 255).astype('uint8')
                cv2.imwrite(save_path, digit_to_save)
            
            digit_count += 1
        
        # Thêm bước 6: Bounding boxes trên ảnh màu gốc
        if return_steps:
            preprocessing_steps['6_contours'] = img_with_boxes
        
        if return_steps:
            return preprocessing_steps, digits
        return digits


# Ví dụ sử dụng (có thể comment lại khi không cần)
if __name__ == "__main__":
    # Khởi tạo preprocessor
    preprocessor = DigitPreprocessor(target_size=(28, 28), inner_size=20)
    
    # Xử lý ảnh
    digits = preprocessor.segment_and_preprocess(
        image_path=r'tieuluan\case_study_1\image.png',
        output_path=r'tieuluan\case_study_1\my_dataset',
        save_images=True
    )