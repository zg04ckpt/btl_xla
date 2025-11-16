"""
Preprocessing Viewer - Hiển thị các bước tiền xử lý ảnh
Mỗi bước hiển thị trên một dòng với tiêu đề và kết quả
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np


class PreprocessingViewer(QWidget):
    """Widget hiển thị các bước tiền xử lý theo dòng"""
    
    def __init__(self):
        super().__init__()
        self.step_labels = []
        self.init_ui()
    
    def init_ui(self):
        """Khởi tạo giao diện"""
        # Layout dọc chính
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(10)
        
        # Vùng cuộn cho các bước xử lý
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container chứa các bước
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout()
        self.steps_layout.setSpacing(5)
        self.steps_container.setLayout(self.steps_layout)
        
        scroll.setWidget(self.steps_container)
        self.main_layout.addWidget(scroll)
        
        self.setLayout(self.main_layout)
    
    def display_preprocessing_steps(self, preprocessing_steps):
        """
        Hiển thị tất cả các bước tiền xử lý
        preprocessing_steps: dict với tên bước là key và ảnh là value
        """
        # Xóa các bước trước đó
        self.clear_steps()
        
        # Phân loại các bước
        main_steps = {}  # Các bước xử lý chính
        digit_steps = {}  # Các chữ số đã cắt
        
        for key, img in preprocessing_steps.items():
            if key.startswith('7_digit_'):
                digit_steps[key] = img
            else:
                main_steps[key] = img
        
        # Hiển thị các bước xử lý chính (1 ảnh/dòng)
        for step_name in sorted(main_steps.keys()):
            img = main_steps[step_name]
            self.add_step_row(step_name, [img])
        
        # Hiển thị tất cả chữ số trên 1 dòng
        if digit_steps:
            digit_images = [digit_steps[k] for k in sorted(digit_steps.keys())]
            self.add_step_row("Segmented Digits (28x28)", digit_images)
    
    def add_step_row(self, step_name, images):
        """Thêm một dòng hiển thị bước xử lý với nhiều ảnh"""
        # Container cho dòng
        row_widget = QWidget()
        row_layout = QVBoxLayout()
        row_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tiêu đề bước
        title_label = QLabel(self.format_step_name(step_name))
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        row_layout.addWidget(title_label)
        
        # Layout ngang cho các ảnh
        images_layout = QHBoxLayout()
        images_layout.setSpacing(5)
        
        for img in images:
            img_label = QLabel()
            pixmap = self.numpy_to_pixmap(img)
            
            # Resize ảnh cho phù hợp (chiều cao tối đa 150px)
            if pixmap.height() > 150:
                pixmap = pixmap.scaledToHeight(150, Qt.SmoothTransformation)
            
            img_label.setPixmap(pixmap)
            img_label.setStyleSheet("border: 1px solid #ccc; padding: 2px;")
            images_layout.addWidget(img_label)
        
        images_layout.addStretch()
        row_layout.addLayout(images_layout)
        
        row_widget.setLayout(row_layout)
        row_widget.setStyleSheet("background-color: #f5f5f5; border-radius: 5px;")
        
        self.steps_layout.addWidget(row_widget)
    
    def format_step_name(self, step_name):
        """Định dạng tên bước cho hiển thị"""
        name_map = {
            '1_original': '1. Ảnh gốc / Original Image',
            '2_grayscale': '2. Chuyển xám / Grayscale',
            '3_blurred': '3. Làm mờ / Gaussian Blur',
            '4_threshold': '4. Nhị phân hóa / Thresholding',
            '5_morphology': '5. Xử lý hình thái / Morphology',
            '6_contours': '6. Phát hiện viền / Contours',
        }
        
        if step_name in name_map:
            return name_map[step_name]
        elif step_name.startswith('7_digit_'):
            return step_name
        elif 'Segmented Digits' in step_name:
            return '7. Các chữ số / ' + step_name
        else:
            return step_name.replace('_', ' ').title()
    
    def numpy_to_pixmap(self, img):
        """Chuyển numpy array sang QPixmap"""
        if len(img.shape) == 2:  # Ảnh xám
            h, w = img.shape
            bytes_per_line = w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Ảnh màu (BGR)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            # Chuyển BGR sang RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        return QPixmap.fromImage(q_img)
    
    def clear_steps(self):
        """Xóa tất cả các bước đã hiển thị"""
        while self.steps_layout.count():
            child = self.steps_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()