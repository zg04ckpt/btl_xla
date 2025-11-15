"""
Preprocessing viewer widget - displays each preprocessing step as a horizontal row of images.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class PreprocessingViewer(QWidget):
    """Widget to display preprocessing steps in rows"""
    
    def __init__(self):
        super().__init__()
        self.step_labels = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI layout"""
        # Main vertical layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(10)
        
        # Scroll area for preprocessing steps
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container widget for steps
        self.steps_container = QWidget()
        self.steps_layout = QVBoxLayout()
        self.steps_layout.setSpacing(5)
        self.steps_container.setLayout(self.steps_layout)
        
        scroll.setWidget(self.steps_container)
        self.main_layout.addWidget(scroll)
        
        self.setLayout(self.main_layout)
    
    def display_preprocessing_steps(self, preprocessing_steps):
        """
        Display all preprocessing steps.
        preprocessing_steps: dict with step names as keys and images as values
        """
        # Clear previous steps
        self.clear_steps()
        
        # Group steps by type
        main_steps = {}
        digit_steps = {}
        
        for key, img in preprocessing_steps.items():
            if key.startswith('7_digit_'):
                digit_steps[key] = img
            else:
                main_steps[key] = img
        
        # Display main preprocessing steps (one image per row)
        for step_name in sorted(main_steps.keys()):
            img = main_steps[step_name]
            self.add_step_row(step_name, [img])
        
        # Display all digits in one row
        if digit_steps:
            digit_images = [digit_steps[k] for k in sorted(digit_steps.keys())]
            self.add_step_row("Segmented Digits (28x28)", digit_images)
    
    def add_step_row(self, step_name, images):
        """Add a row showing one preprocessing step with multiple images"""
        # Row container
        row_widget = QWidget()
        row_layout = QVBoxLayout()
        row_layout.setContentsMargins(5, 5, 5, 5)
        
        # Step title
        title_label = QLabel(self.format_step_name(step_name))
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        row_layout.addWidget(title_label)
        
        # Horizontal layout for images
        images_layout = QHBoxLayout()
        images_layout.setSpacing(5)
        
        for img in images:
            img_label = QLabel()
            pixmap = self.numpy_to_pixmap(img)
            
            # Scale image for display (max height 150px)
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
        """Format step name for display"""
        name_map = {
            '1_original': '1. Original Image',
            '2_grayscale': '2. Grayscale Conversion',
            '3_blurred': '3. Gaussian Blur (Noise Reduction)',
            '4_threshold': '4. Adaptive Thresholding',
            '5_morphology': '5. Morphological Operations',
            '6_contours': '6. Contour Detection & Filtering',
        }
        
        if step_name in name_map:
            return name_map[step_name]
        elif step_name.startswith('7_digit_'):
            return step_name
        elif 'Segmented Digits' in step_name:
            return '7. ' + step_name
        else:
            return step_name.replace('_', ' ').title()
    
    def numpy_to_pixmap(self, img):
        """Convert numpy array to QPixmap"""
        if len(img.shape) == 2:  # Grayscale
            h, w = img.shape
            bytes_per_line = w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color (BGR)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        return QPixmap.fromImage(q_img)
    
    def clear_steps(self):
        """Clear all displayed steps"""
        while self.steps_layout.count():
            child = self.steps_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()