"""
Main Window - Cửa sổ chính ứng dụng nhận dạng chữ số viết tay
"""
from PyQt5.QtWidgets import (QMainWindow, QLabel, QPushButton, QFileDialog, 
                              QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, 
                              QFrame, QTextEdit, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt
import os

from application.preprocessing.image_processor import ImageProcessor
from application.recognition.digit_recognizer import DigitRecognizer
from application.recognition.shape_recognizer import ShapeRecognizer
from application.gui.preprocessing_viewer import PreprocessingViewer
from application.gui.result_dialog import ResultDialog

class MainWindow(QMainWindow):
    """Cửa sổ chính ứng dụng"""
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.recognition_mode = 'digits'  # Chế độ: 'digits' hoặc 'shapes'
        self.init_ui()
        
        # Hiển thị cửa sổ trước khi khởi tạo model
        self.show()
        self.repaint()
        
        self.result_text.setText("⏳ Đang khởi tạo...")
        
        # Khởi tạo các bộ xử lý
        self.image_processor = ImageProcessor()
        self.digit_recognizer = DigitRecognizer()
        self.shape_recognizer = ShapeRecognizer()
        
        self.result_text.setText("✓ Sẵn sàng! Kéo thả hoặc tải ảnh để bắt đầu.")
    
    def init_ui(self):
        """Khởi tạo giao diện người dùng"""
        self.setWindowTitle("Nhận Dạng Chữ Số và Hình Học")
        self.setGeometry(100, 100, 850, 650)  # Kích thước 850x650 (tăng 50px)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # === Mode selection (Digits / Shapes) ===
        mode_frame = QFrame()
        mode_frame.setStyleSheet("""
            QFrame {
                background-color: #fff3e0;
                border: 2px solid #ff9800;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        mode_layout = QHBoxLayout()
        
        mode_label = QLabel("Chế độ:")
        mode_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        # Radio buttons for mode selection
        self.digit_mode_radio = QRadioButton("Chữ số")
        self.digit_mode_radio.setChecked(True)
        self.digit_mode_radio.setStyleSheet("font-size: 15px;")
        
        self.shape_mode_radio = QRadioButton("Hình học")
        self.shape_mode_radio.setStyleSheet("font-size: 15px;")
        
        # Button group
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.digit_mode_radio)
        self.mode_button_group.addButton(self.shape_mode_radio)
        
        # Connect signals
        self.digit_mode_radio.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.digit_mode_radio)
        mode_layout.addWidget(self.shape_mode_radio)
        mode_layout.addStretch()
        
        mode_frame.setLayout(mode_layout)
        main_layout.addWidget(mode_frame)
        
        # === 1. Image upload area (drag-drop/paste/upload) ===
        upload_frame = QFrame()
        upload_frame.setFrameStyle(QFrame.Box | QFrame.Sunken)
        upload_frame.setLineWidth(2)
        upload_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 2px dashed #999;
                border-radius: 5px;
                min-height: 250px;
            }
        """)
        
        upload_layout = QVBoxLayout()
        
        # Image display label
        self.image_label = QLabel("Kéo thả ảnh vào đây hoặc nhấn nút tải ảnh")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("font-size: 16px; color: #666;")
        upload_layout.addWidget(self.image_label)
        
        # Upload button
        self.upload_button = QPushButton("📁 Tải ảnh")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_button.clicked.connect(self.upload_image)
        upload_layout.addWidget(self.upload_button)
        
        upload_frame.setLayout(upload_layout)
        main_layout.addWidget(upload_frame)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # === 2. Process button ===
        self.process_button = QPushButton("▶ Xử lý ảnh")
        self.process_button.setEnabled(False)
        self.process_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.process_button.clicked.connect(self.process_image)
        main_layout.addWidget(self.process_button)
        
        # === 3. Preprocessing steps viewer ===
        self.preprocessing_viewer = PreprocessingViewer()
        main_layout.addWidget(self.preprocessing_viewer, stretch=1)
        
        # === 4. Result display ===
        result_frame = QFrame()
        result_frame.setFrameStyle(QFrame.Box)
        result_frame.setStyleSheet("""
            QFrame {
                background-color: #e8f5e9;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        result_layout = QVBoxLayout()
        
        result_title = QLabel("📊 Kết quả nhận dạng:")
        result_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        result_layout.addWidget(result_title)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 8px;
            }
        """)
        self.result_text.setText("Chưa có kết quả")
        result_layout.addWidget(self.result_text)
        
        result_frame.setLayout(result_layout)
        main_layout.addWidget(result_frame)
        
        main_widget.setLayout(main_layout)
    
    def on_mode_changed(self):
        """Xử lý khi đổi chế độ (Chữ số/Hình học)"""
        if self.digit_mode_radio.isChecked():
            self.recognition_mode = 'digits'
        else:
            self.recognition_mode = 'shapes'
        
        # Xóa kết quả cũ khi đổi chế độ
        if self.image_path:
            self.result_text.setText(f"Chế độ: {'Chữ số' if self.recognition_mode == 'digits' else 'Hình học'}\n\nNhấn 'Xử lý' để nhận dạng.")
    
    def upload_image(self):
        """Mở hộp thoại chọn file ảnh"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Chọn ảnh", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", 
            options=options
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Tải và hiển thị ảnh đã chọn"""
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Lỗi", f"File không tồn tại: {file_path}")
            return
        
        self.image_path = file_path
        
        # Display image
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Lỗi", "Không thể tải ảnh!")
            return
        
        # Scale to fit label
        scaled_pixmap = pixmap.scaled(400, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
        # Enable process button
        self.process_button.setEnabled(True)
        
        # Clear previous results
        self.result_text.setText("Ảnh đã tải. Nhấn 'Xử lý' để nhận dạng.")
        self.preprocessing_viewer.clear_steps()
    
    def process_image(self):
        """Xử lý ảnh và nhận dạng chữ số hoặc hình học"""
        if not self.image_path:
            return
        
        try:
            # Hiển thị thông báo đang xử lý
            mode_text = "chữ số" if self.recognition_mode == 'digits' else "hình học"
            self.result_text.setText(f"Đang xử lý {mode_text}...")
            
            # Bước 1: Tiền xử lý ảnh theo chế độ
            preprocessing_steps, object_images = self.image_processor.process_image(self.image_path, self.recognition_mode)
            
            # Hiển thị các bước tiền xử lý
            self.preprocessing_viewer.display_preprocessing_steps(preprocessing_steps)
            
            # Bước 2: Nhận dạng theo chế độ
            if not object_images:
                self.result_text.setText(f"❌ Không phát hiện {mode_text} nào!")
                return
            
            if self.recognition_mode == 'digits':
                # Nhận dạng chữ số
                results = self.digit_recognizer.recognize_digits(object_images)
                
                # Tạo chuỗi kết quả
                digits_only = "".join([str(digit) for digit, _ in results])
                
                # Hiển thị trong text area
                result_text = f"✓ Phát hiện {len(object_images)} chữ số\n\n"
                result_text += f"Kết quả: {' '.join([str(d) for d, _ in results])}\n\n"
                result_text += f"Chuỗi số: {digits_only}\n"
                result_text += f"Độ tin cậy TB: {sum(c for _, c in results) / len(results) * 100:.1f}%"
                self.result_text.setText(result_text)
                
                # Hiển thị popup kết quả ở giữa màn hình (kích thước 567x433 = 2/3 giao diện chính)
                dialog_text = f"Số nhận dạng được:\n\n{digits_only}\n\n({len(object_images)} chữ số)"
                dialog = ResultDialog(dialog_text, self)
                dialog.exec_()
                
            else:  # Chế độ hình học
                # Nhận dạng hình học
                results = self.shape_recognizer.recognize_shapes(object_images)
                
                # Tên hình bằng tiếng Việt
                shape_names = {'circle': 'Hình tròn', 'rectangle': 'Hình chữ nhật', 'triangle': 'Tam giác'}
                
                # Đếm số lượng từng loại hình
                shape_counts = {}
                for shape, _ in results:
                    shape_counts[shape] = shape_counts.get(shape, 0) + 1
                
                # Hiển thị trong text area
                result_text = f"✓ Phát hiện {len(object_images)} hình\n\nKết quả:\n"
                for i, (shape, confidence) in enumerate(results):
                    vn_shape = shape_names.get(shape, shape)
                    result_text += f"  {i+1}. {vn_shape} ({confidence*100:.1f}%)\n"
                result_text += f"\nThống kê:\n"
                for shape, count in sorted(shape_counts.items()):
                    result_text += f"  {shape_names.get(shape, shape)}: {count}\n"
                result_text += f"\nĐộ tin cậy TB: {sum(c for _, c in results) / len(results) * 100:.1f}%"
                self.result_text.setText(result_text)
                
                # Hiển thị popup kết quả ở giữa màn hình (kích thước 567x433 = 2/3 giao diện chính)
                summary = "\n".join([f"{shape_names.get(s, s)}: {c}" for s, c in sorted(shape_counts.items())])
                dialog_text = f"Phát hiện {len(object_images)} hình:\n\n{summary}"
                dialog = ResultDialog(dialog_text, self)
                dialog.exec_()
            
        except Exception as e:
            error_msg = f"Lỗi khi xử lý ảnh:\n{str(e)}"
            self.result_text.setText(f"❌ {error_msg}")
            QMessageBox.critical(self, "Lỗi", error_msg)
            import traceback
            traceback.print_exc()
    
    # === Hỗ trợ Kéo & Thả ===
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Xử lý khi kéo file vào cửa sổ"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Xử lý khi thả file vào cửa sổ"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_image(file_path)