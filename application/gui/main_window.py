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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.recognition_mode = 'digits'  # 'digits' or 'shapes'
        self.init_ui()
        
        # Show window first, then init processors
        self.show()
        self.repaint()
        
        self.result_text.setText("⏳ Đang khởi tạo...")
        
        self.image_processor = ImageProcessor()
        self.digit_recognizer = DigitRecognizer()
        self.shape_recognizer = ShapeRecognizer()
        
        self.result_text.setText("✓ Sẵn sàng! Kéo thả hoặc tải ảnh để bắt đầu.")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Nhận Dạng Chữ Số và Hình Học")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
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
        mode_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        mode_layout.addWidget(mode_label)
        
        # Radio buttons for mode selection
        self.digit_mode_radio = QRadioButton("Chữ số")
        self.digit_mode_radio.setChecked(True)
        self.digit_mode_radio.setStyleSheet("font-size: 13px;")
        
        self.shape_mode_radio = QRadioButton("Hình học")
        self.shape_mode_radio.setStyleSheet("font-size: 13px;")
        
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
                min-height: 200px;
            }
        """)
        
        upload_layout = QVBoxLayout()
        
        # Image display label
        self.image_label = QLabel("Kéo thả ảnh vào đây hoặc nhấn nút tải ảnh")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(150)
        self.image_label.setStyleSheet("font-size: 14px; color: #666;")
        upload_layout.addWidget(self.image_label)
        
        # Upload button
        self.upload_button = QPushButton("📁 Tải ảnh")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
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
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
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
        result_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        result_layout.addWidget(result_title)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(100)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        self.result_text.setText("Chưa có kết quả")
        result_layout.addWidget(self.result_text)
        
        result_frame.setLayout(result_layout)
        main_layout.addWidget(result_frame)
        
        main_widget.setLayout(main_layout)
    
    def on_mode_changed(self):
        """Handle mode change (Digits/Shapes)"""
        if self.digit_mode_radio.isChecked():
            self.recognition_mode = 'digits'
        else:
            self.recognition_mode = 'shapes'
        
        # Clear previous results when mode changes
        if self.image_path:
            self.result_text.setText(f"Chế độ: {'Chữ số' if self.recognition_mode == 'digits' else 'Hình học'}\n\nNhấn 'Xử lý' để nhận dạng.")
    
    def upload_image(self):
        """Open file dialog to select image"""
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
        """Load and display the selected image"""
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
        """Process the image and recognize digits or shapes"""
        if not self.image_path:
            return
        
        try:
            # Show processing message
            mode_text = "chữ số" if self.recognition_mode == 'digits' else "hình học"
            self.result_text.setText(f"Đang xử lý {mode_text}...")
            QMessageBox.information(self, "Thông báo", f"Bắt đầu xử lý ảnh ({mode_text})...")
            
            # Step 1: Preprocess image with mode
            preprocessing_steps, object_images = self.image_processor.process_image(self.image_path, self.recognition_mode)
            
            # Display preprocessing steps
            self.preprocessing_viewer.display_preprocessing_steps(preprocessing_steps)
            
            # Step 2: Recognize based on mode
            if not object_images:
                self.result_text.setText(f"❌ Không phát hiện {mode_text} nào!")
                return
            
            if self.recognition_mode == 'digits':
                results = self.digit_recognizer.recognize_digits(object_images)
                
                # Display digit results
                result_text = f"✓ Phát hiện {len(object_images)} chữ số\n\n"
                result_text += "Kết quả: "
                
                digits_only = ""
                for i, (digit, confidence) in enumerate(results):
                    digits_only += str(digit)
                    result_text += f"{digit} "
                
                result_text += f"\n\nChuỗi số: {digits_only}\n"
                result_text += f"Độ tin cậy trung bình: {sum(c for _, c in results) / len(results) * 100:.1f}%"
                
                self.result_text.setText(result_text)
                QMessageBox.information(self, "Hoàn thành", f"Kết quả: {digits_only}")
                
            else:  # shapes mode
                results = self.shape_recognizer.recognize_shapes(object_images)
                
                # Display shape results
                shape_names = {'circle': 'Hình tròn', 'rectangle': 'Hình chữ nhật', 'triangle': 'Tam giác'}
                result_text = f"✓ Phát hiện {len(object_images)} hình\n\n"
                result_text += "Kết quả:\n"
                
                shape_counts = {}
                for i, (shape, confidence) in enumerate(results):
                    vn_shape = shape_names.get(shape, shape)
                    result_text += f"  {i+1}. {vn_shape} ({confidence*100:.1f}%)\n"
                    shape_counts[shape] = shape_counts.get(shape, 0) + 1
                
                result_text += f"\nThống kê:\n"
                for shape, count in sorted(shape_counts.items()):
                    vn_shape = shape_names.get(shape, shape)
                    result_text += f"  {vn_shape}: {count}\n"
                
                result_text += f"\nĐộ tin cậy trung bình: {sum(c for _, c in results) / len(results) * 100:.1f}%"
                
                self.result_text.setText(result_text)
                
                summary = ", ".join([f"{count} {shape}" for shape, count in sorted(shape_counts.items())])
                QMessageBox.information(self, "Hoàn thành", f"Kết quả: {summary}")
            
        except Exception as e:
            error_msg = f"Lỗi khi xử lý ảnh:\n{str(e)}"
            self.result_text.setText(f"❌ {error_msg}")
            QMessageBox.critical(self, "Lỗi", error_msg)
            import traceback
            traceback.print_exc()
    
    # === Drag & Drop support ===
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_image(file_path)