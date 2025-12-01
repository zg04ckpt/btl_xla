"""
Main Window - Cửa sổ chính ứng dụng nhận dạng chữ số viết tay
"""
from PyQt5.QtWidgets import (QMainWindow, QLabel, QPushButton, QFileDialog, 
                              QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, 
                              QFrame, QTextEdit, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QKeyEvent
from PyQt5.QtCore import Qt
import os

from application.preprocessing import GeneralPreprocessor
from application.recognition.digit_recognizer import DigitRecognizer
from application.recognition.shape_recognizer import ShapeRecognizer
from application.recognition.letter_recognizer import LetterRecognizer
from application.gui.preprocessing_viewer import PreprocessingViewer
from application.gui.result_dialog import ResultDialog
from application.gui.drawing_canvas import DrawingCanvas
from application.gui.processing_worker import ProcessingWorker
import tempfile

class MainWindow(QMainWindow):
    """Cửa sổ chính ứng dụng"""
    
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.temp_canvas_file = None
        self.recognition_mode = 'digits'  # Chế độ: 'digits', 'shapes', hoặc 'letters'
        self.worker = None  # Worker thread cho xử lý ảnh
        self.init_ui()
        
        # Hiển thị cửa sổ trước khi khởi tạo model
        self.show()
        self.repaint()
        
        self.result_text.setText("⏳ Đang khởi tạo...")
        
        # Khởi tạo các bộ xử lý riêng biệt cho từng mode

        self.digit_preprocessor = GeneralPreprocessor(target_size=(28, 28), inner_size=20, min_h=10, min_w=10, min_area=100)
        self.letter_preprocessor = GeneralPreprocessor(target_size=(28, 28), inner_size=20, min_h=20, min_w=10, min_area=200)
        self.shape_preprocessor = GeneralPreprocessor(target_size=(64, 64), inner_size=50, min_h=20, min_w=20, min_area=400)
        
        self.digit_recognizer = DigitRecognizer()
        self.shape_recognizer = ShapeRecognizer()
        self.letter_recognizer = LetterRecognizer()
        
        self.result_text.setText("✓ Sẵn sàng! Kéo thả hoặc tải ảnh để bắt đầu.")
    
    def init_ui(self):
        """Khởi tạo giao diện người dùng"""
        self.setWindowTitle("Nhận Dạng Chữ Số, Chữ Cái và Hình Học - Nhấn F11 để thoát fullscreen")
        self.showFullScreen()  # Full screen (F11 hoặc ESC để thoát)
        
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
        
        self.letter_mode_radio = QRadioButton("Chữ cái")
        self.letter_mode_radio.setStyleSheet("font-size: 15px;")
        
        self.shape_mode_radio = QRadioButton("Hình học")
        self.shape_mode_radio.setStyleSheet("font-size: 15px;")
        
        # Button group
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.digit_mode_radio)
        self.mode_button_group.addButton(self.letter_mode_radio)
        self.mode_button_group.addButton(self.shape_mode_radio)
        
        # Connect signals
        self.digit_mode_radio.toggled.connect(self.on_mode_changed)
        self.letter_mode_radio.toggled.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.digit_mode_radio)
        mode_layout.addWidget(self.letter_mode_radio)
        mode_layout.addWidget(self.shape_mode_radio)
        mode_layout.addStretch()
        
        mode_frame.setLayout(mode_layout)
        main_layout.addWidget(mode_frame)
        
        # Layout chính cho phần nội dung chia trái/phải
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # === Khu vực bên trái: tải ảnh + kết quả ===
        left_panel = QFrame()
        left_panel.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        
        # === 1. Drawing Canvas ===
        canvas_frame = QFrame()
        canvas_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 2px solid #999;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        canvas_layout = QVBoxLayout()
        
        canvas_title = QLabel("✏️ Khu vực vẽ/Tải ảnh")
        canvas_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        canvas_layout.addWidget(canvas_title)
        
        # Drawing canvas (responsive size)
        self.canvas = DrawingCanvas(width=600, height=400)
        canvas_layout.addWidget(self.canvas, stretch=1)
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        
        # Clear button
        self.clear_button = QPushButton("🗑️ Xóa")
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.clear_button.clicked.connect(self.clear_canvas)
        buttons_layout.addWidget(self.clear_button)
        
        # Upload button
        self.upload_button = QPushButton("📁 Tải ảnh")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_button.clicked.connect(self.upload_image)
        buttons_layout.addWidget(self.upload_button)
        
        buttons_layout.addStretch()
        canvas_layout.addLayout(buttons_layout)
        
        canvas_frame.setLayout(canvas_layout)
        left_layout.addWidget(canvas_frame)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # === 2. Process button ===
        self.process_button = QPushButton("▶ Xử lý ảnh")
        self.process_button.setEnabled(True)  # Luôn bật cho canvas vẽ
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
        left_layout.addWidget(self.process_button)
        
        # === 3. Result display ===
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
        left_layout.addWidget(result_frame)
        left_layout.addStretch(1)
        left_panel.setLayout(left_layout)
        content_layout.addWidget(left_panel, stretch=1)
        
        # === Khu vực bên phải: các bước xử lý ===
        right_panel = QFrame()
        right_panel.setStyleSheet("""
            QFrame {
                background-color: #f7f9fc;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        steps_label = QLabel("🔍 Các bước xử lý ảnh")
        steps_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(steps_label)
        
        self.preprocessing_viewer = PreprocessingViewer()
        right_layout.addWidget(self.preprocessing_viewer, stretch=1)
        
        right_panel.setLayout(right_layout)
        content_layout.addWidget(right_panel, stretch=1)
        
        main_layout.addLayout(content_layout, stretch=1)
        
        main_widget.setLayout(main_layout)
    
    def on_mode_changed(self):
        """Xử lý khi đổi chế độ (Chữ số/Chữ cái/Hình học)"""
        if self.digit_mode_radio.isChecked():
            self.recognition_mode = 'digits'
        elif self.letter_mode_radio.isChecked():
            self.recognition_mode = 'letters'
        else:
            self.recognition_mode = 'shapes'
        
        # Xóa canvas và kết quả cũ khi đổi chế độ
        self.canvas.clear_canvas()
        self.image_path = None
        self.preprocessing_viewer.clear_steps()
        
        mode_names = {'digits': 'Chữ số', 'letters': 'Chữ cái', 'shapes': 'Hình học'}
        self.result_text.setText(f"Chế độ: {mode_names[self.recognition_mode]}\n\nVẽ hoặc tải ảnh để nhận dạng.")
    
    def clear_canvas(self):
        """Xóa canvas vẽ"""
        self.canvas.clear_canvas()
        self.image_path = None
        self.result_text.setText("Canvas đã xóa. Vẽ hoặc tải ảnh để nhận dạng.")
        self.preprocessing_viewer.clear_steps()
    
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
        
        # Load image to canvas
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Lỗi", "Không thể tải ảnh!")
            return
        
        # Scale and draw on canvas
        scaled_pixmap = pixmap.scaled(self.canvas.width(), self.canvas.height(), 
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.canvas.image = scaled_pixmap.toImage()
        self.canvas.update()
        
        # Enable process button
        self.process_button.setEnabled(True)
        
        # Clear previous results
        self.result_text.setText("Ảnh đã tải. Nhấn 'Xử lý' để nhận dạng.")
        self.preprocessing_viewer.clear_steps()
    
    def process_image(self):
        """Xử lý ảnh và nhận dạng chữ số hoặc hình học (async với QThread)"""
        # Kiểm tra nếu đang xử lý thì không cho xử lý tiếp
        if self.worker and self.worker.isRunning():
            return
        
        try:
            # Lưu canvas thành file tạm
            if self.temp_canvas_file:
                try:
                    os.unlink(self.temp_canvas_file)
                except:
                    pass
            
            self.temp_canvas_file = tempfile.mktemp(suffix='.png')
            self.canvas.save_to_file(self.temp_canvas_file)
            self.image_path = self.temp_canvas_file
            
            # Hiển thị thông báo đang xử lý và disable buttons
            mode_texts = {'digits': 'chữ số', 'letters': 'chữ cái', 'shapes': 'hình học'}
            mode_text = mode_texts.get(self.recognition_mode, 'chữ số')
            self.result_text.setText(f"⏳ Đang xử lý {mode_text}...\n\nVui lòng chờ...")
            
            # Disable buttons
            self._set_buttons_enabled(False)
            
            # Tạo worker thread để xử lý
            preprocessors = {
                'digit': self.digit_preprocessor,
                'letter': self.letter_preprocessor,
                'shape': self.shape_preprocessor
            }
            recognizers = {
                'digit': self.digit_recognizer,
                'letter': self.letter_recognizer,
                'shape': self.shape_recognizer
            }
            
            self.worker = ProcessingWorker(
                self.image_path,
                self.recognition_mode,
                preprocessors,
                recognizers
            )
            
            # Kết nối signals
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.error.connect(self.on_processing_error)
            self.worker.progress.connect(self.on_processing_progress)
            
            # Bắt đầu xử lý
            self.worker.start()
            
        except Exception as e:
            error_msg = f"Lỗi khi khởi tạo xử lý:\n{str(e)}"
            self.result_text.setText(f"❌ {error_msg}")
            QMessageBox.critical(self, "Lỗi", error_msg)
            self._set_buttons_enabled(True)
    
    def _set_buttons_enabled(self, enabled: bool):
        """Bật/tắt tất cả các buttons"""
        self.process_button.setEnabled(enabled)
        self.upload_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)
        self.digit_mode_radio.setEnabled(enabled)
        self.letter_mode_radio.setEnabled(enabled)
        self.shape_mode_radio.setEnabled(enabled)
    
    def on_processing_progress(self, message: str):
        """Cập nhật tiến trình xử lý"""
        mode_texts = {'digits': 'chữ số', 'letters': 'chữ cái', 'shapes': 'hình học'}
        mode_text = mode_texts.get(self.recognition_mode, 'đối tượng')
        self.result_text.setText(f"⏳ {message}\n\nVui lòng chờ...")
    
    def on_processing_error(self, error_msg: str):
        """Xử lý lỗi từ worker"""
        self.result_text.setText(f"❌ {error_msg}")
        QMessageBox.critical(self, "Lỗi", error_msg)
        self._set_buttons_enabled(True)
    
    def on_processing_finished(self, result: dict):
        """Xử lý kết quả từ worker"""
        try:
            preprocessing_steps = result['preprocessing_steps']
            object_images = result['object_images']
            results = result['results']
            mode = result['mode']
            
            # Hiển thị các bước tiền xử lý
            self.preprocessing_viewer.display_preprocessing_steps(preprocessing_steps)
            
            # Hiển thị kết quả theo chế độ
            if mode == 'digits':
                self._show_digit_results(object_images, results)
            elif mode == 'letters':
                self._show_letter_results(object_images, results)
            else:  # shapes
                self._show_shape_results(object_images, results)
                
        except Exception as e:
            error_msg = f"Lỗi khi hiển thị kết quả:\n{str(e)}"
            self.result_text.setText(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
        finally:
            # Enable lại buttons
            self._set_buttons_enabled(True)
    
    def _show_digit_results(self, object_images, results):
        """Hiển thị kết quả nhận dạng chữ số"""
        digits_only = "".join([str(digit) for digit, _ in results])
        
        result_text = f"✓ Phát hiện {len(object_images)} chữ số\n\n"
        result_text += f"Kết quả: {' '.join([str(d) for d, _ in results])}\n\n"
        result_text += f"Chuỗi số: {digits_only}\n"
        result_text += f"Độ tin cậy TB: {sum(c for _, c in results) / len(results) * 100:.1f}%"
        self.result_text.setText(result_text)
        
        dialog_text = f"Số nhận dạng được:\n\n{digits_only}\n\n({len(object_images)} chữ số)"
        dialog = ResultDialog(dialog_text, self)
        dialog.show()
    
    def _show_letter_results(self, object_images, results):
        """Hiển thị kết quả nhận dạng chữ cái"""
        letters_only = "".join([letter for letter, _ in results])
        
        result_text = f"✓ Phát hiện {len(object_images)} chữ cái\n\n"
        result_text += f"Kết quả: {' '.join([l for l, _ in results])}\n\n"
        result_text += f"Chuỗi chữ: {letters_only}\n"
        result_text += f"Độ tin cậy TB: {sum(c for _, c in results) / len(results) * 100:.1f}%"
        self.result_text.setText(result_text)
        
        dialog_text = f"Chữ nhận dạng được:\n\n{letters_only}\n\n({len(object_images)} chữ cái)"
        dialog = ResultDialog(dialog_text, self)
        dialog.show()
    
    def _show_shape_results(self, object_images, results):
        """Hiển thị kết quả nhận dạng hình học"""
        shape_names = {'circle': 'Hình tròn', 'rectangle': 'Hình chữ nhật', 'triangle': 'Tam giác'}
        
        shape_counts = {}
        for shape, _ in results:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        result_text = f"✓ Phát hiện {len(object_images)} hình\n\nKết quả:\n"
        for i, (shape, confidence) in enumerate(results):
            vn_shape = shape_names.get(shape, shape)
            result_text += f"  {i+1}. {vn_shape} ({confidence*100:.1f}%)\n"
        result_text += f"\nThống kê:\n"
        for shape, count in sorted(shape_counts.items()):
            result_text += f"  {shape_names.get(shape, shape)}: {count}\n"
        result_text += f"\nĐộ tin cậy TB: {sum(c for _, c in results) / len(results) * 100:.1f}%"
        self.result_text.setText(result_text)
        
        summary = "\n".join([f"{shape_names.get(s, s)}: {c}" for s, c in sorted(shape_counts.items())])
        dialog_text = f"Phát hiện {len(object_images)} hình:\n\n{summary}"
        dialog = ResultDialog(dialog_text, self)
        dialog.show()
    
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
    
    def keyPressEvent(self, event: QKeyEvent):
        """Xử lý phím tắt"""
        if event.key() == Qt.Key_F11 or event.key() == Qt.Key_Escape:
            # F11 hoặc ESC để toggle fullscreen
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)