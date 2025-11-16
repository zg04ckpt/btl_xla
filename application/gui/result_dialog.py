"""
Result Dialog - Hiển thị kết quả nhận dạng ở giữa màn hình
"""
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class ResultDialog(QDialog):
    """Dialog hiển thị kết quả nhận dạng (kích thước 567x433 = 2/3 của 850x650)"""
    
    def __init__(self, result_text, parent=None):
        super().__init__(parent)
        self.result_text = result_text
        self.init_ui()
    
    def init_ui(self):
        """Khởi tạo giao diện"""
        self.setWindowTitle("Kết quả Nhận dạng")
        self.setFixedSize(567, 433)  # 2/3 kích thước app (850x650)
        
        # Layout chính
        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)
        
        # Tiêu đề
        title = QLabel("🎯 KẾT QUẢ")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #2196F3;")
        layout.addWidget(title)
        
        # Khung kết quả
        result_frame = QFrame()
        result_frame.setStyleSheet("""
            QFrame {
                background-color: #E3F2FD;
                border: 2px solid #2196F3;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        result_layout = QVBoxLayout()
        
        # Nội dung kết quả
        result_label = QLabel(self.result_text)
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setWordWrap(True)
        result_font = QFont()
        result_font.setPointSize(16)
        result_label.setFont(result_font)
        result_label.setStyleSheet("color: #1565C0;")
        result_layout.addWidget(result_label)
        
        result_frame.setLayout(result_layout)
        layout.addWidget(result_frame, 1)  # Chiếm phần lớn không gian
        
        # Nút đóng
        close_btn = QPushButton("✓ Đóng")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        
        # Căn giữa màn hình
        self.center()
    
    def center(self):
        """Căn giữa dialog trên màn hình"""
        if self.parent():
            # Căn giữa so với cửa sổ cha
            parent_geo = self.parent().geometry()
            x = parent_geo.x() + (parent_geo.width() - self.width()) // 2
            y = parent_geo.y() + (parent_geo.height() - self.height()) // 2
            self.move(x, y)
