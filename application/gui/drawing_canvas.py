"""
Drawing Canvas - Canvas vẽ tay cho nhận dạng
"""
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import cv2

class DrawingCanvas(QWidget):
    """Canvas để vẽ tay trực tiếp"""
    
    def __init__(self, width=600, height=400):
        super().__init__()
        self.canvas_width = width
        self.canvas_height = height
        self.image = QImage(width, height, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        
        self.drawing = False
        self.last_point = QPoint()
        self.pen_width = 8
        
        # Set size policy để canvas có thể co giãn
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #999;
                border-radius: 5px;
            }
        """)
    
    def mousePressEvent(self, event):
        """Bắt đầu vẽ khi nhấn chuột"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
    
    def mouseMoveEvent(self, event):
        """Vẽ khi di chuyển chuột"""
        if self.drawing and (event.buttons() & Qt.LeftButton):
            painter = QPainter(self.image)
            pen = QPen(Qt.black, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Kết thúc vẽ khi thả chuột"""
        if event.button() == Qt.LeftButton:
            self.drawing = False
    
    def resizeEvent(self, event):
        """Xử lý khi resize cửa sổ - mở rộng canvas nếu cần"""
        new_size = event.size()
        if new_size.width() > self.image.width() or new_size.height() > self.image.height():
            # Tạo image mới lớn hơn
            new_image = QImage(max(new_size.width(), self.image.width()),
                             max(new_size.height(), self.image.height()),
                             QImage.Format_RGB32)
            new_image.fill(Qt.white)
            
            # Copy nội dung cũ sang image mới
            painter = QPainter(new_image)
            painter.drawImage(0, 0, self.image)
            painter.end()
            
            self.image = new_image
        super().resizeEvent(event)
    
    def paintEvent(self, event):
        """Vẽ canvas"""
        painter = QPainter(self)
        # Vẽ background trắng đầy đủ
        painter.fillRect(self.rect(), Qt.white)
        # Vẽ image
        painter.drawImage(0, 0, self.image)
    
    def clear_canvas(self):
        """Xóa toàn bộ canvas"""
        self.image.fill(Qt.white)
        self.update()
    
    def save_to_file(self, filepath):
        """Lưu canvas thành file ảnh"""
        return self.image.save(filepath)
    
    def get_numpy_array(self):
        """Chuyển canvas thành numpy array để xử lý"""
        # Convert QImage to numpy array
        width = self.image.width()
        height = self.image.height()
        
        ptr = self.image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        
        # Convert RGBA to BGR (OpenCV format)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return bgr
