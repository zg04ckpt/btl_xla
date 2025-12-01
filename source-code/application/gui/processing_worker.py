"""
Processing Worker - Thread để xử lý ảnh không block UI
"""
from PyQt5.QtCore import QThread, pyqtSignal
import traceback


class ProcessingWorker(QThread):
    """Worker thread để xử lý ảnh và nhận dạng"""
    
    # Signals để giao tiếp với main thread
    finished = pyqtSignal(dict)  # Khi hoàn thành, trả về kết quả
    error = pyqtSignal(str)  # Khi có lỗi
    progress = pyqtSignal(str)  # Cập nhật tiến trình
    
    def __init__(self, image_path, recognition_mode, preprocessors, recognizers):
        super().__init__()
        self.image_path = image_path
        self.recognition_mode = recognition_mode
        self.preprocessors = preprocessors  # dict: {'digit': DigitPreprocessor, 'letter': LetterPreprocessor, 'shape': ShapePreprocessor}
        self.recognizers = recognizers  # dict: {'digit': DigitRecognizer, 'letter': LetterRecognizer, 'shape': ShapeRecognizer}
        
    def run(self):
        """Chạy xử lý ảnh trên background thread"""
        try:
            # Bước 1: Tiền xử lý ảnh
            self.progress.emit("Đang tiền xử lý ảnh...")
            
            # Sử dụng preprocessor phù hợp với mode
            if self.recognition_mode == 'letters':
                preprocessing_steps, object_images = self.preprocessors['letter'].segment_and_preprocess(
                    self.image_path, output_path=None, save_images=False, return_steps=True
                )
            elif self.recognition_mode == 'shapes':
                preprocessing_steps, object_images = self.preprocessors['shape'].segment_and_preprocess(
                    self.image_path, output_path=None, save_images=False, return_steps=True
                )
            else:  # digits
                preprocessing_steps, object_images = self.preprocessors['digit'].segment_and_preprocess(
                    self.image_path, output_path=None, save_images=False, return_steps=True
                )
            
            if not object_images:
                self.error.emit("Không phát hiện đối tượng nào!")
                return
            
            # Bước 2: Nhận dạng
            mode_texts = {'digits': 'chữ số', 'letters': 'chữ cái', 'shapes': 'hình học'}
            mode_text = mode_texts.get(self.recognition_mode, 'đối tượng')
            self.progress.emit(f"Đang nhận dạng {mode_text}...")
            
            if self.recognition_mode == 'digits':
                results = self.recognizers['digit'].recognize_digits(object_images)
            elif self.recognition_mode == 'letters':
                results = self.recognizers['letter'].recognize_letters(object_images)
            else:  # shapes
                results = self.recognizers['shape'].recognize_shapes(object_images)
            
            # Trả kết quả về main thread
            self.finished.emit({
                'preprocessing_steps': preprocessing_steps,
                'object_images': object_images,
                'results': results,
                'mode': self.recognition_mode
            })
            
        except Exception as e:
            error_msg = f"Lỗi xử lý: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
