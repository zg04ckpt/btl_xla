"""
Letter Recognition Recognizer - Nhận dạng chữ cái in hoa viết tay (A-Z)
Sử dụng CNN model được train trên EMNIST Letters dataset
"""
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class LetterRecognizer:
    """Nhận dạng chữ cái in hoa viết tay A-Z"""
    
    # Mapping từ class index (0-25) sang chữ cái (A-Z)
    LETTER_MAPPING = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    DEFAULT_MODEL_CANDIDATES = (
        'custom_letters_model_final.h5',
        'custom_letters_model.h5',
    )
    
    def __init__(self, model_path: str = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(os.path.dirname(script_dir))
        self.model = None
        self.model_loaded = False
        
        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.extend(self.DEFAULT_MODEL_CANDIDATES)
        tried_paths = []
        resolved_path = None
        for candidate in candidates:
            path = candidate
            if not os.path.isabs(path):
                path = os.path.join(self.project_root, candidate)
            if path not in tried_paths:
                tried_paths.append(path)
            if os.path.exists(path):
                resolved_path = path
                break
        
        self.model_path = resolved_path
        if not self.model_path:
            checked = '\n - '.join(tried_paths) if tried_paths else 'Không có đường dẫn nào'
            print("⚠ Letter model not found. Đã kiểm tra:\n - " + checked)
            return
        
        print(f"✓ Letter model ready ({os.path.getsize(self.model_path):,} bytes) -> {os.path.basename(self.model_path)}")
    
    def recognize_letters(self, letter_images):
        """
        Nhận dạng chữ cái từ danh sách ảnh 28x28
        
        Args:
            letter_images: List of 28x28 numpy arrays (grayscale, 0-255)
            
        Returns:
            List of tuples (letter, confidence), e.g., [('A', 0.95), ('B', 0.88)]
        """
        if not self.model_path:
            print("⚠ Không tìm thấy model chữ cái, sử dụng fallback demo mode")
            return [(self.LETTER_MAPPING[i % 26], 0.90) for i in range(len(letter_images))]

        try:
            return self._recognize_with_subprocess(letter_images)
        except Exception as e:
            print(f"⚠ Letter prediction failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: demo mode
            print("⚠ Using demo mode - returning A, B, C, D, E, F...")
            return [(self.LETTER_MAPPING[i % 26], 0.90) for i in range(len(letter_images))]
    
    def _recognize_with_subprocess(self, letter_images):
        """Chạy prediction trong subprocess để tránh xung đột GUI/TensorFlow"""
        import subprocess
        import pickle
        import tempfile
        
        # Lưu images vào temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(letter_images, f)
            temp_file = f.name
        
        try:
            predict_script = os.path.join(self.project_root, 'predict_letters_subprocess.py')
            
            # Tạo script nếu chưa có
            if not os.path.exists(predict_script):
                self._create_prediction_script(predict_script)
            
            # Chạy subprocess
            result = subprocess.run(
                ['python', predict_script, temp_file, self.model_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"❌ Subprocess stderr: {result.stderr}")
                print(f"❌ Subprocess stdout: {result.stdout}")
                raise Exception(f"Subprocess failed: {result.stderr}")
            
            # Parse kết quả (format: "A:0.95,B:0.88,...")
            predictions = []
            for pred in result.stdout.strip().split(','):
                if ':' in pred:
                    letter, conf = pred.split(':')
                    predictions.append((letter, float(conf)))
            
            return predictions
            
        finally:
            # Xóa temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _create_prediction_script(self, script_path):
        """Tạo subprocess script để load model và predict"""
        default_model, fallback_model = self.DEFAULT_MODEL_CANDIDATES
        script_content = f'''"""
Subprocess script for letter prediction
Tách biệt để tránh xung đột GUI/TensorFlow
"""
import sys
import pickle
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DEFAULT_MODELS = {list(self.DEFAULT_MODEL_CANDIDATES)}

def _resolve_model_path(path_from_cli=None):
    candidates = []
    if path_from_cli:
        candidates.append(path_from_cli)
    candidates.extend(DEFAULT_MODELS)
    searched = []
    for candidate in candidates:
        candidate_path = candidate
        if not os.path.isabs(candidate_path):
            candidate_path = os.path.join(os.getcwd(), candidate)
        if candidate_path not in searched:
            searched.append(candidate_path)
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError("Letter model not found. Checked: " + " | ".join(searched))

def predict_letters(images_file, model_path=None):
    """Load model và predict letters"""
    import tensorflow as tf
    from tensorflow import keras
    
    model_real_path = _resolve_model_path(model_path)
    model = keras.models.load_model(model_real_path)
    
    # Load images
    with open(images_file, 'rb') as f:
        images = pickle.load(f)
    
    # Chuẩn bị dữ liệu
    X = np.array(images, dtype=np.float32)
    
    # Normalize về [0, 1]
    X = X / 255.0
    
    # Reshape thành (N, 28, 28, 1) cho CNN
    X = X.reshape(-1, 28, 28, 1)
    
    # Predict
    predictions = model.predict(X, verbose=0)
    
    # Chuyển thành chữ cái và confidence
    LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    results = []
    for pred in predictions:
        idx = np.argmax(pred)
        confidence = float(pred[idx])
        letter = LETTERS[idx]
        results.append(f"{{letter}}:{{confidence:.4f}}")
    
    # In kết quả
    print(','.join(results))

if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print("Usage: python predict_letters_subprocess.py <images_pickle_file> [model_path]", file=sys.stderr)
        sys.exit(1)
    
    images_arg = sys.argv[1]
    model_arg = sys.argv[2] if len(sys.argv) == 3 else None
    predict_letters(images_arg, model_arg)
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✓ Created prediction script: {script_path}")
