"""
Subprocess script for letter prediction
Tách biệt để tránh xung đột GUI/TensorFlow
"""
import sys
import pickle
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DEFAULT_MODELS = ['custom_letters_model_final.h5', 'custom_letters_model.h5']

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
        results.append(f"{letter}:{confidence:.4f}")
    
    # In kết quả
    print(','.join(results))

if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print("Usage: python predict_letters_subprocess.py <images_pickle_file> [model_path]", file=sys.stderr)
        sys.exit(1)
    
    images_arg = sys.argv[1]
    model_arg = sys.argv[2] if len(sys.argv) == 3 else None
    predict_letters(images_arg, model_arg)
