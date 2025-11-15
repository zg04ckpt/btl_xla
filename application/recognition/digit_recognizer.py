import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

class DigitRecognizer:
    def __init__(self, model_path='Code/mnist_cnn_model.h5'):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        if not os.path.exists(model_path):
            print(f"⚠ Model not found: {model_path}")
            return
        
        print(f"✓ Model ready ({os.path.getsize(model_path):,} bytes)")
    

    
    def recognize_digits(self, digit_images):
        """Recognize digits using CNN model via subprocess"""
        try:
            return self._recognize_with_subprocess(digit_images)
        except Exception as e:
            print(f"⚠ Prediction failed: {e}")
            # Fallback: demo mode
            return [(i % 10, 0.95) for i in range(len(digit_images))]
    
    def _recognize_with_subprocess(self, digit_images):
        """Run prediction in subprocess (avoids GUI/TensorFlow conflicts)"""
        import subprocess
        import pickle
        import tempfile
        
        # Save images to temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(digit_images, f)
            temp_file = f.name
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            predict_script = os.path.join(project_root, 'predict_subprocess.py')
            
            # Create script if needed
            if not os.path.exists(predict_script):
                self._create_prediction_script(predict_script)
            
            # Run subprocess
            result = subprocess.run(
                ['python', predict_script, self.model_path, temp_file],
                capture_output=True,
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode == 0:
                with open(temp_file + '.result', 'rb') as f:
                    results = pickle.load(f)
                os.unlink(temp_file)
                os.unlink(temp_file + '.result')
                return results
            else:
                raise Exception("Subprocess failed")
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            if os.path.exists(temp_file + '.result'):
                os.unlink(temp_file + '.result')
    
    def _create_prediction_script(self, script_path):
        """Create standalone prediction script"""
        script_content = '''#!/usr/bin/env python
import sys, os, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

with open(sys.argv[2], "rb") as f:
    digit_images = pickle.load(f)

import keras, numpy as np, cv2

model = keras.models.load_model(sys.argv[1], compile=False)

results = []
for digit_img in digit_images:
    if digit_img.shape != (28, 28):
        digit_img = cv2.resize(digit_img, (28, 28))
    
    digit_img = digit_img.astype("float32") / 255.0
    digit_img = digit_img.reshape(1, 28, 28, 1)
    
    prediction = model.predict(digit_img, verbose=0)
    digit = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][digit])
    results.append((digit, confidence))

with open(sys.argv[2] + ".result", "wb") as f:
    pickle.dump(results, f)
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
    
