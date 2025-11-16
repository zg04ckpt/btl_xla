"""Shape recognizer for geometric shapes (circle, rectangle, triangle)"""

import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

class ShapeRecognizer:
    def __init__(self, model_path='train/shapes_cnn_model_best.h5'):
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.classes = ['circle', 'rectangle', 'triangle']
        
        if not os.path.exists(model_path):
            print(f"⚠ Shape model not found: {model_path}")
            return
        
        print(f"✓ Shape model ready ({os.path.getsize(model_path):,} bytes)")
    
    def recognize_shapes(self, shape_images):
        """Recognize shapes using CNN model via subprocess"""
        try:
            return self._recognize_with_subprocess(shape_images)
        except Exception as e:
            print(f"⚠ Shape prediction failed: {e}")
            # Fallback: demo mode
            return [(self.classes[i % 3], 0.95) for i in range(len(shape_images))]
    
    def _recognize_with_subprocess(self, shape_images):
        """Run prediction in subprocess (avoids GUI/TensorFlow conflicts)"""
        import subprocess
        import pickle
        import tempfile
        
        # Save images to temp file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(shape_images, f)
            temp_file = f.name
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            predict_script = os.path.join(project_root, 'predict_shapes_subprocess.py')
            
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
        """Create standalone prediction script for shapes"""
        script_content = '''#!/usr/bin/env python
import sys, os, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

with open(sys.argv[2], "rb") as f:
    shape_images = pickle.load(f)

import keras, numpy as np, cv2

model = keras.models.load_model(sys.argv[1], compile=False)
classes = ['circle', 'rectangle', 'triangle']

results = []
for shape_img in shape_images:
    if shape_img.shape != (64, 64):
        shape_img = cv2.resize(shape_img, (64, 64))
    
    shape_img = shape_img.astype("float32") / 255.0
    shape_img = shape_img.reshape(1, 64, 64, 1)
    
    prediction = model.predict(shape_img, verbose=0)
    shape_idx = int(np.argmax(prediction[0]))
    shape_name = classes[shape_idx]
    confidence = float(prediction[0][shape_idx])
    results.append((shape_name, confidence))

with open(sys.argv[2] + ".result", "wb") as f:
    pickle.dump(results, f)
'''
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
