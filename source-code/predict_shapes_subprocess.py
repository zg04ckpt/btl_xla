#!/usr/bin/env python
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
    
    # Preprocessor đã normalize rồi, không cần chia 255 nữa
    shape_img = shape_img.astype("float32")
    shape_img = shape_img.reshape(1, 64, 64, 1)
    
    prediction = model.predict(shape_img, verbose=0)
    shape_idx = int(np.argmax(prediction[0]))
    shape_name = classes[shape_idx]
    confidence = float(prediction[0][shape_idx])
    results.append((shape_name, confidence))

with open(sys.argv[2] + ".result", "wb") as f:
    pickle.dump(results, f)
