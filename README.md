# Handwritten Digit Recognition / Nhận dạng Chữ số Viết tay

Ứng dụng desktop nhận dạng chữ số viết tay sử dụng CNN và OpenCV.

## Tính năng

- ✅ Nhận dạng nhiều chữ số trong một ảnh
- ✅ Hiển thị 7 bước tiền xử lý ảnh  
- ✅ Hỗ trợ drag & drop, upload ảnh
- ✅ Giao diện song ngữ Việt/Anh

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
python src/main.py
```

## Cấu trúc dự án

```
├── application/
│   ├── main.py                      # Entry point
│   ├── gui/
│   │   ├── main_window.py          # Giao diện chính
│   │   └── preprocessing_viewer.py  # Hiển thị preprocessing
│   ├── preprocessing/
│   │   └── image_processor.py       # Xử lý ảnh 7 bước
│   └── recognition/
│       └── digit_recognizer.py      # Nhận dạng CNN
├── Codes/
│   └── mnist_cnn_model.h5           # Model đã train
├── config.py                         # Cấu hình
└── predict_subprocess.py            # Script prediction riêng
```

## Các bước xử lý ảnh

1. **Original** - Ảnh gốc
2. **Grayscale** - Chuyển sang ảnh xám
3. **Blurred** - Giảm nhiễu Gaussian
4. **Threshold** - Adaptive thresholding
5. **Morphology** - Phép toán hình thái
6. **Contours** - Phát hiện & lọc contour
7. **Digits** - Tách chữ số 28x28

## Yêu cầu hệ thống

- Python 3.8+
- TensorFlow/Keras 2.10+
- OpenCV 4.5+
- PyQt5 5.15+

## Lưu ý

- Model được load qua subprocess để tránh xung đột GUI/TensorFlow
- Nếu model không load được, ứng dụng chạy ở demo mode
