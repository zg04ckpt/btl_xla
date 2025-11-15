# Cập nhật chức năng phát hiện và cắt ảnh (Detection & Crop)

## Tổng quan
Đã thêm các chức năng phát hiện và cắt ảnh từ file `Code/test.py` vào `application/preprocessing/image_processor.py` để cải thiện độ chính xác trong việc nhận dạng chữ số viết tay.

## Các thay đổi chính

### 1. **Thay đổi phương pháp ngưỡng hóa (Thresholding)**
- **Trước:** Adaptive Threshold
- **Sau:** OTSU Threshold (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
- **Lý do:** OTSU tự động tìm ngưỡng tối ưu, phù hợp hơn cho chữ số viết tay

```python
# OTSU thresholding (better than adaptive for handwritten digits)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

### 2. **Cải thiện Morphological Operations**
- Tăng số lần iteration cho MORPH_CLOSE (1 → 2) để loại bỏ nhiễu tốt hơn
- Thêm MORPH_OPEN để làm sạch thêm

```python
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
```

### 3. **Lọc contour thông minh với Median Height**
- Tính median height của tất cả contours để lọc thích ứng
- Mở rộng phạm vi aspect ratio (0.2 - 8.0) để bắt được chữ số mỏng như "1"
- Giảm ngưỡng diện tích tối thiểu để không bỏ sót chữ số nhỏ

```python
# Calculate median height for adaptive filtering
all_heights = [cv2.boundingRect(cnt)[3] for cnt in contours]
median_height = np.median(all_heights) if all_heights else 0

# More flexible filtering
if (h > max(10, median_height * 0.2) and  # Lower minimum height
    0.2 < aspect_ratio < 8.0 and  # Much wider range for thin digits
    area > 50 and  # Lower minimum area
    w > 3):  # Lower minimum width for "1"
```

### 4. **Sắp xếp theo hàng (Row-based Sorting)**
Thêm hàm `_sort_boxes_by_row()` để xử lý chữ số trên nhiều hàng:
- Nhóm các chữ số theo hàng dựa trên tọa độ y
- Sắp xếp từ trên xuống dưới (top-to-bottom)
- Trong cùng một hàng, sắp xếp từ trái sang phải (left-to-right)

```python
def _sort_boxes_by_row(self, boxes, median_height):
    """Sort bounding boxes by row then column (top-to-bottom, left-to-right)"""
    # Groups boxes by rows and sorts them properly
```

### 5. **Tiền xử lý chữ số cải tiến**
Thêm hàm `_preprocess_digit()` với các bước:
1. Tìm và cắt bounding box chính xác
2. Thêm padding 2px
3. Resize với tỷ lệ khung hình được bảo toàn (max 20x20)
4. Đặt vào canvas 28x28 (chuẩn MNIST)

```python
def _preprocess_digit(self, digit_image):
    """Convert digit image to MNIST format (28x28, black background, white digit)"""
    # Find bounding box and crop
    # Add padding
    # Resize maintaining aspect ratio
    # Create 28x28 canvas
```

### 6. **Cải thiện Cropping với Padding**
- Thêm padding 5px khi crop để giữ nguyên vẹn chữ số
- Sử dụng PIL Image để crop chính xác hơn

```python
padding = 5
digit_pil = pil_image.crop((
    max(0, x - padding), 
    max(0, y - padding), 
    min(morph.shape[1], x + w + padding), 
    min(morph.shape[0], y + h + padding)
))
```

## Lợi ích

✅ **Phát hiện tốt hơn:** Bắt được cả chữ số mỏng (như "1") và chữ số nhỏ  
✅ **Xử lý nhiều hàng:** Hỗ trợ ảnh có chữ số sắp xếp trên nhiều hàng  
✅ **Cắt chính xác:** Padding và bounding box chính xác hơn  
✅ **Chuẩn MNIST:** Tiền xử lý đúng format 28x28 như mô hình được train  
✅ **Loại bỏ nhiễu:** Morphological operations mạnh mẽ hơn  

## Cách sử dụng

Không cần thay đổi gì! Các chức năng mới được tích hợp vào `ImageProcessor` class:

```python
processor = ImageProcessor()
preprocessing_steps, digit_images = processor.process_image("path/to/image.png")
```

## Testing

Để kiểm tra, chạy ứng dụng:
```bash
python application/main.py
```

Tải ảnh có chữ số viết tay và nhấn "Xử lý" để thấy kết quả cải thiện!
