"""Debug script to test shape preprocessing"""
from application.preprocessing import ShapePreprocessor

proc = ShapePreprocessor(target_size=(64, 64), inner_size=56)

# Test với ảnh vẽ tay hình học
steps, imgs = proc.process_image('test-chu-3.png', mode='shapes')

print("Steps keys:", list(steps.keys()))
print("Number of shapes detected:", len(imgs))

# Kiểm tra xem có step nào chứa '7_shape_' không
shape_steps = [k for k in steps.keys() if k.startswith('7_shape_')]
letter_steps = [k for k in steps.keys() if k.startswith('7_letter_')]
digit_steps = [k for k in steps.keys() if k.startswith('7_digit_')]

print(f"\nShape steps (7_shape_*): {len(shape_steps)}")
print(f"Letter steps (7_letter_*): {len(letter_steps)}")
print(f"Digit steps (7_digit_*): {len(digit_steps)}")
