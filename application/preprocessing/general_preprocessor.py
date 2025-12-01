import numpy as np
from PIL import Image
import os

class GeneralPreprocessor:
    def __init__(self, target_size=(28, 28), inner_size=20, min_h=10, min_w=10, min_area=100):
        self.target_size = target_size
        self.inner_size = inner_size
        self.min_h = min_h
        self.min_w = min_w
        self.min_area = min_area

    def read_grayscale(self, path):
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.uint8)
        R = arr[:, :, 0].astype(np.float32)
        G = arr[:, :, 1].astype(np.float32)
        B = arr[:, :, 2].astype(np.float32)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
        return gray

    def otsu_threshold_binary(self, gray_image):
        hist = np.bincount(gray_image.flatten(), minlength=256)
        total_pixels = gray_image.size
        total_pixel_values = np.sum(np.arange(256) * hist)
        sumB = 0
        wB = 0
        max_var = 0
        threshold = 0
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total_pixels - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (total_pixel_values - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = t
        binary_image = np.zeros_like(gray_image, dtype=np.uint8)
        binary_image[gray_image >= threshold] = 255
        return binary_image

    def connected_components(self, binary):
        H, W = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        components = []
        for y in range(H):
            for x in range(W):
                if binary[y, x] == 255 and not visited[y, x]:
                    stack = [(x, y)]
                    visited[y, x] = True
                    pixels = []
                    while stack:
                        px, py = stack.pop()
                        pixels.append((px, py))
                        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < W and 0 <= ny < H:
                                if binary[ny, nx] == 255 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    stack.append((nx, ny))
                    components.append(pixels)
        return components

    def crop_component(self, binary, component):
        xs = [p[0] for p in component]
        ys = [p[1] for p in component]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return binary[y1:y2+1, x1:x2+1].copy()

    def resize_nearest(self, img, new_w, new_h):
        h, w = img.shape
        if new_h <= 0 or new_w <= 0:
            return np.zeros((max(1, new_h), max(1, new_w)), dtype=np.uint8)
        y_scale = h / new_h
        x_scale = w / new_w
        resized_img = np.zeros((new_h, new_w), dtype=np.uint8)
        for i in range(new_h):
            for j in range(new_w):
                src_y = min(int(i * y_scale), h - 1)
                src_x = min(int(j * x_scale), w - 1)
                resized_img[i, j] = img[src_y, src_x]
        return resized_img

    def remove_padding(self, img):
        rows_sum = np.sum(img, axis=1)
        rows_idx = np.where(rows_sum > 0)[0]
        if len(rows_idx) == 0:
            return img
        cols_sum = np.sum(img, axis=0)
        cols_idx = np.where(cols_sum > 0)[0]
        if len(cols_idx) == 0:
            return img
        img = img[rows_idx[0]:rows_idx[-1]+1, cols_idx[0]:cols_idx[-1]+1]
        return img

    def center_of_mass(self, binary):
        ys, xs = np.nonzero(binary)
        if len(xs) == 0:
            return binary.shape[1] // 2, binary.shape[0] // 2
        cx = np.mean(xs)
        cy = np.mean(ys)
        return cx, cy

    def shift_image(self, img, sx, sy):
        h, w = img.shape
        new = np.zeros_like(img)
        for y in range(h):
            for x in range(w):
                nx = x + sx
                ny = y + sy
                if 0 <= nx < w and 0 <= ny < h:
                    new[ny, nx] = img[y, x]
        return new

    def segment_and_preprocess(self, image_path, output_path, save_images=True, return_steps=False):
        print(f"Đang xử lý: {image_path}")
        img = self.read_grayscale(image_path)
        print(f"Đã đọc ảnh: {img.shape}")
        img_inv = 255 - img
        binary = self.otsu_threshold_binary(img_inv)
        print("Đã tạo ảnh nhị phân")
        comps = self.connected_components(binary)
        print(f"Tìm thấy {len(comps)} components")
        comps = sorted(comps, key=lambda comp: min([p[0] for p in comp]))
        if save_images and output_path:
            os.makedirs(output_path, exist_ok=True)
        results = []
        count = 0
        target_h, target_w = self.target_size
        # Lưu các bước nếu cần
        steps = {}
        if return_steps:
            steps["1_grayscale"] = img.copy()
            steps["2_inverted"] = img_inv.copy()
            steps["3_binary"] = binary.copy()
        boxes = []
        for comp in comps:
            digit = self.crop_component(binary, comp)
            h, w = digit.shape
            if not self.filter_component(h, w):
                continue
            digit = self.remove_padding(digit)
            if digit.shape[0] == 0 or digit.shape[1] == 0:
                continue
            h, w = digit.shape
            if h > w:
                new_h = self.inner_size
                new_w = max(1, int(round(w * (float(self.inner_size) / h))))
            else:
                new_w = self.inner_size
                new_h = max(1, int(round(h * (float(self.inner_size) / w))))
            digit = self.resize_nearest(digit, new_w, new_h)
            pad_h = target_h - new_h
            pad_w = target_w - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            digit = np.pad(
                digit,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
            if digit.shape[0] != target_h or digit.shape[1] != target_w:
                digit = digit[:target_h, :target_w]
            cx, cy = self.center_of_mass(digit)
            shift_x = int(round(target_w / 2.0 - cx))
            shift_y = int(round(target_h / 2.0 - cy))
            digit = self.shift_image(digit, shift_x, shift_y)
            digit_norm = digit / 255.0
            results.append(digit_norm)
            if save_images and output_path:
                save_path = os.path.join(output_path, f'digit_{count}.png')
                Image.fromarray(digit).save(save_path)
                print(f"✓ Đã lưu ảnh {count} tại: {save_path}")
            if return_steps:
                steps[f"7_object_{count}"] = digit.copy()
                boxes.append(self._get_bbox_from_component(comp))
            count += 1
        if return_steps:
            # Vẽ bounding box lên ảnh nhị phân
            if len(boxes) > 0:
                import copy
                bin_rgb = np.stack([binary]*3, axis=-1)
                for (x1, y1, x2, y2) in boxes:
                    bin_rgb[y1:y2+1, x1] = [0,255,0]
                    bin_rgb[y1:y2+1, x2] = [0,255,0]
                    bin_rgb[y1, x1:x2+1] = [0,255,0]
                    bin_rgb[y2, x1:x2+1] = [0,255,0]
                steps["6_contours"] = bin_rgb
            return steps, results
        print(f"\nHoàn thành")
        return results

    def filter_component(self, h, w):
        return h >= self.min_h and w >= self.min_w and h * w >= self.min_area

    def _get_bbox_from_component(self, comp):
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (x1, y1, x2, y2)
