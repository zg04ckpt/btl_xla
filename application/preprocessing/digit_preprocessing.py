import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.ndimage import binary_closing, binary_opening, label, find_objects

from .base_preprocessor import BasePreprocessor


class DigitPreprocessor(BasePreprocessor):
    def __init__(self, target_size=(28, 28), inner_size=20):
        super().__init__(target_size, inner_size)

    def _load_gray_and_rgb(self, path: str):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            gray = ImageOps.grayscale(rgb)
            return np.array(rgb), np.array(gray)

    def _gaussian_blur(self, image: np.ndarray, radius: float = 1.0) -> np.ndarray:
        return np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius)))

    def _otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        prob = hist / hist.sum()

        cum_prob = np.cumsum(prob)
        cum_mean = np.cumsum(prob * np.arange(256))
        global_mean = cum_mean[-1]

        denom = cum_prob * (1 - cum_prob)
        denom[denom == 0] = 1

        sigma_b = (global_mean * cum_prob - cum_mean) ** 2 / denom
        thresh = int(np.argmax(sigma_b))

        return (image > thresh).astype(np.uint8) * 255

    def _morphology(self, binary: np.ndarray) -> np.ndarray:
        mask = binary > 0
        kernel = np.ones((3, 3), dtype=bool)
        closed = binary_closing(mask, structure=kernel, iterations=2)
        opened = binary_opening(closed, structure=kernel, iterations=1)
        return opened.astype(np.uint8) * 255

    def _draw_boxes(self, base_img: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        pil = Image.fromarray(base_img.copy())
        draw = ImageDraw.Draw(pil)
        for x0, y0, x1, y1 in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        return np.array(pil)

    # --- filter ---
    def filter_bbox(self, bbox, min_w=10, min_h=20, min_area=200, min_ratio=0.2, max_ratio=5.0):
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0

        if w < min_w or h < min_h or w * h < min_area:
            return False

        ratio = w / h if h > 0 else 0
        return min_ratio <= ratio <= max_ratio

    # --- main API ---
    def segment_and_preprocess(self, image_path, output_path=None, save_images=True, return_steps=False):
        rgb, gray = self._load_gray_and_rgb(image_path)
        inverted = 255 - gray
        blurred = self._gaussian_blur(inverted, radius=1.0)
        binary = self._otsu_threshold(blurred)
        morph = self._morphology(binary)

        mask = morph > 0
        labeled, count = label(mask)
        slices = find_objects(labeled)

        boxes = []
        for slc in slices:
            if slc is None:
                continue
            y0, y1 = slc[0].start, slc[0].stop
            x0, x1 = slc[1].start, slc[1].stop
            bbox = (x0, y0, x1, y1)
            if self.filter_bbox(bbox):
                boxes.append(bbox)

        digits = []
        if save_images and output_path:
            os.makedirs(output_path, exist_ok=True)

        for idx, (x0, y0, x1, y1) in enumerate(boxes):
            roi = morph[y0:y1, x0:x1]
            processed = self.preprocess_single(roi)
            digits.append(processed)

            if save_images and output_path:
                Image.fromarray((processed * 255).astype(np.uint8)).save(
                    os.path.join(output_path, f"digit_{idx}.png")
                )

        if not return_steps:
            return digits

        steps: Dict[str, np.ndarray] = {
            "1_original": rgb,
            "2_grayscale": gray,
            "3_blurred": blurred,
            "4_threshold": binary,
            "5_morphology": morph,
            "6_contours": self._draw_boxes(rgb, boxes),
        }

        for idx, processed in enumerate(digits):
            steps[f"7_digit_{idx}"] = (processed * 255).astype(np.uint8)

        return steps, digits
