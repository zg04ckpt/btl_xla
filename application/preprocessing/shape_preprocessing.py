import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy.ndimage import binary_closing, binary_opening, find_objects, label

from .base_preprocessor import BasePreprocessor


class ShapePreprocessor(BasePreprocessor):
    """
    Preprocessor cho nhận dạng hình học (tam giác / vuông / tròn)
    sử dụng ảnh 64x64.
    """

    def __init__(self, target_size: Tuple[int, int] = (64, 64), inner_size: int = 56) -> None:
        super().__init__(target_size, inner_size)


    def _load_images(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Đọc ảnh RGB + grayscale."""
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            gray = ImageOps.grayscale(rgb)
            return np.array(rgb, dtype=np.uint8), np.array(gray, dtype=np.uint8)

    def _gaussian_blur(self, image: np.ndarray, radius: float = 1.0) -> np.ndarray:
        pil_img = Image.fromarray(image)
        return np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.uint8)

    def _otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return np.zeros_like(image, dtype=np.uint8)

        prob = hist / total
        cum_prob = np.cumsum(prob)
        cum_mean = np.cumsum(prob * np.arange(256))
        global_mean = cum_mean[-1]

        denominator = cum_prob * (1 - cum_prob)
        denominator[denominator == 0] = 1
        between_var = (global_mean * cum_prob - cum_mean) ** 2 / denominator

        threshold = int(np.argmax(between_var))
        return (image > threshold).astype(np.uint8) * 255

    def _morphology(self, binary: np.ndarray) -> np.ndarray:
        mask = binary > 0
        struct = np.ones((3, 3), dtype=bool)
        closed = binary_closing(mask, structure=struct, iterations=2)
        opened = binary_opening(closed, structure=struct, iterations=1)
        return opened.astype(np.uint8) * 255

    def _label_components(self, mask: np.ndarray):
        labeled, count = label(mask)
        slices = find_objects(labeled)
        return labeled, count, slices

    def _draw_boxes(self, base_image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        pil_img = Image.fromarray(base_image.copy())
        draw = ImageDraw.Draw(pil_img)
        for (x0, y0, x1, y1) in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        return np.array(pil_img, dtype=np.uint8)

    # Logic dành cho shape
    def filter_contour(self, bbox: Tuple[int, int, int, int],
                       min_w: int = 20, min_h: int = 20, min_area: int = 400) -> bool:
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if w < min_w or h < min_h:
            return False
        if w * h < min_area:
            return False
        return True

    def preprocess_shape_with_morph(self, binary: np.ndarray, apply_morph: bool) -> np.ndarray:
        return self._morphology(binary) if apply_morph else binary

    def segment_and_preprocess(
        self,
        image_path: str,
        output_path: str | None = None,
        save_images: bool = True,
        apply_morph: bool = True,
        return_steps: bool = False,
    ):
        rgb_img, gray_img = self._load_images(image_path)
        inverted = 255 - gray_img
        blurred = self._gaussian_blur(inverted, radius=1.0)
        binary = self._otsu_threshold(blurred)
        morph = self.preprocess_shape_with_morph(binary, apply_morph)
        mask = morph > 0

        labeled, _, slices = self._label_components(mask)
        boxes: List[Tuple[int, int, int, int]] = []

        for slc in slices:
            if slc is None:
                continue
            y0, y1 = slc[0].start, slc[0].stop
            x0, x1 = slc[1].start, slc[1].stop
            bbox = (x0, y0, x1, y1)
            if self.filter_contour(bbox):
                boxes.append(bbox)

        drawn = self._draw_boxes(rgb_img, boxes)

        if save_images and output_path:
            os.makedirs(output_path, exist_ok=True)

        shapes = []
        for idx, (x0, y0, x1, y1) in enumerate(boxes):
            roi = morph[y0:y1, x0:x1]
            if roi.size == 0:
                continue

            processed = self.preprocess_single(roi)
            shapes.append(processed)

            if save_images and output_path:
                out_path = os.path.join(output_path, f"shape_{idx}.png")
                Image.fromarray((processed * 255).astype(np.uint8)).save(out_path)

        if not return_steps:
            return shapes

        steps: Dict[str, np.ndarray] = {
            "1_original": rgb_img,
            "2_grayscale": gray_img,
            "3_blurred": blurred,
            "4_threshold": binary,
            "5_morphology": morph,
            "6_contours": drawn,
        }

        for idx, processed in enumerate(shapes):
            steps[f"7_shape_{idx}"] = (processed * 255).astype(np.uint8)

        return steps, shapes
