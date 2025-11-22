"""Image preprocessing module for shape recognition."""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional


class ImageProcessor:
    """Process images for shape recognition using CNN model."""
    
    # Preprocessing constants
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    MORPH_KERNEL_SIZE = (3, 3)
    MORPH_CLOSE_ITERATIONS = 2
    MORPH_OPEN_ITERATIONS = 1
    
    # Shape preprocessing constants
    SHAPE_SIZE = 64
    SHAPE_RESIZE_MAX = 56
    SHAPE_PADDING = 10
    SHAPE_BORDER_PADDING = 4
    
    # Contour filtering thresholds for shapes
    MIN_HEIGHT = 10
    MIN_WIDTH = 3
    MIN_AREA = 50
    ROW_THRESHOLD_FACTOR = 0.5
    
    def __init__(self):
        """Initialize ImageProcessor with empty preprocessing steps dictionary."""
        self.preprocessing_steps: Dict[str, np.ndarray] = {}
    
    def process_image(self, image_path: str, mode: str = 'shapes') -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Process image through all preprocessing steps for shape recognition.
        
        Args:
            image_path: Path to the input image file
            mode: Recognition mode - only 'shapes' is supported
            
        Returns:
            Tuple containing:
                - Dictionary of preprocessing steps for visualization
                - List of preprocessed 64x64 shape images
                
        Raises:
            ValueError: If image cannot be loaded
        """
        # Xóa các bước xử lý cũ từ lần trước
        self.preprocessing_steps.clear()
        
        # Load and validate image
        original = self._load_image(image_path)
        self.preprocessing_steps['1_original'] = original.copy()
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        self.preprocessing_steps['2_grayscale'] = gray
        
        # Step 2: Noise reduction
        blurred = cv2.GaussianBlur(gray, self.GAUSSIAN_KERNEL_SIZE, 0)
        self.preprocessing_steps['3_blurred'] = blurred
        
        # Step 3: OTSU thresholding
        thresh = self._apply_threshold(blurred)
        self.preprocessing_steps['4_threshold'] = thresh
        
        # Step 4: Morphological operations
        morph = self._apply_morphology(thresh)
        self.preprocessing_steps['5_morphology'] = morph
        
        # Step 5: Detect and filter shape contours
        valid_contours, median_height = self._detect_shapes(morph, original, gray.shape)
        
        # Step 6: Sort shapes by reading order
        sorted_contours = self._sort_boxes_by_row(valid_contours, median_height)
        
        # Step 7: Extract and preprocess individual shapes
        shape_images = self._extract_shapes(morph, sorted_contours)
        
        return self.preprocessing_steps, shape_images
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Cannot load image from: {image_path}")
        return original
    
    def _apply_threshold(self, blurred: np.ndarray) -> np.ndarray:
        """Apply OTSU thresholding for better digit extraction."""
        _, thresh = cv2.threshold(
            blurred, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return thresh
    
    def _apply_morphology(self, thresh: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean noise."""
        kernel = np.ones(self.MORPH_KERNEL_SIZE, np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 
                                  iterations=self.MORPH_CLOSE_ITERATIONS)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, 
                                 iterations=self.MORPH_OPEN_ITERATIONS)
        return morph
    
    def _detect_shapes(self, morph: np.ndarray, original: np.ndarray, 
                       image_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        Detect and filter valid shape contours.
        
        Returns:
            Tuple of (valid_contours, median_height)
        """
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate median height for adaptive filtering
        all_heights = [cv2.boundingRect(cnt)[3] for cnt in contours]
        median_height = np.median(all_heights) if all_heights else 0
        
        image_height, image_width = image_shape
        valid_contours = []
        contour_img = original.copy()
        
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            
            # Check if contour is a valid shape
            is_valid = self._is_valid_shape_contour(bbox, contour, median_height, 
                                                    image_height, image_width)
            
            if is_valid:
                x, y, w, h = bbox
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                valid_contours.append(bbox)
            else:
                x, y, w, h = bbox
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        
        self.preprocessing_steps['6_contours'] = contour_img
        return valid_contours, median_height
    
    def _is_valid_shape_contour(self, bbox: Tuple[int, int, int, int], 
                                contour: np.ndarray, median_height: float,
                                image_height: int, image_width: int) -> bool:
        """Check if contour is a valid shape (more relaxed aspect ratio for squares/circles)."""
        x, y, w, h = bbox
        
        if w == 0 or h == 0:
            return False
        
        aspect_ratio = h / float(w)
        area = cv2.contourArea(contour)
        
        # Shapes are typically more square-like, so wider aspect ratio range
        min_size = max(20, min(image_height, image_width) * 0.1)
        
        return (w > min_size and
                h > min_size and
                0.3 < aspect_ratio < 3.0 and  # Allow rectangles
                area > 500 and  # Larger minimum area for shapes
                w < image_width * 0.9 and
                h < image_height * 0.9)
    
    def _extract_shapes(self, morph: np.ndarray, 
                       contours: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Extract and preprocess individual shapes from the image."""
        shape_images = []
        pil_image = Image.fromarray(morph)
        
        for i, (x, y, w, h) in enumerate(contours):
            # Crop with padding
            shape_pil = pil_image.crop((
                max(0, x - self.SHAPE_PADDING), 
                max(0, y - self.SHAPE_PADDING), 
                min(morph.shape[1], x + w + self.SHAPE_PADDING), 
                min(morph.shape[0], y + h + self.SHAPE_PADDING)
            ))
            
            # Preprocess to 64x64 format with center of mass
            shape_processed = self._preprocess_shape(shape_pil)
            if shape_processed is not None:
                shape_images.append(shape_processed)
                self.preprocessing_steps[f'7_shape_{i}'] = shape_processed
        
        return shape_images
    
    def _sort_boxes_by_row(self, boxes: List[Tuple[int, int, int, int]], 
                          median_height: float) -> List[Tuple[int, int, int, int]]:
        """
        Sort bounding boxes by row then column (top-to-bottom, left-to-right).
        
        Args:
            boxes: List of bounding boxes (x, y, w, h)
            median_height: Median height used for row grouping threshold
            
        Returns:
            Sorted list of bounding boxes in reading order
        """
        if not boxes:
            return boxes
        
        # Sort by y-coordinate first
        boxes_sorted = sorted(boxes, key=lambda b: b[1])
        
        # Group boxes into rows
        rows = []
        current_row = [boxes_sorted[0]]
        row_threshold = median_height * self.ROW_THRESHOLD_FACTOR
        
        for box in boxes_sorted[1:]:
            # Check if box belongs to current row
            if abs(box[1] - current_row[0][1]) < row_threshold:
                current_row.append(box)
            else:
                # Save current row (sorted by x) and start new row
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
        
        # Add the last row
        rows.append(sorted(current_row, key=lambda b: b[0]))
        
        # Flatten all rows into a single list
        return [box for row in rows for box in row]
    
    def _crop_to_content(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Crop image to tight bounding box around non-zero pixels."""
        coords = cv2.findNonZero(image)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return image[y:y+h, x:x+w]
        return None
    
    def _preprocess_shape(self, shape_image: Image.Image) -> Optional[np.ndarray]:
        """
        Convert shape image to 64x64 format (black background, white shape).
        Uses center of mass alignment for better shape recognition.
        
        Args:
            shape_image: PIL Image of the shape
            
        Returns:
            64x64 numpy array ready for CNN model, or None if processing fails
        """
        try:
            shape_np = np.array(shape_image)
            
            # Crop to tight bounding box
            shape_np = self._crop_to_content(shape_np)
            if shape_np is None:
                return None
            
            # Add border padding
            shape_np = cv2.copyMakeBorder(
                shape_np, 
                self.SHAPE_BORDER_PADDING, self.SHAPE_BORDER_PADDING, 
                self.SHAPE_BORDER_PADDING, self.SHAPE_BORDER_PADDING, 
                cv2.BORDER_CONSTANT, value=0
            )
            
            # Resize with aspect ratio preservation (max 56x56)
            shape_resized = self._resize_shape_with_aspect_ratio(shape_np)
            if shape_resized is None:
                return None
            
            # Center on 64x64 canvas
            shape_centered = self._center_shape_on_canvas(shape_resized)
            
            # Apply center of mass alignment
            shape_aligned = self._align_center_of_mass(shape_centered)
            
            return shape_aligned
            
        except Exception as e:
            print(f"Error preprocessing shape: {e}")
            return None
    
    def _resize_shape_with_aspect_ratio(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Resize shape maintaining aspect ratio, max dimension = SHAPE_RESIZE_MAX."""
        height, width = image.shape
        
        if height == 0 or width == 0:
            return None
        
        # Calculate new dimensions
        if height > width:
            new_height = self.SHAPE_RESIZE_MAX
            new_width = max(1, int(self.SHAPE_RESIZE_MAX * width / height))
        else:
            new_width = self.SHAPE_RESIZE_MAX
            new_height = max(1, int(self.SHAPE_RESIZE_MAX * height / width))
        
        # Ensure dimensions don't exceed maximum
        new_width = min(new_width, self.SHAPE_RESIZE_MAX)
        new_height = min(new_height, self.SHAPE_RESIZE_MAX)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _center_shape_on_canvas(self, image: np.ndarray) -> np.ndarray:
        """Center shape on 64x64 canvas."""
        canvas = np.zeros((self.SHAPE_SIZE, self.SHAPE_SIZE), dtype=np.uint8)
        
        height, width = image.shape
        paste_x = (self.SHAPE_SIZE - width) // 2
        paste_y = (self.SHAPE_SIZE - height) // 2
        
        canvas[paste_y:paste_y+height, paste_x:paste_x+width] = image
        return canvas
    
    def _align_center_of_mass(self, image: np.ndarray) -> np.ndarray:
        """
        Align shape by shifting to center of mass.
        This ensures shapes are consistently positioned for better recognition.
        """
        cy, cx = self._get_center_of_mass(image)
        
        # Calculate shift needed to center the shape
        center_y = self.SHAPE_SIZE // 2
        center_x = self.SHAPE_SIZE // 2
        shift_y = center_y - cy
        shift_x = center_x - cx
        
        # Limit shift to prevent moving shape out of bounds
        max_shift = self.SHAPE_SIZE // 4
        shift_y = np.clip(shift_y, -max_shift, max_shift)
        shift_x = np.clip(shift_x, -max_shift, max_shift)
        
        # Apply shift
        return self._shift_image(image, shift_x, shift_y)
    
    def _get_center_of_mass(self, image: np.ndarray) -> Tuple[int, int]:
        """Calculate center of mass of white pixels in image."""
        if image.sum() == 0:
            # Return image center if no white pixels
            return image.shape[0] // 2, image.shape[1] // 2
        
        # Calculate moments
        m = cv2.moments(image)
        if m['m00'] == 0:
            return image.shape[0] // 2, image.shape[1] // 2
        
        cy = int(m['m01'] / m['m00'])
        cx = int(m['m10'] / m['m00'])
        return cy, cx
    
    def _shift_image(self, image: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
        """Shift image by (shift_x, shift_y) pixels."""
        rows, cols = image.shape
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (cols, rows), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=0)
        return shifted
    
    def resize_with_padding(self, image: np.ndarray, 
                           target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Resize image to target size with aspect ratio preserved and center padding.
        
        Args:
            image: Input grayscale image
            target_size: Target dimensions (height, width), default 64x64 for shapes
            
        Returns:
            Resized and padded image
        """
        height, width = image.shape
        target_h, target_w = target_size
        
        # Calculate scale with 80% of target size
        scale = min(target_w / width, target_h / height) * 0.8
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center the image
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        x_offset = (target_w - new_width) // 2
        y_offset = (target_h - new_height) // 2
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas