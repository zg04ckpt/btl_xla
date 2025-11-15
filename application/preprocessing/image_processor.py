"""Image preprocessing module for handwritten digit recognition."""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional


class ImageProcessor:
    """Process images for handwritten digit recognition using CNN model."""
    
    # Preprocessing constants
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    MORPH_KERNEL_SIZE = (3, 3)
    MORPH_CLOSE_ITERATIONS = 2
    MORPH_OPEN_ITERATIONS = 1
    
    # Contour filtering thresholds
    MIN_HEIGHT = 10
    MIN_WIDTH = 3
    MIN_AREA = 50
    MIN_ASPECT_RATIO = 0.2
    MAX_ASPECT_RATIO = 8.0
    MAX_HEIGHT_RATIO = 0.9
    MAX_WIDTH_RATIO = 0.6
    MEDIAN_HEIGHT_FACTOR = 0.2
    
    # Digit preprocessing constants
    CROP_PADDING = 5
    BORDER_PADDING = 2
    MNIST_SIZE = 28
    RESIZE_MAX_DIM = 20
    ROW_THRESHOLD_FACTOR = 0.5
    
    def __init__(self):
        """Initialize ImageProcessor with empty preprocessing steps dictionary."""
        self.preprocessing_steps: Dict[str, np.ndarray] = {}
    
    def process_image(self, image_path: str) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Process image through all preprocessing steps for digit recognition.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            Tuple containing:
                - Dictionary of preprocessing steps for visualization
                - List of preprocessed digit images (28x28)
                
        Raises:
            ValueError: If image cannot be loaded
        """
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
        
        # Step 5: Detect and filter digit contours
        valid_contours, median_height = self._detect_digits(morph, original, gray.shape)
        
        # Step 6: Sort digits by reading order
        sorted_contours = self._sort_boxes_by_row(valid_contours, median_height)
        
        # Step 7: Extract and preprocess individual digits
        digit_images = self._extract_digits(morph, sorted_contours)
        
        return self.preprocessing_steps, digit_images
    
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
    
    def _detect_digits(self, morph: np.ndarray, original: np.ndarray, 
                       image_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        Detect and filter valid digit contours.
        
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
            if self._is_valid_digit_contour(bbox, contour, median_height, 
                                           image_height, image_width):
                x, y, w, h = bbox
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                valid_contours.append(bbox)
            else:
                x, y, w, h = bbox
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        
        self.preprocessing_steps['6_contours'] = contour_img
        return valid_contours, median_height
    
    def _is_valid_digit_contour(self, bbox: Tuple[int, int, int, int], 
                                contour: np.ndarray, median_height: float,
                                image_height: int, image_width: int) -> bool:
        """Check if contour is a valid digit based on size and aspect ratio."""
        x, y, w, h = bbox
        
        if w == 0:
            return False
            
        aspect_ratio = h / float(w)
        area = cv2.contourArea(contour)
        min_height = max(self.MIN_HEIGHT, median_height * self.MEDIAN_HEIGHT_FACTOR)
        
        return (h > min_height and
                self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO and
                area > self.MIN_AREA and
                w > self.MIN_WIDTH and
                h < image_height * self.MAX_HEIGHT_RATIO and
                w < image_width * self.MAX_WIDTH_RATIO)
    
    def _extract_digits(self, morph: np.ndarray, 
                       contours: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Extract and preprocess individual digits from the image."""
        digit_images = []
        pil_image = Image.fromarray(morph)
        
        for i, (x, y, w, h) in enumerate(contours):
            # Crop with padding
            digit_pil = pil_image.crop((
                max(0, x - self.CROP_PADDING), 
                max(0, y - self.CROP_PADDING), 
                min(morph.shape[1], x + w + self.CROP_PADDING), 
                min(morph.shape[0], y + h + self.CROP_PADDING)
            ))
            
            # Preprocess to MNIST format
            digit_processed = self._preprocess_digit(digit_pil)
            if digit_processed is not None:
                digit_images.append(digit_processed)
                self.preprocessing_steps[f'7_digit_{i}'] = digit_processed
        
        return digit_images
    
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
    
    def _preprocess_digit(self, digit_image: Image.Image) -> Optional[np.ndarray]:
        """
        Convert digit image to MNIST format (28x28, black background, white digit).
        
        Args:
            digit_image: PIL Image of the digit
            
        Returns:
            28x28 numpy array ready for CNN model, or None if processing fails
        """
        try:
            digit_np = np.array(digit_image)
            
            # Crop to tight bounding box
            digit_np = self._crop_to_content(digit_np)
            if digit_np is None:
                return None
            
            # Add border padding
            digit_np = cv2.copyMakeBorder(
                digit_np, 
                self.BORDER_PADDING, self.BORDER_PADDING, 
                self.BORDER_PADDING, self.BORDER_PADDING, 
                cv2.BORDER_CONSTANT, value=0
            )
            
            # Resize with aspect ratio preservation
            digit_resized = self._resize_with_aspect_ratio(digit_np)
            if digit_resized is None:
                return None
            
            # Center on 28x28 canvas
            return self._center_on_canvas(digit_resized)
            
        except Exception as e:
            print(f"Error preprocessing digit: {e}")
            return None
    
    def _crop_to_content(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Crop image to tight bounding box around non-zero pixels."""
        coords = cv2.findNonZero(image)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return image[y:y+h, x:x+w]
        return None
    
    def _resize_with_aspect_ratio(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Resize image maintaining aspect ratio, max dimension = RESIZE_MAX_DIM."""
        height, width = image.shape
        
        if height == 0 or width == 0:
            return None
        
        # Calculate new dimensions
        if height > width:
            new_height = self.RESIZE_MAX_DIM
            new_width = max(1, int(self.RESIZE_MAX_DIM * width / height))
        else:
            new_width = self.RESIZE_MAX_DIM
            new_height = max(1, int(self.RESIZE_MAX_DIM * height / width))
        
        # Ensure dimensions don't exceed maximum
        new_width = min(new_width, self.RESIZE_MAX_DIM)
        new_height = min(new_height, self.RESIZE_MAX_DIM)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _center_on_canvas(self, image: np.ndarray) -> np.ndarray:
        """Center image on 28x28 canvas."""
        canvas = np.zeros((self.MNIST_SIZE, self.MNIST_SIZE), dtype=np.uint8)
        
        height, width = image.shape
        paste_x = (self.MNIST_SIZE - width) // 2
        paste_y = (self.MNIST_SIZE - height) // 2
        
        canvas[paste_y:paste_y+height, paste_x:paste_x+width] = image
        return canvas
    
    def resize_with_padding(self, image: np.ndarray, 
                           target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """
        Resize image to target size with aspect ratio preserved and center padding.
        
        Args:
            image: Input grayscale image
            target_size: Target dimensions (height, width)
            
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