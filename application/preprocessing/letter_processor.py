"""Letter image preprocessing module for handwritten letter recognition."""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional


class LetterProcessor:
    """Process images for handwritten letter recognition using CNN model."""
    
    # Preprocessing constants
    GAUSSIAN_KERNEL_SIZE = (5, 5)
    MORPH_KERNEL_SIZE = (3, 3)
    
    # Letter preprocessing constants
    LETTER_SIZE = 28
    LETTER_PADDING = 20
    
    # Contour filtering thresholds for letters
    MIN_HEIGHT = 20
    MIN_WIDTH = 10
    MIN_AREA = 200
    MIN_ASPECT_RATIO = 0.3
    MAX_ASPECT_RATIO = 5.0
    MAX_HEIGHT_RATIO = 0.9
    MAX_WIDTH_RATIO = 0.6
    MEDIAN_HEIGHT_FACTOR = 0.2
    ROW_THRESHOLD_FACTOR = 0.5
    
    def __init__(self):
        """Initialize LetterProcessor with empty preprocessing steps dictionary."""
        self.preprocessing_steps: Dict[str, np.ndarray] = {}
    
    def process_image(self, image_path: str, mode: str = 'letters') -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        """
        Process image through all preprocessing steps for letter recognition.
        
        Args:
            image_path: Path to the input image file
            mode: Recognition mode - only 'letters' is supported
            
        Returns:
            Tuple containing:
                - Dictionary of preprocessing steps for visualization
                - List of preprocessed 28x28 letter images
                
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
        
        # Step 4: Morphological operations for letters (CLOSE + OPEN)
        morph = self._apply_morphology(thresh)
        self.preprocessing_steps['5_morphology'] = morph
        
        # Step 5: Detect and filter letter contours
        valid_contours, median_height = self._detect_letters(morph, original, gray.shape)
        
        # Step 6: Sort letters by reading order
        sorted_contours = self._sort_boxes_by_row(valid_contours, median_height)
        
        # Step 7: Extract and preprocess individual letters
        letter_images = self._extract_letters(morph, sorted_contours)
        
        return self.preprocessing_steps, letter_images
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Cannot load image from: {image_path}")
        return original
    
    def _apply_threshold(self, blurred: np.ndarray) -> np.ndarray:
        """Apply OTSU thresholding for better letter extraction."""
        _, thresh = cv2.threshold(
            blurred, 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return thresh
    
    def _apply_morphology(self, thresh: np.ndarray) -> np.ndarray:
        """Apply morphological operations: CLOSE then OPEN for letters."""
        kernel = np.ones(self.MORPH_KERNEL_SIZE, np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        return morph
    
    def _detect_letters(self, morph: np.ndarray, original: np.ndarray, 
                       image_shape: Tuple[int, int]) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        Detect and filter valid letter contours.
        
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
            
            # Check if contour is a valid letter
            is_valid = self._is_valid_letter_contour(bbox, contour, median_height, 
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
    
    def _is_valid_letter_contour(self, bbox: Tuple[int, int, int, int], 
                                contour: np.ndarray, median_height: float,
                                image_height: int, image_width: int) -> bool:
        """Check if contour is a valid letter."""
        x, y, w, h = bbox
        
        if w == 0:
            return False
            
        aspect_ratio = h / float(w)
        area = cv2.contourArea(contour)
        
        # Letters can be wide (W, M) or narrow (I, l)
        return (h > self.MIN_HEIGHT and
                w > self.MIN_WIDTH and
                self.MIN_ASPECT_RATIO < aspect_ratio < self.MAX_ASPECT_RATIO and
                area > self.MIN_AREA and
                h < image_height * self.MAX_HEIGHT_RATIO and
                w < image_width * self.MAX_WIDTH_RATIO)
    
    def _extract_letters(self, morph: np.ndarray, 
                        contours: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Extract and preprocess individual letters from the image."""
        letter_images = []
        pil_image = Image.fromarray(morph)
        
        for i, (x, y, w, h) in enumerate(contours):
            # Crop with padding
            letter_pil = pil_image.crop((
                max(0, x - self.LETTER_PADDING), 
                max(0, y - self.LETTER_PADDING), 
                min(morph.shape[1], x + w + self.LETTER_PADDING), 
                min(morph.shape[0], y + h + self.LETTER_PADDING)
            ))
            
            # Preprocess to 28x28 format (giống MNIST/EMNIST)
            letter_processed = self._preprocess_letter(letter_pil)
            if letter_processed is not None:
                letter_images.append(letter_processed)
                self.preprocessing_steps[f'7_letter_{i}'] = letter_processed
        
        return letter_images
    
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
    
    def _preprocess_letter(self, letter_image: Image.Image) -> Optional[np.ndarray]:
        """
        Convert letter image to EMNIST format (28x28, white letter on black background).
        
        Args:
            letter_image: PIL Image of the letter
            
        Returns:
            28x28 numpy array ready for CNN model, or None if processing fails
        """
        try:
            letter_np = np.array(letter_image)
            
            # Crop to tight bounding box (find non-zero pixels)
            coords = cv2.findNonZero(letter_np)
            if coords is None:
                return None
            
            x, y, w, h = cv2.boundingRect(coords)
            
            # Add padding
            padding = self.LETTER_PADDING
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(letter_np.shape[1], x + w + padding)
            y2 = min(letter_np.shape[0], y + h + padding)
            
            # Crop
            cropped = letter_np[y1:y2, x1:x2]
            
            # Make square (quan trọng để giữ tỷ lệ đúng)
            h, w = cropped.shape
            size = max(h, w)
            # Padding with BLACK (0) vì letter là WHITE on BLACK
            square = np.zeros((size, size), dtype=np.uint8)
            
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
            
            # Resize to 28x28
            resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

            # Làm nét chữ đậm hơn bằng dilation nhẹ
            kernel_dilate = np.ones((2, 2), np.uint8)
            resized = cv2.dilate(resized, kernel_dilate, iterations=1)

            # Căn chỉnh tâm khối lượng để đồng nhất vị trí
            aligned = self._align_center_of_mass(resized)

            return aligned
            
        except Exception as e:
            print(f"Error preprocessing letter: {e}")
            return None
    
    def _align_center_of_mass(self, image: np.ndarray) -> np.ndarray:
        """
        Align letter by shifting to center of mass.
        """
        if image.sum() == 0:
            return image
        
        # Calculate center of mass
        cy, cx = self._get_center_of_mass(image)
        
        # Calculate shift needed to center the letter
        center_y = self.LETTER_SIZE // 2
        center_x = self.LETTER_SIZE // 2
        shift_y = center_y - cy
        shift_x = center_x - cx
        
        # Limit shift to prevent moving letter out of bounds
        max_shift = self.LETTER_SIZE // 4
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
