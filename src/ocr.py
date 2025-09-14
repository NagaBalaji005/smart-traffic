"""
OCR Module for Number Plate Recognition
"""

import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple, Optional
import re
from config import OCR_CONFIG
from utils import enhance_plate_image, validate_plate_format

class PlateOCR:
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        Initialize EasyOCR reader for number plate recognition
        Args:
            languages: List of languages to recognize
            gpu: Whether to use GPU for OCR
        """
        self.languages = languages or ['en']
        self.gpu = gpu
        self.confidence_threshold = OCR_CONFIG['confidence_threshold']
        self.preprocessing = OCR_CONFIG['preprocessing']
        self.upscale_factor = OCR_CONFIG['upscale_factor']
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            model_storage_directory='models/ocr',
            download_enabled=True
        )
        
        print(f"OCR initialized with languages: {self.languages}, GPU: {self.gpu}")
    
    def read_plate(self, plate_image: np.ndarray) -> Optional[Dict]:
        """
        Read number plate text from image
        Args:
            plate_image: Number plate image
        Returns:
            OCR result dictionary or None
        """
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Preprocess image if enabled
        if self.preprocessing:
            processed_image = self._preprocess_plate(plate_image)
        else:
            processed_image = plate_image
        
        # Upscale if image is too small
        if processed_image.shape[0] < 50 or processed_image.shape[1] < 100:
            processed_image = self._upscale_image(processed_image)
        
        try:
            # Run OCR
            results = self.reader.readtext(
                processed_image,
                detail=1,
                paragraph=False,
                height_ths=0.5,
                width_ths=0.5
            )
            
            if not results:
                return None
            
            # Process results
            best_result = self._get_best_result(results)
            
            if best_result:
                return {
                    'text': best_result['text'],
                    'confidence': best_result['confidence'],
                    'bbox': best_result['bbox'],
                    'is_valid': validate_plate_format(best_result['text'])
                }
            
        except Exception as e:
            print(f"OCR error: {e}")
            return None
        
        return None
    
    def _preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR
        Args:
            image: Input plate image
        Returns:
            Preprocessed image
        """
        # Use utility function for enhancement
        enhanced = enhance_plate_image(image)
        
        # Additional preprocessing steps
        # Resize to standard size
        height, width = enhanced.shape[:2]
        target_width = 300
        target_height = int(height * target_width / width)
        
        resized = cv2.resize(enhanced, (target_width, target_height))
        
        # Apply morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _upscale_image(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale image for better OCR
        Args:
            image: Input image
        Returns:
            Upscaled image
        """
        height, width = image.shape[:2]
        new_width = int(width * self.upscale_factor)
        new_height = int(height * self.upscale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def _get_best_result(self, results: List[Tuple]) -> Optional[Dict]:
        """
        Get best OCR result based on confidence and text quality
        Args:
            results: List of OCR results
        Returns:
            Best result dictionary
        """
        if not results:
            return None
        
        # Filter results by confidence
        high_conf_results = [
            result for result in results 
            if result[2] >= self.confidence_threshold
        ]
        
        if not high_conf_results:
            # If no high confidence results, use the best available
            high_conf_results = results
        
        # Sort by confidence
        high_conf_results.sort(key=lambda x: x[2], reverse=True)
        
        # Get the best result
        best_result = high_conf_results[0]
        
        # Clean up text
        text = self._clean_text(best_result[1])
        
        return {
            'text': text,
            'confidence': best_result[2],
            'bbox': best_result[0]
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up OCR text
        Args:
            text: Raw OCR text
        Returns:
            Cleaned text
        """
        # Remove extra spaces and convert to uppercase
        cleaned = re.sub(r'\s+', '', text.upper())
        
        # Remove non-alphanumeric characters except common plate characters
        cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
        
        return cleaned
    
    def read_multiple_plates(self, plate_images: List[np.ndarray]) -> List[Dict]:
        """
        Read multiple number plates
        Args:
            plate_images: List of plate images
        Returns:
            List of OCR results
        """
        results = []
        
        for i, plate_image in enumerate(plate_images):
            result = self.read_plate(plate_image)
            if result:
                result['image_index'] = i
                results.append(result)
        
        return results
    
    def validate_plate(self, plate_text: str) -> Dict:
        """
        Validate number plate format
        Args:
            plate_text: Plate text to validate
        Returns:
            Validation result
        """
        is_valid = validate_plate_format(plate_text)
        
        return {
            'text': plate_text,
            'is_valid': is_valid,
            'format_type': self._get_format_type(plate_text) if is_valid else 'unknown'
        }
    
    def _get_format_type(self, plate_text: str) -> str:
        """
        Determine plate format type
        Args:
            plate_text: Plate text
        Returns:
            Format type string
        """
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', plate_text):
            return 'standard_indian'
        elif re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$', plate_text):
            return 'short_indian'
        elif re.match(r'^\d{2}[A-Z]{2}\d{4}$', plate_text):
            return 'numeric_start'
        else:
            return 'custom'
    
    def get_ocr_statistics(self) -> Dict:
        """
        Get OCR statistics
        Returns:
            OCR statistics
        """
        return {
            'languages': self.languages,
            'gpu_enabled': self.gpu,
            'confidence_threshold': self.confidence_threshold,
            'preprocessing_enabled': self.preprocessing,
            'upscale_factor': self.upscale_factor
        }
