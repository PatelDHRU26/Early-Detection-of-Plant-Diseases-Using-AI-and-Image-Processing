"""
Image processing utility for plant disease detection.
Handles image preprocessing, resizing, and normalization.
"""

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import io

# Image processing configuration
TARGET_SIZE = (224, 224)
NORMALIZATION_FACTOR = 255.0

def preprocess_image(image_file):
    """
    Preprocess uploaded image for model prediction.
    
    Args:
        image_file: Uploaded file object from Streamlit
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    try:
        # Read image using PIL
        image = Image.open(image_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(TARGET_SIZE)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values
        image_array = image_array.astype(np.float32) / NORMALIZATION_FACTOR
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        raise

def enhance_image(image_array):
    """
    Apply image enhancement techniques for better analysis.
    
    Args:
        image_array: Input image array
        
    Returns:
        numpy.ndarray: Enhanced image array
    """
    try:
        # Convert back to uint8 for OpenCV operations
        image_uint8 = (image_array * 255).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to float32 and normalize
        enhanced = enhanced.astype(np.float32) / NORMALIZATION_FACTOR
        
        return enhanced
        
    except Exception as e:
        st.warning(f"Image enhancement failed: {str(e)}")
        return image_array

def extract_features(image_array):
    """
    Extract basic features from the image for analysis.
    
    Args:
        image_array: Input image array
        
    Returns:
        dict: Dictionary containing extracted features
    """
    try:
        # Convert to uint8 for feature extraction
        image_uint8 = (image_array * 255).astype(np.uint8)
        
        # Convert to grayscale for some features
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        
        features = {
            'mean_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'contrast': np.max(gray) - np.min(gray),
            'brightness': np.mean(image_uint8),
            'saturation': np.mean(cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)[:, :, 1])
        }
        
        return features
        
    except Exception as e:
        st.warning(f"Feature extraction failed: {str(e)}")
        return {}

def validate_image(image_file):
    """
    Validate uploaded image file.
    
    Args:
        image_file: Uploaded file object
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        # Check file size (max 10MB)
        if image_file.size > 10 * 1024 * 1024:
            st.error("Image file too large. Please upload an image smaller than 10MB.")
            return False
        
        # Check file format
        allowed_formats = ['png', 'jpg', 'jpeg']
        file_extension = image_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_formats:
            st.error(f"Unsupported file format. Please upload: {', '.join(allowed_formats)}")
            return False
        
        # Try to open image
        image = Image.open(image_file)
        image.verify()
        
        return True
        
    except Exception as e:
        st.error(f"Invalid image file: {str(e)}")
        return False

def create_image_preview(image_file, size=(300, 300)):
    """
    Create a preview of the uploaded image.
    
    Args:
        image_file: Uploaded file object
        size: Tuple of (width, height) for preview size
        
    Returns:
        PIL.Image: Preview image
    """
    try:
        image = Image.open(image_file)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
        
    except Exception as e:
        st.error(f"Error creating preview: {str(e)}")
        return None

def get_image_info(image_file):
    """
    Get information about the uploaded image.
    
    Args:
        image_file: Uploaded file object
        
    Returns:
        dict: Image information
    """
    try:
        image = Image.open(image_file)
        
        info = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height,
            'file_size': f"{image_file.size / 1024:.1f} KB"
        }
        
        return info
        
    except Exception as e:
        st.error(f"Error getting image info: {str(e)}")
        return {} 