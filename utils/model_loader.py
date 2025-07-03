"""
Model loader utility for plant disease detection.
Handles loading and caching of the trained CNN model.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import streamlit as st
from pathlib import Path

# Model configuration
MODEL_PATH = "models/plant_disease_model.h5"
IMG_SIZE = (224, 224)
NUM_CLASSES = 38

# Class names for plant diseases (PlantVillage dataset)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus'
]

@st.cache_resource
def load_model():
    """
    Load the trained CNN model with caching for better performance.
    
    Returns:
        keras.Model: Loaded model
    """
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.warning("Model file not found. Creating a demo model...")
            return create_demo_model()
        
        # Load the model
        model = keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Creating a demo model for testing...")
        return create_demo_model()

def create_demo_model():
    """
    Create a demo model for testing when the trained model is not available.
    
    Returns:
        keras.Model: Demo model
    """
    # Create a simple CNN model for demo purposes
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(*IMG_SIZE, 3)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_class_names():
    """
    Get the list of class names for disease classification.
    
    Returns:
        list: List of class names
    """
    return CLASS_NAMES

def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model information
    """
    return {
        'input_shape': (*IMG_SIZE, 3),
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'model_path': MODEL_PATH
    } 