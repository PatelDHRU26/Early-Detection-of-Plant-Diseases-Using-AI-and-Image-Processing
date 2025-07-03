"""
Prediction utility for plant disease detection.
Handles model inference and result processing.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from datetime import datetime
from .model_loader import get_class_names

def predict_disease(model, preprocessed_image):
    """
    Predict plant disease from preprocessed image.
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array
        
    Returns:
        dict: Prediction results with disease info and confidence
    """
    try:
        # Make prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Get class names
        class_names = get_class_names()
        predicted_class = class_names[predicted_class_idx]
        
        # Parse disease and plant information
        disease_info = parse_disease_info(predicted_class)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            top_3_predictions.append({
                'disease': class_names[idx],
                'confidence': float(predictions[0][idx] * 100),
                'plant_type': parse_disease_info(class_names[idx])['plant_type']
            })
        
        # Create result dictionary
        result = {
            'disease_name': disease_info['disease_name'],
            'plant_type': disease_info['plant_type'],
            'confidence': confidence,
            'predicted_class': predicted_class,
            'class_index': int(predicted_class_idx),
            'all_predictions': predictions[0].tolist(),
            'top_3_predictions': top_3_predictions,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_confidence': 'high' if confidence > 80 else 'medium' if confidence > 60 else 'low'
        }
        
        return result
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return create_demo_prediction()

def parse_disease_info(class_name):
    """
    Parse disease and plant information from class name.
    
    Args:
        class_name: Class name from model output
        
    Returns:
        dict: Parsed disease and plant information
    """
    try:
        # Split class name by '___'
        parts = class_name.split('___')
        
        if len(parts) >= 2:
            plant_type = parts[0].strip()
            disease_name = parts[1].strip()
            
            # Clean up disease name
            disease_name = disease_name.replace('_', ' ').title()
            
            # Handle special cases
            if disease_name.lower() == 'healthy':
                disease_name = 'Healthy'
            elif 'bacterial' in disease_name.lower():
                disease_name = disease_name.replace('Bacterial', 'Bacterial ')
            elif 'leaf' in disease_name.lower():
                disease_name = disease_name.replace('Leaf', 'Leaf ')
            
            return {
                'plant_type': plant_type.replace('_', ' ').title(),
                'disease_name': disease_name
            }
        else:
            return {
                'plant_type': 'Unknown',
                'disease_name': class_name.replace('_', ' ').title()
            }
            
    except Exception as e:
        return {
            'plant_type': 'Unknown',
            'disease_name': class_name.replace('_', ' ').title()
        }

def create_demo_prediction():
    """
    Create a demo prediction for testing purposes.
    
    Returns:
        dict: Demo prediction result
    """
    demo_diseases = [
        'Tomato___Bacterial_spot',
        'Potato___Early_blight',
        'Apple___Apple_scab',
        'Corn___Common_rust',
        'Grape___Black_rot'
    ]
    
    import random
    demo_class = random.choice(demo_diseases)
    demo_confidence = random.uniform(75, 95)
    
    disease_info = parse_disease_info(demo_class)
    
    return {
        'disease_name': disease_info['disease_name'],
        'plant_type': disease_info['plant_type'],
        'confidence': demo_confidence,
        'predicted_class': demo_class,
        'class_index': 0,
        'all_predictions': [0.1] * 38,  # Demo predictions
        'top_3_predictions': [
            {'disease': demo_class, 'confidence': demo_confidence, 'plant_type': disease_info['plant_type']},
            {'disease': 'Tomato___healthy', 'confidence': 15.0, 'plant_type': 'Tomato'},
            {'disease': 'Potato___healthy', 'confidence': 10.0, 'plant_type': 'Potato'}
        ],
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_confidence': 'high' if demo_confidence > 80 else 'medium',
        'is_demo': True
    }

def analyze_prediction_confidence(prediction_result):
    """
    Analyze the confidence level of the prediction.
    
    Args:
        prediction_result: Prediction result dictionary
        
    Returns:
        dict: Confidence analysis
    """
    confidence = prediction_result['confidence']
    
    if confidence >= 90:
        confidence_level = "Very High"
        recommendation = "High confidence prediction. Recommended actions should be followed."
    elif confidence >= 80:
        confidence_level = "High"
        recommendation = "Good confidence prediction. Consider additional verification."
    elif confidence >= 70:
        confidence_level = "Medium"
        recommendation = "Moderate confidence. Consider consulting an expert."
    elif confidence >= 60:
        confidence_level = "Low"
        recommendation = "Low confidence. Please upload a clearer image or consult an expert."
    else:
        confidence_level = "Very Low"
        recommendation = "Very low confidence. Image quality may be poor or disease not recognized."
    
    return {
        'confidence_level': confidence_level,
        'recommendation': recommendation,
        'requires_expert': confidence < 70
    }

def get_prediction_summary(prediction_result):
    """
    Generate a summary of the prediction results.
    
    Args:
        prediction_result: Prediction result dictionary
        
    Returns:
        str: Summary text
    """
    disease_name = prediction_result['disease_name']
    plant_type = prediction_result['plant_type']
    confidence = prediction_result['confidence']
    
    if disease_name.lower() == 'healthy':
        summary = f"The {plant_type} plant appears to be healthy with {confidence:.1f}% confidence."
    else:
        summary = f"Detected {disease_name} in {plant_type} with {confidence:.1f}% confidence."
    
    return summary

def validate_prediction(prediction_result):
    """
    Validate prediction results for consistency.
    
    Args:
        prediction_result: Prediction result dictionary
        
    Returns:
        bool: True if prediction is valid, False otherwise
    """
    try:
        required_keys = [
            'disease_name', 'plant_type', 'confidence', 
            'predicted_class', 'timestamp'
        ]
        
        for key in required_keys:
            if key not in prediction_result:
                return False
        
        # Check confidence range
        if not (0 <= prediction_result['confidence'] <= 100):
            return False
        
        # Check timestamp format
        datetime.strptime(prediction_result['timestamp'], "%Y-%m-%d %H:%M:%S")
        
        return True
        
    except Exception:
        return False 