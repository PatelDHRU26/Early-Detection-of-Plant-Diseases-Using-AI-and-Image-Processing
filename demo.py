"""
Demo script for plant disease detection.
This script demonstrates the functionality without requiring a trained model.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# Import utility functions
from utils.image_processor import preprocess_image, validate_image, get_image_info
from utils.predictor import create_demo_prediction, analyze_prediction_confidence
from utils.disease_info import get_disease_info
from utils.pdf_generator import create_simple_report

def create_demo_image():
    """
    Create a demo image for testing.
    
    Returns:
        PIL.Image: Demo image
    """
    # Create a simple green leaf-like image
    img_array = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    img_array[:, :, 1] = np.random.randint(150, 255, (224, 224), dtype=np.uint8)  # More green
    
    # Add some texture to simulate leaf
    for i in range(50):
        x = np.random.randint(0, 224)
        y = np.random.randint(0, 224)
        img_array[x:x+10, y:y+10] = np.random.randint(50, 100, 3)
    
    return Image.fromarray(img_array)

def run_demo():
    """
    Run the demo application.
    """
    st.set_page_config(
        page_title="Plant Disease Detection - Demo",
        page_icon="🌱",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #2E8B57;
            text-align: center;
            margin-bottom: 2rem;
        }
        .demo-box {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #2E8B57;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection - Demo Mode</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-box">
    <h3>🎯 Demo Information</h3>
    <p>This is a demonstration of the plant disease detection system. Since no trained model is available, 
    the system will generate demo predictions to showcase the functionality.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo options
    st.markdown("### 🎮 Demo Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Option 1: Use Demo Image")
        if st.button("🖼️ Generate Demo Image", type="primary"):
            demo_img = create_demo_image()
            st.session_state.demo_image = demo_img
            st.session_state.use_demo_image = True
            st.success("Demo image generated!")
    
    with col2:
        st.markdown("#### Option 2: Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a plant leaf"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.use_demo_image = False
    
    # Process image and show results
    if hasattr(st.session_state, 'demo_image') or hasattr(st.session_state, 'uploaded_file'):
        st.markdown("### 📸 Image Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if hasattr(st.session_state, 'use_demo_image') and st.session_state.use_demo_image:
                st.image(st.session_state.demo_image, caption="Demo Image", use_column_width=True)
                image_info = {
                    'format': 'PNG',
                    'size': (224, 224),
                    'mode': 'RGB',
                    'file_size': 'Demo Image'
                }
            else:
                st.image(st.session_state.uploaded_file, caption="Uploaded Image", use_column_width=True)
                image_info = get_image_info(st.session_state.uploaded_file)
        
        with col2:
            st.markdown("#### 📋 Image Information")
            for key, value in image_info.items():
                st.write(f"**{key.title()}:** {value}")
        
        # Generate demo prediction
        if st.button("🔬 Analyze Image (Demo)", type="primary"):
            with st.spinner("🔬 Analyzing image (demo mode)..."):
                # Create demo prediction
                prediction_result = create_demo_prediction()
                
                # Get disease information
                disease_info = get_disease_info(prediction_result['disease_name'])
                
                # Store results
                st.session_state.prediction_result = prediction_result
                st.session_state.disease_info = disease_info
                
                st.success("Analysis completed!")
        
        # Display results if available
        if hasattr(st.session_state, 'prediction_result'):
            display_demo_results(st.session_state.prediction_result, st.session_state.disease_info)

def display_demo_results(prediction_result, disease_info):
    """
    Display demo prediction results.
    
    Args:
        prediction_result: Demo prediction results
        disease_info: Disease information
    """
    st.markdown("### 🔍 Analysis Results (Demo)")
    
    # Main prediction box
    st.markdown(f"""
    <div style="background-color: #f0fff0; padding: 1.5rem; border-radius: 10px; border: 2px solid #228B22; margin: 1rem 0;">
    <h3>🎯 Prediction: {prediction_result['disease_name']}</h3>
    <p><strong>Confidence:</strong> {prediction_result['confidence']:.2f}%</p>
    <p><strong>Plant Type:</strong> {prediction_result['plant_type']}</p>
    <p><strong>Demo Mode:</strong> ✅ This is a demo prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📋 Disease Information")
        st.write(f"**Description:** {disease_info['description']}")
        st.write(f"**Symptoms:** {disease_info['symptoms']}")
        
        # Confidence analysis
        confidence_analysis = analyze_prediction_confidence(prediction_result)
        st.markdown("#### 📊 Confidence Analysis")
        st.write(f"**Level:** {confidence_analysis['confidence_level']}")
        st.write(f"**Recommendation:** {confidence_analysis['recommendation']}")
        
        # Progress bar
        confidence = prediction_result['confidence']
        st.progress(confidence / 100)
        st.write(f"{confidence:.1f}% confident")
    
    with col2:
        st.markdown("#### 💡 Treatment Recommendations")
        st.write(f"**Immediate Actions:** {disease_info['immediate_actions']}")
        st.write(f"**Prevention:** {disease_info['prevention']}")
        st.write(f"**Treatment Time:** {disease_info['treatment_time']}")
        
        # Organic treatments
        st.markdown("#### 🌿 Organic Treatments")
        for treatment in disease_info.get('organic_treatments', []):
            st.write(f"• {treatment}")
        
        # Chemical treatments
        st.markdown("#### 🧪 Chemical Treatments")
        for treatment in disease_info.get('chemical_treatments', []):
            st.write(f"• {treatment}")
    
    # Generate simple report
    st.markdown("### 📄 Generate Report")
    if st.button("📄 Generate Simple Report", type="secondary"):
        report = create_simple_report(prediction_result, disease_info)
        
        st.markdown("#### 📋 Simple Report")
        st.text_area("Report Content", report, height=400)
        
        # Download button
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report,
            file_name=f"demo_report_{prediction_result['disease_name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )

def main():
    """
    Main function to run the demo.
    """
    run_demo()

if __name__ == "__main__":
    main() 