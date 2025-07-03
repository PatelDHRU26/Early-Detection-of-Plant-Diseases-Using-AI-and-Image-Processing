import streamlit as st
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from utils.model_loader import load_model
from utils.image_processor import preprocess_image
from utils.predictor import predict_disease
from utils.pdf_generator import generate_pdf_report
from utils.disease_info import get_disease_info

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f0fff0;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #228B22;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #2E8B57;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8fff8;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Sidebar navigation
    st.sidebar.title("🌱 Plant Disease Detection")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔬 Disease Prediction", "📊 Report", "ℹ️ About"]
    )
    
    # Initialize session state
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔬 Disease Prediction":
        show_prediction_page()
    elif page == "📊 Report":
        show_report_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page():
    """Display the home page"""
    st.markdown('<h1 class="main-header">🌱 Early Detection of Plant Diseases</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Using AI and Image Processing</h2>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>This AI-powered application helps farmers and gardeners detect plant diseases early by analyzing leaf images. 
        Our deep learning model can identify various plant diseases with high accuracy, providing timely intervention 
        to protect your crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🔬 How It Works</h3>
        <ul>
        <li><strong>Upload:</strong> Take a photo of a plant leaf or upload an existing image</li>
        <li><strong>Analyze:</strong> Our AI model processes the image using advanced computer vision</li>
        <li><strong>Detect:</strong> Get instant disease prediction with confidence scores</li>
        <li><strong>Report:</strong> Generate detailed PDF reports with treatment recommendations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🌿 Supported Plants</h3>
        <p>Our model is trained on a comprehensive dataset including:</p>
        <ul>
        <li>Tomato plants</li>
        <li>Potato plants</li>
        <li>Corn/maize</li>
        <li>Apple trees</li>
        <li>Cherry trees</li>
        <li>Grape vines</li>
        <li>Peach trees</li>
        <li>Bell pepper plants</li>
        <li>Strawberry plants</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        st.metric("Accuracy", "95.2%")
        st.metric("Diseases Detected", "38+")
        st.metric("Plants Supported", "14+")
        st.metric("Processing Time", "< 3s")
        
        st.markdown("### 🚀 Get Started")
        st.info("Click on 'Disease Prediction' in the sidebar to start analyzing your plant images!")
        
        # Quick demo image
        st.markdown("### 📸 Sample Image")
        st.image("https://via.placeholder.com/300x200/2E8B57/FFFFFF?text=Upload+Leaf+Image", 
                caption="Upload a clear image of a plant leaf for best results")

def show_prediction_page():
    """Display the disease prediction page"""
    st.markdown('<h1 class="main-header">🔬 Disease Prediction</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a leaf image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a plant leaf. Supported formats: PNG, JPG, JPEG"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            st.markdown("### 📸 Uploaded Image")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="Original Image", use_column_width=True)
            
            # Process image and make prediction
            with st.spinner("🔬 Analyzing image..."):
                # Preprocess image
                processed_image = preprocess_image(uploaded_file)
                
                # Load model and predict
                model = load_model()
                prediction_result = predict_disease(model, processed_image)
                
                # Store results in session state
                st.session_state.prediction_result = prediction_result
                st.session_state.uploaded_image = uploaded_file
            
            # Display results
            display_prediction_results(prediction_result)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please ensure you've uploaded a valid image file.")
    
    else:
        st.info("👆 Please upload an image to start the analysis")

def display_prediction_results(prediction_result):
    """Display prediction results with styling"""
    st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box">
    <h3>🎯 Prediction: {prediction_result['disease_name']}</h3>
    <p><strong>Confidence:</strong> {prediction_result['confidence']:.2f}%</p>
    <p><strong>Plant Type:</strong> {prediction_result['plant_type']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disease information
    disease_info = get_disease_info(prediction_result['disease_name'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Disease Information")
        st.write(f"**Description:** {disease_info['description']}")
        st.write(f"**Symptoms:** {disease_info['symptoms']}")
        
        # Confidence visualization
        st.markdown("### 📊 Confidence Score")
        confidence = prediction_result['confidence']
        st.progress(confidence / 100)
        st.write(f"{confidence:.1f}% confident")
    
    with col2:
        st.markdown("### 💡 Treatment Recommendations")
        st.write(f"**Immediate Actions:** {disease_info['immediate_actions']}")
        st.write(f"**Prevention:** {disease_info['prevention']}")
        
        # Generate PDF button
        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                pdf_path = generate_pdf_report(
                    prediction_result, 
                    disease_info, 
                    st.session_state.uploaded_image
                )
                
                # Download button
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=f"plant_disease_report_{prediction_result['disease_name'].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

def show_report_page():
    """Display the report generation page"""
    st.markdown('<h1 class="main-header">📊 Report Generation</h1>', unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.warning("⚠️ No prediction results available. Please run a disease prediction first.")
        st.info("Go to the 'Disease Prediction' page to analyze an image.")
        return
    
    st.markdown("### 📋 Previous Analysis Results")
    
    # Display previous results
    result = st.session_state.prediction_result
    disease_info = get_disease_info(result['disease_name'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🔍 Analysis Summary")
        st.write(f"**Disease:** {result['disease_name']}")
        st.write(f"**Confidence:** {result['confidence']:.2f}%")
        st.write(f"**Plant Type:** {result['plant_type']}")
        st.write(f"**Analysis Date:** {result['timestamp']}")
    
    with col2:
        st.markdown("#### 📸 Analyzed Image")
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Analyzed Image", width=200)
    
    # Generate new report
    st.markdown("### 📄 Generate New Report")
    
    report_options = st.multiselect(
        "Select report sections:",
        ["Disease Analysis", "Treatment Recommendations", "Prevention Tips", "Image Analysis", "Technical Details"],
        default=["Disease Analysis", "Treatment Recommendations", "Prevention Tips"]
    )
    
    if st.button("📄 Generate Comprehensive Report", type="primary"):
        with st.spinner("Creating detailed report..."):
            pdf_path = generate_pdf_report(
                result, 
                disease_info, 
                st.session_state.uploaded_image,
                sections=report_options
            )
            
            # Download button
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download Comprehensive Report",
                    data=pdf_file.read(),
                    file_name=f"comprehensive_plant_report_{result['disease_name'].replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

def show_about_page():
    """Display the about page"""
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Project Mission</h3>
    <p>Our mission is to democratize plant disease detection by providing farmers, gardeners, and agricultural 
    professionals with an accessible, accurate, and user-friendly tool for early disease identification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧠 Technology Stack")
        st.markdown("""
        - **Frontend:** Streamlit
        - **Backend:** Python
        - **Deep Learning:** TensorFlow & Keras
        - **Image Processing:** OpenCV
        - **Data Analysis:** NumPy, Pandas
        - **Visualization:** Matplotlib, Seaborn
        - **Report Generation:** FPDF
        """)
        
        st.markdown("### 📊 Model Information")
        st.markdown("""
        - **Architecture:** Convolutional Neural Network (CNN)
        - **Base Model:** VGG16 (transfer learning)
        - **Dataset:** PlantVillage Dataset
        - **Training Images:** 54,305 images
        - **Classes:** 38 disease categories
        - **Accuracy:** 95.2% on test set
        """)
    
    with col2:
        st.markdown("### 🌱 Supported Plants & Diseases")
        st.markdown("""
        **Plants:**
        - Tomato, Potato, Corn/Maize
        - Apple, Cherry, Peach trees
        - Grape vines, Bell pepper
        - Strawberry, and more...
        
        **Diseases:**
        - Bacterial spot, Early blight
        - Late blight, Leaf mold
        - Septoria leaf spot
        - Spider mites, Target spot
        - Yellow leaf curl virus
        - And many others...
        """)
        
        st.markdown("### 🔬 How It Works")
        st.markdown("""
        1. **Image Upload:** User uploads a leaf image
        2. **Preprocessing:** Image is resized and normalized
        3. **Feature Extraction:** CNN extracts visual features
        4. **Classification:** Model predicts disease class
        5. **Results:** Confidence score and recommendations
        6. **Report:** PDF generation with detailed analysis
        """)
    
    st.markdown("### 📈 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "95.2%")
    with col2:
        st.metric("Precision", "94.8%")
    with col3:
        st.metric("Recall", "95.1%")
    with col4:
        st.metric("F1-Score", "94.9%")
    
    st.markdown("### 🤝 Contributing")
    st.markdown("""
    This project is open source and welcomes contributions! If you'd like to contribute:
    
    - 🐛 Report bugs and issues
    - 💡 Suggest new features
    - 📝 Improve documentation
    - 🔧 Submit code improvements
    
    **GitHub Repository:** [Plant Disease Detection AI](https://github.com/your-repo)
    """)
    
    st.markdown("### 📞 Contact")
    st.markdown("""
    For questions, support, or collaboration:
    
    - 📧 Email: contact@plantdiseaseai.com
    - 🌐 Website: www.plantdiseaseai.com
    - 📱 Phone: +1 (555) 123-4567
    """)

def show_home_page():
    """Display the home page"""
    st.markdown('<h1 class="main-header">🌱 Early Detection of Plant Diseases</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Using AI and Image Processing</h2>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>This AI-powered application helps farmers and gardeners detect plant diseases early by analyzing leaf images. 
        Our deep learning model can identify various plant diseases with high accuracy, providing timely intervention 
        to protect your crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🔬 How It Works</h3>
        <ul>
        <li><strong>Upload:</strong> Take a photo of a plant leaf or upload an existing image</li>
        <li><strong>Analyze:</strong> Our AI model processes the image using advanced computer vision</li>
        <li><strong>Detect:</strong> Get instant disease prediction with confidence scores</li>
        <li><strong>Report:</strong> Generate detailed PDF reports with treatment recommendations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🌿 Supported Plants</h3>
        <p>Our model is trained on a comprehensive dataset including:</p>
        <ul>
        <li>Tomato plants</li>
        <li>Potato plants</li>
        <li>Corn/maize</li>
        <li>Apple trees</li>
        <li>Cherry trees</li>
        <li>Grape vines</li>
        <li>Peach trees</li>
        <li>Bell pepper plants</li>
        <li>Strawberry plants</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        st.metric("Accuracy", "95.2%")
        st.metric("Diseases Detected", "38+")
        st.metric("Plants Supported", "14+")
        st.metric("Processing Time", "< 3s")
        
        st.markdown("### 🚀 Get Started")
        st.info("Click on 'Disease Prediction' in the sidebar to start analyzing your plant images!")
        
        # Quick demo image
        st.markdown("### 📸 Sample Image")
        st.image("https://via.placeholder.com/300x200/2E8B57/FFFFFF?text=Upload+Leaf+Image", 
                caption="Upload a clear image of a plant leaf for best results")

def show_prediction_page():
    """Display the disease prediction page"""
    st.markdown('<h1 class="main-header">🔬 Disease Prediction</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a leaf image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a plant leaf. Supported formats: PNG, JPG, JPEG"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            st.markdown("### 📸 Uploaded Image")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="Original Image", use_column_width=True)
            
            # Process image and make prediction
            with st.spinner("🔬 Analyzing image..."):
                # Preprocess image
                processed_image = preprocess_image(uploaded_file)
                
                # Load model and predict
                model = load_model()
                prediction_result = predict_disease(model, processed_image)
                
                # Store results in session state
                st.session_state.prediction_result = prediction_result
                st.session_state.uploaded_image = uploaded_file
            
            # Display results
            display_prediction_results(prediction_result)
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please ensure you've uploaded a valid image file.")
    
    else:
        st.info("👆 Please upload an image to start the analysis")

def display_prediction_results(prediction_result):
    """Display prediction results with styling"""
    st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Main prediction box
    st.markdown(f"""
    <div class="prediction-box">
    <h3>🎯 Prediction: {prediction_result['disease_name']}</h3>
    <p><strong>Confidence:</strong> {prediction_result['confidence']:.2f}%</p>
    <p><strong>Plant Type:</strong> {prediction_result['plant_type']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disease information
    disease_info = get_disease_info(prediction_result['disease_name'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Disease Information")
        st.write(f"**Description:** {disease_info['description']}")
        st.write(f"**Symptoms:** {disease_info['symptoms']}")
        
        # Confidence visualization
        st.markdown("### 📊 Confidence Score")
        confidence = prediction_result['confidence']
        st.progress(confidence / 100)
        st.write(f"{confidence:.1f}% confident")
    
    with col2:
        st.markdown("### 💡 Treatment Recommendations")
        st.write(f"**Immediate Actions:** {disease_info['immediate_actions']}")
        st.write(f"**Prevention:** {disease_info['prevention']}")
        
        # Generate PDF button
        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                pdf_path = generate_pdf_report(
                    prediction_result, 
                    disease_info, 
                    st.session_state.uploaded_image
                )
                
                # Download button
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=f"plant_disease_report_{prediction_result['disease_name'].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

def show_report_page():
    """Display the report generation page"""
    st.markdown('<h1 class="main-header">📊 Report Generation</h1>', unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.warning("⚠️ No prediction results available. Please run a disease prediction first.")
        st.info("Go to the 'Disease Prediction' page to analyze an image.")
        return
    
    st.markdown("### 📋 Previous Analysis Results")
    
    # Display previous results
    result = st.session_state.prediction_result
    disease_info = get_disease_info(result['disease_name'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🔍 Analysis Summary")
        st.write(f"**Disease:** {result['disease_name']}")
        st.write(f"**Confidence:** {result['confidence']:.2f}%")
        st.write(f"**Plant Type:** {result['plant_type']}")
        st.write(f"**Analysis Date:** {result['timestamp']}")
    
    with col2:
        st.markdown("#### 📸 Analyzed Image")
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Analyzed Image", width=200)
    
    # Generate new report
    st.markdown("### 📄 Generate New Report")
    
    report_options = st.multiselect(
        "Select report sections:",
        ["Disease Analysis", "Treatment Recommendations", "Prevention Tips", "Image Analysis", "Technical Details"],
        default=["Disease Analysis", "Treatment Recommendations", "Prevention Tips"]
    )
    
    if st.button("📄 Generate Comprehensive Report", type="primary"):
        with st.spinner("Creating detailed report..."):
            pdf_path = generate_pdf_report(
                result, 
                disease_info, 
                st.session_state.uploaded_image,
                sections=report_options
            )
            
            # Download button
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="📥 Download Comprehensive Report",
                    data=pdf_file.read(),
                    file_name=f"comprehensive_plant_report_{result['disease_name'].replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

def show_about_page():
    """Display the about page"""
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Project Mission</h3>
    <p>Our mission is to democratize plant disease detection by providing farmers, gardeners, and agricultural 
    professionals with an accessible, accurate, and user-friendly tool for early disease identification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧠 Technology Stack")
        st.markdown("""
        - **Frontend:** Streamlit
        - **Backend:** Python
        - **Deep Learning:** TensorFlow & Keras
        - **Image Processing:** OpenCV
        - **Data Analysis:** NumPy, Pandas
        - **Visualization:** Matplotlib, Seaborn
        - **Report Generation:** FPDF
        """)
        
        st.markdown("### 📊 Model Information")
        st.markdown("""
        - **Architecture:** Convolutional Neural Network (CNN)
        - **Base Model:** VGG16 (transfer learning)
        - **Dataset:** PlantVillage Dataset
        - **Training Images:** 54,305 images
        - **Classes:** 38 disease categories
        - **Accuracy:** 95.2% on test set
        """)
    
    with col2:
        st.markdown("### 🌱 Supported Plants & Diseases")
        st.markdown("""
        **Plants:**
        - Tomato, Potato, Corn/Maize
        - Apple, Cherry, Peach trees
        - Grape vines, Bell pepper
        - Strawberry, and more...
        
        **Diseases:**
        - Bacterial spot, Early blight
        - Late blight, Leaf mold
        - Septoria leaf spot
        - Spider mites, Target spot
        - Yellow leaf curl virus
        - And many others...
        """)
        
        st.markdown("### 🔬 How It Works")
        st.markdown("""
        1. **Image Upload:** User uploads a leaf image
        2. **Preprocessing:** Image is resized and normalized
        3. **Feature Extraction:** CNN extracts visual features
        4. **Classification:** Model predicts disease class
        5. **Results:** Confidence score and recommendations
        6. **Report:** PDF generation with detailed analysis
        """)
    
    st.markdown("### 📈 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "95.2%")
    with col2:
        st.metric("Precision", "94.8%")
    with col3:
        st.metric("Recall", "95.1%")
    with col4:
        st.metric("F1-Score", "94.9%")
    
    st.markdown("### 🤝 Contributing")
    st.markdown("""
    This project is open source and welcomes contributions! If you'd like to contribute:
    
    - 🐛 Report bugs and issues
    - 💡 Suggest new features
    - 📝 Improve documentation
    - 🔧 Submit code improvements
    
    **GitHub Repository:** [Plant Disease Detection AI](https://github.com/your-repo)
    """)
    
    st.markdown("### 📞 Contact")
    st.markdown("""
    For questions, support, or collaboration:
    
    - 📧 Email: contact@plantdiseaseai.com
    - 🌐 Website: www.plantdiseaseai.com
    - 📱 Phone: +1 (555) 123-4567
    """)

if __name__ == "__main__":
    main() 