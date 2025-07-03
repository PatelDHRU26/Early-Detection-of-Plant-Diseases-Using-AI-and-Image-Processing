"""
Streamlit-based Dataset Manager for Plant Disease Detection.
Provides a user-friendly web interface for downloading and managing the PlantVillage dataset.
"""

import streamlit as st
import os
import shutil
from pathlib import Path
import time

# Optional: Import kagglehub if available
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Dataset Manager - Plant Disease Detection",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .download-box {
        background-color: #f0fff0;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #228B22;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def check_dataset_structure():
    """Check if the dataset directory structure is correct."""
    dataset_path = Path("dataset/plantvillage")
    
    if not dataset_path.exists():
        return False
    
    # Check for some key directories
    expected_dirs = [
        "Apple___Apple_scab",
        "Apple___Black_rot", 
        "Tomato___Bacterial_spot",
        "Tomato___healthy",
        "Potato___Early_blight"
    ]
    
    for dir_name in expected_dirs:
        if not (dataset_path / dir_name).exists():
            return False
    
    return True

def count_images():
    """Count the total number of images in the dataset."""
    dataset_path = Path("dataset/plantvillage")
    
    if not dataset_path.exists():
        return 0
    
    count = 0
    for ext in ['*.JPG', '*.jpg', '*.png', '*.PNG']:
        count += len(list(dataset_path.rglob(ext)))
    
    return count

def get_dataset_info():
    """Get information about the current dataset."""
    dataset_path = Path("dataset/plantvillage")
    
    if not dataset_path.exists():
        return {
            'exists': False,
            'message': 'Dataset not found'
        }
    
    # Count images
    total_images = count_images()
    
    # Count directories (classes)
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    num_classes = len(classes)
    
    # Get class names
    class_names = [d.name for d in classes]
    
    return {
        'exists': True,
        'total_images': total_images,
        'num_classes': num_classes,
        'class_names': class_names,
        'message': f'Found {total_images} images across {num_classes} classes'
    }

def download_with_kagglehub():
    """Download the PlantVillage dataset using kagglehub."""
    if not KAGGLEHUB_AVAILABLE:
        st.error("❌ kagglehub is not installed. Please run: `pip install kagglehub`")
        return False
    
    try:
        with st.spinner("📥 Downloading PlantVillage dataset using kagglehub..."):
            path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
            st.info(f"Dataset downloaded to: {path}")
        
        # Move files to correct directory
        target_dir = "dataset/plantvillage"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        with st.spinner("📁 Organizing dataset files..."):
            for item in os.listdir(path):
                s = os.path.join(path, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    if not os.path.exists(d):
                        shutil.move(s, d)
                else:
                    shutil.move(s, d)
        
        st.success(f"✅ PlantVillage dataset is ready at: {target_dir}")
        return True
        
    except Exception as e:
        st.error(f"❌ Error downloading dataset: {str(e)}")
        return False

def create_sample_structure():
    """Create a sample directory structure for testing."""
    dataset_path = Path("dataset/plantvillage")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample directories
    sample_classes = [
        "Apple___Apple_scab",
        "Apple___healthy", 
        "Tomato___Bacterial_spot",
        "Tomato___healthy"
    ]
    
    for class_name in sample_classes:
        class_path = dataset_path / class_name
        class_path.mkdir(exist_ok=True)
        
        # Create a placeholder file
        placeholder_file = class_path / "README.txt"
        placeholder_file.write_text(f"Place {class_name} images in this directory")
    
    st.success("✅ Sample directory structure created")
    st.info("📁 Add your images to the respective directories")

def validate_dataset():
    """Validate the dataset structure and provide recommendations."""
    info = get_dataset_info()
    
    if not info['exists']:
        st.error("❌ Dataset not found")
        st.info("📥 Please download the PlantVillage dataset")
        return False
    
    st.success(f"✅ Dataset found: {info['message']}")
    
    # Check for minimum requirements
    if info['total_images'] < 100:
        st.warning("⚠️  Very few images found. Consider downloading the full dataset.")
        return False
    
    if info['num_classes'] < 10:
        st.warning("⚠️  Few classes found. Consider downloading the full dataset.")
        return False
    
    st.success("✅ Dataset validation passed!")
    return True

def main():
    """Main function for the Streamlit dataset manager."""
    st.markdown('<h1 class="main-header">📊 Dataset Manager</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Plant Disease Detection AI</h2>', unsafe_allow_html=True)
    
    # Check current dataset status
    info = get_dataset_info()
    
    # Display current status
    st.markdown("### 📊 Current Dataset Status")
    
    if info['exists']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Images", f"{info['total_images']:,}")
        
        with col2:
            st.metric("Classes", info['num_classes'])
        
        with col3:
            if info['total_images'] > 1000:
                status = "✅ Complete"
            elif info['total_images'] > 100:
                status = "⚠️ Partial"
            else:
                status = "❌ Incomplete"
            st.metric("Status", status)
        
        # Show class information
        if info['total_images'] > 0:
            st.markdown("### 📁 Found Classes")
            
            # Display classes in columns
            cols = st.columns(3)
            for i, class_name in enumerate(info['class_names']):
                col_idx = i % 3
                with cols[col_idx]:
                    st.write(f"• {class_name}")
        
        # Validate dataset
        st.markdown("### 🔍 Dataset Validation")
        validate_dataset()
        
    else:
        st.markdown("""
        <div class="status-box">
        <h3>❌ No Dataset Found</h3>
        <p>The PlantVillage dataset is not currently available. You can download it using one of the options below.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Download options
    st.markdown("### 📥 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="download-box">
        <h3>🚀 Automatic Download (Recommended)</h3>
        <p>Download the complete PlantVillage dataset automatically using kagglehub.</p>
        <ul>
        <li>Requires kagglehub installation</li>
        <li>Requires Kaggle API credentials</li>
        <li>Downloads ~1.5 GB of data</li>
        <li>Includes all 38 disease classes</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if KAGGLEHUB_AVAILABLE:
            if st.button("📥 Download with kagglehub", type="primary"):
                if download_with_kagglehub():
                    st.rerun()
        else:
            st.error("❌ kagglehub not available")
            st.info("Install with: `pip install kagglehub`")
    
    with col2:
        st.markdown("""
        <div class="download-box">
        <h3>📁 Manual Download</h3>
        <p>Download and extract the dataset manually.</p>
        <ol>
        <li>Visit: <a href="https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset" target="_blank">Kaggle Dataset</a></li>
        <li>Download the ZIP file</li>
        <li>Extract to dataset/plantvillage/</li>
        <li>Ensure proper directory structure</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📁 Create Sample Structure"):
            create_sample_structure()
            st.rerun()
    
    # Installation instructions
    if not KAGGLEHUB_AVAILABLE:
        st.markdown("### 🔧 Installation Instructions")
        st.markdown("""
        To enable automatic download, install kagglehub:
        
        ```bash
        pip install kagglehub
        ```
        
        Then set up your Kaggle API credentials:
        1. Go to https://www.kaggle.com/settings/account
        2. Create API token
        3. Download kaggle.json
        4. Place in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\\.kaggle\\ (Windows)
        """)
    
    # Next steps
    st.markdown("### 🚀 Next Steps")
    
    if info['exists'] and info['total_images'] > 1000:
        st.success("✅ Dataset is ready for training!")
        st.markdown("""
        You can now:
        1. **Train the model**: `python train_model.py`
        2. **Run the application**: `streamlit run app.py`
        3. **Test with demo**: `streamlit run demo.py`
        """)
    else:
        st.info("📋 After downloading the dataset:")
        st.markdown("""
        1. **Verify the structure** using this tool
        2. **Train the model**: `python train_model.py`
        3. **Run the application**: `streamlit run app.py`
        """)
    
    # Refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()

if __name__ == "__main__":
    main() 