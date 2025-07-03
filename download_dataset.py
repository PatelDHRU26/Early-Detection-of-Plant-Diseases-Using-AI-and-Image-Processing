"""
Dataset download and preparation script for plant disease detection.
This script helps users download and set up the PlantVillage dataset.
"""

import os
import zipfile
import requests
import shutil
from pathlib import Path
import streamlit as st

# Optional: Import kagglehub if available
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

def check_dataset_structure():
    """
    Check if the dataset directory structure is correct.
    
    Returns:
        bool: True if structure is correct, False otherwise
    """
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
    """
    Count the total number of images in the dataset.
    
    Returns:
        int: Total number of images
    """
    dataset_path = Path("dataset/plantvillage")
    
    if not dataset_path.exists():
        return 0
    
    count = 0
    for ext in ['*.JPG', '*.jpg', '*.png', '*.PNG']:
        count += len(list(dataset_path.rglob(ext)))
    
    return count

def get_dataset_info():
    """
    Get information about the current dataset.
    
    Returns:
        dict: Dataset information
    """
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

def create_sample_structure():
    """
    Create a sample directory structure for testing.
    """
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
    
    print("✅ Sample directory structure created")
    print("📁 Add your images to the respective directories")

def download_sample_images():
    """
    Download a small sample of images for testing.
    Note: This is a placeholder function - actual download would require
    proper dataset access or sample images.
    """
    print("📥 Sample image download not implemented")
    print("💡 Please download the full dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")

def download_with_kagglehub():
    """
    Download the PlantVillage dataset using kagglehub and move it to the correct directory.
    """
    if not KAGGLEHUB_AVAILABLE:
        print("❌ kagglehub is not installed. Please run: pip install kagglehub")
        return
    print("📥 Downloading PlantVillage dataset using kagglehub...")
    path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print("Path to dataset files:", path)
    target_dir = "dataset/plantvillage"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                shutil.move(s, d)
        else:
            shutil.move(s, d)
    print(f"✅ PlantVillage dataset is ready at: {target_dir}")

def validate_dataset():
    """
    Validate the dataset structure and provide recommendations.
    """
    info = get_dataset_info()
    
    if not info['exists']:
        print("❌ Dataset not found")
        print("📥 Please download the PlantVillage dataset")
        return False
    
    print(f"✅ Dataset found: {info['message']}")
    
    # Check for minimum requirements
    if info['total_images'] < 100:
        print("⚠️  Very few images found. Consider downloading the full dataset.")
        return False
    
    if info['num_classes'] < 10:
        print("⚠️  Few classes found. Consider downloading the full dataset.")
        return False
    
    print("✅ Dataset validation passed!")
    return True

def main():
    """
    Main function for dataset management.
    """
    print("🌱 Plant Disease Detection - Dataset Manager")
    print("=" * 50)
    
    # Check current status
    info = get_dataset_info()
    
    if info['exists']:
        print(f"📊 Current Dataset Status: {info['message']}")
        
        if info['total_images'] > 0:
            print("\n📁 Found Classes:")
            for i, class_name in enumerate(info['class_names'][:10], 1):
                print(f"   {i}. {class_name}")
            
            if len(info['class_names']) > 10:
                print(f"   ... and {len(info['class_names']) - 10} more classes")
        
        # Validate dataset
        print("\n🔍 Validating dataset...")
        validate_dataset()
        
    else:
        print("❌ No dataset found")
        print("\n📥 To add the dataset:")
        print("1. Download from Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("2. Extract to dataset/plantvillage/")
        print("3. Ensure proper directory structure")
        
        if KAGGLEHUB_AVAILABLE:
            response = input("\n🤔 Would you like to download the dataset automatically using kagglehub? (y/n): ")
            if response.lower() == 'y':
                download_with_kagglehub()
                print("\n🔍 Validating dataset after download...")
                validate_dataset()
        else:
            response = input("\n🤔 Would you like to create a sample directory structure? (y/n): ")
            if response.lower() == 'y':
                create_sample_structure()
    
    print("\n🚀 Next Steps:")
    print("1. Ensure dataset is properly structured")
    print("2. Run training: python train_model.py")
    print("3. Test the model: streamlit run app.py")

if __name__ == "__main__":
    main() 