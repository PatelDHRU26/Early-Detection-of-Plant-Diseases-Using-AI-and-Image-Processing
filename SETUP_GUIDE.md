# 🚀 Setup Guide - Plant Disease Detection AI

This guide will help you set up and run the Plant Disease Detection application.

## 📋 Prerequisites

- **Python 3.8 or higher**
- **pip package manager**
- **Git** (optional, for cloning)

## 🛠️ Installation Steps

### 1. Clone or Download the Project

**Option A: Using Git**
```bash
git clone <https://github.com/PatelDHRU26/Early-Detection-of-Plant-Diseases-Using-AI-and-Image-Processing.git>
cd plant-disease-detection
```

**Option B: Download ZIP**
- Download the project ZIP file
- Extract to your desired location
- Open terminal/command prompt in the project directory

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import streamlit, tensorflow, opencv-python; print('✅ All dependencies installed successfully!')"
```

## 🎮 Running the Application

### Option 1: Full Application (Requires Trained Model)

```bash
streamlit run app.py
```

**Note:** If you don't have a trained model, the application will create a demo model for testing.

### Option 2: Demo Mode (No Model Required)

```bash
streamlit run demo.py
```

This runs a simplified version that demonstrates the functionality with demo predictions.

### Option 3: Training Your Own Model

1. **Download the PlantVillage Dataset**
   - Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
   - Download and extract to `dataset/plantvillage/`

2. **Run Training Script**
   ```bash
   python train_model.py
   ```

3. **Use Your Trained Model**
   - The trained model will be saved in `models/plant_disease_model.h5`
   - Run the full application: `streamlit run app.py`

## 🌐 Accessing the Application

After running the application, open your web browser and navigate to:
- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501 (for access from other devices)

## 📁 Project Structure

```
plant-disease-detection/
├── app.py                 # Main Streamlit application
├── demo.py               # Demo version (no model required)
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── SETUP_GUIDE.md       # This setup guide
├── utils/               # Utility modules
│   ├── __init__.py
│   ├── model_loader.py   # Model loading and caching
│   ├── image_processor.py # Image preprocessing
│   ├── predictor.py      # Disease prediction logic
│   ├── disease_info.py   # Disease database and info
│   └── pdf_generator.py  # PDF report generation
├── models/              # Trained model files
└── reports/             # Generated PDF reports
```

## 🎯 Quick Start Guide

### 1. First Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo to test functionality
streamlit run demo.py
```

### 2. Using the Application
1. **Upload Image**: Click "Browse files" and select a plant leaf image
2. **Analyze**: Click "Analyze Image" to process the image
3. **View Results**: See disease prediction, confidence score, and recommendations
4. **Generate Report**: Click "Generate PDF Report" to download a detailed report

### 3. Supported Image Formats
- **PNG** (recommended)
- **JPG/JPEG**
- **Maximum file size**: 10MB

## 🔧 Configuration Options

### Customizing the Model
Edit `utils/model_loader.py`:
```python
# Change model path
MODEL_PATH = "models/your_custom_model.h5"

# Modify image size
IMG_SIZE = (224, 224)

# Update number of classes
NUM_CLASSES = 38
```

### Adding New Diseases
Edit `utils/disease_info.py`:
```python
# Add new disease to DISEASE_DATABASE
'New_Plant___New_Disease': {
    'description': 'Description of the disease',
    'symptoms': 'Common symptoms',
    'immediate_actions': 'What to do immediately',
    'prevention': 'How to prevent this disease',
    # ... other fields
}
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Reinstall dependencies
pip uninstall -r requirements.txt
pip install -r requirements.txt
```

**2. Model Not Found**
```
✅ Solution: The app will create a demo model automatically
```

**3. Port Already in Use**
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

**4. Memory Issues**
```bash
# Solution: Reduce batch size in train_model.py
BATCH_SIZE = 16  # Instead of 32
```

**5. PDF Generation Fails**
```bash
# Solution: Check write permissions
chmod 755 reports/  # Linux/macOS
```

### Performance Optimization

**For Better Performance:**
1. Use GPU if available (install tensorflow-gpu)
2. Reduce image size for faster processing
3. Use SSD storage for model loading
4. Close other applications to free memory

**For Development:**
```bash
# Run with debug mode
streamlit run app.py --logger.level debug

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

## 📊 Testing the Application

### Test Images
You can test with:
1. **Real plant images** from your garden
2. **Sample images** from PlantVillage dataset
3. **Demo images** generated by the demo mode

### Expected Results
- **Healthy plants**: Should show "Healthy" with high confidence
- **Diseased plants**: Should identify specific diseases
- **Unknown plants**: May show lower confidence or generic predictions

## 🔒 Security Considerations

1. **File Upload**: Images are processed locally, not uploaded to external servers
2. **Model Security**: Use only trusted model files
3. **Data Privacy**: No user data is stored or transmitted
4. **Network Access**: Application runs locally by default

## 📞 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the terminal
2. **Verify dependencies**: Ensure all packages are installed correctly
3. **Test with demo**: Try the demo mode first
4. **Check file permissions**: Ensure write access to reports/ directory

### Support Resources
- **Documentation**: README.md
- **Issues**: Create an issue on GitHub
- **Community**: Check discussion forums

## 🎉 Success!

Once you see the Streamlit interface with the plant disease detection application, you're ready to start analyzing plant images!

**Next Steps:**
1. Try uploading a plant image
2. Explore different pages (Home, Prediction, Report, About)
3. Generate your first PDF report
4. Consider training your own model for better accuracy

---

**Happy Plant Disease Detection! 🌱🔬** 
