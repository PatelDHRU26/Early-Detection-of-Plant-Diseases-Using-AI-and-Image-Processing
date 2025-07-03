🌱 PlantVillage Dataset Directory

This directory should contain the PlantVillage dataset with the following structure:

📁 Expected Directory Structure:
├── Apple___Apple_scab/
│   ├── image1.JPG
│   ├── image2.JPG
│   └── ...
├── Apple___Black_rot/
│   ├── image1.JPG
│   └── ...
├── Tomato___Bacterial_spot/
│   ├── image1.JPG
│   └── ...
├── Tomato___healthy/
│   ├── image1.JPG
│   └── ...
└── ... (38 total classes)

📥 How to Add the Dataset:

1. Download from Kaggle:
   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

2. Extract the ZIP file contents to this directory

3. Ensure each class has its own folder named exactly as shown above

4. Verify the structure by running:
   python download_dataset.py

📊 Dataset Information:
- Total Images: ~54,305
- Classes: 38 disease categories
- Plants: 14 different plant types
- Format: PNG/JPG images

🔍 Current Status:
- Sample directories created for demonstration
- Add actual images to each directory
- Run validation to check completeness

🚀 Next Steps:
1. Add images to each directory
2. Run: python download_dataset.py (to validate)
3. Run: python train_model.py (to train model)
4. Run: streamlit run app.py (to test application) 