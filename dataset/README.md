# 📊 Dataset Directory

This directory contains the training data for the plant disease detection model.

## 🌱 PlantVillage Dataset

The PlantVillage dataset is a large-scale dataset of plant disease images used for training our deep learning model.

### 📥 Download Instructions

#### Option 1: Kaggle Dataset (Recommended)
1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download the dataset ZIP file
3. Extract to `dataset/plantvillage/`
4. Ensure the structure matches the expected format

#### Option 2: Direct Download
- **Size**: ~1.5 GB
- **Images**: 54,305 images
- **Classes**: 38 disease categories
- **Format**: PNG images

### 📁 Expected Directory Structure

```
dataset/plantvillage/
├── Apple___Apple_scab/
│   ├── 0a5b9329-dbad-432d-ac58-d291718345d9___FREC_Scab 3335.JPG
│   ├── 0a5b9329-dbad-432d-ac58-d291718345d9___FREC_Scab 3336.JPG
│   └── ...
├── Apple___Black_rot/
│   ├── 0a5b9329-dbad-432d-ac58-d291718345d9___FREC_Scab 3335.JPG
│   └── ...
├── Apple___Cedar_apple_rust/
│   └── ...
├── Apple___healthy/
│   └── ...
├── Cherry___healthy/
│   └── ...
├── Cherry___Powdery_mildew/
│   └── ...
├── Corn___Cercospora_leaf_spot Gray_leaf_spot/
│   └── ...
├── Corn___Common_rust/
│   └── ...
├── Corn___healthy/
│   └── ...
├── Corn___Northern_Leaf_Blight/
│   └── ...
├── Grape___Black_rot/
│   └── ...
├── Grape___Esca_(Black_Measles)/
│   └── ...
├── Grape___healthy/
│   └── ...
├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
│   └── ...
├── Peach___Bacterial_spot/
│   └── ...
├── Peach___healthy/
│   └── ...
├── Pepper,_bell___Bacterial_spot/
│   └── ...
├── Pepper,_bell___healthy/
│   └── ...
├── Potato___Early_blight/
│   └── ...
├── Potato___healthy/
│   └── ...
├── Potato___Late_blight/
│   └── ...
├── Strawberry___healthy/
│   └── ...
├── Strawberry___Leaf_scorch/
│   └── ...
├── Tomato___Bacterial_spot/
│   └── ...
├── Tomato___Early_blight/
│   └── ...
├── Tomato___healthy/
│   └── ...
├── Tomato___Late_blight/
│   └── ...
├── Tomato___Leaf_Mold/
│   └── ...
├── Tomato___Septoria_leaf_spot/
│   └── ...
├── Tomato___Spider_mites Two-spotted_spider_mite/
│   └── ...
├── Tomato___Target_Spot/
│   └── ...
├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   └── ...
└── Tomato___Tomato_mosaic_virus/
    └── ...
```

### 🔍 Dataset Information

#### Supported Plants
- **Apple** (4 classes: Apple scab, Black rot, Cedar apple rust, Healthy)
- **Cherry** (2 classes: Healthy, Powdery mildew)
- **Corn/Maize** (4 classes: Cercospora leaf spot, Common rust, Healthy, Northern leaf blight)
- **Grape** (4 classes: Black rot, Esca, Healthy, Leaf blight)
- **Peach** (2 classes: Bacterial spot, Healthy)
- **Bell Pepper** (2 classes: Bacterial spot, Healthy)
- **Potato** (3 classes: Early blight, Healthy, Late blight)
- **Strawberry** (2 classes: Healthy, Leaf scorch)
- **Tomato** (10 classes: Bacterial spot, Early blight, Healthy, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus)

#### Image Specifications
- **Format**: PNG
- **Size**: Variable (typically 256x256 or larger)
- **Channels**: RGB (3 channels)
- **Quality**: High-quality images with clear disease symptoms

### 🛠️ Dataset Preparation

#### 1. Download and Extract
```bash
# Download from Kaggle
# Extract to dataset/plantvillage/
```

#### 2. Verify Structure
```bash
# Check if all directories exist
ls dataset/plantvillage/
```

#### 3. Count Images
```bash
# Count total images
find dataset/plantvillage/ -name "*.JPG" | wc -l
```

### 📊 Dataset Statistics

| Plant Type | Disease Classes | Total Images | Healthy Images | Diseased Images |
|------------|----------------|--------------|----------------|-----------------|
| Apple | 4 | ~1,800 | ~500 | ~1,300 |
| Cherry | 2 | ~1,000 | ~500 | ~500 |
| Corn | 4 | ~2,000 | ~500 | ~1,500 |
| Grape | 4 | ~1,500 | ~400 | ~1,100 |
| Peach | 2 | ~1,000 | ~500 | ~500 |
| Bell Pepper | 2 | ~1,000 | ~500 | ~500 |
| Potato | 3 | ~1,500 | ~500 | ~1,000 |
| Strawberry | 2 | ~1,000 | ~500 | ~500 |
| Tomato | 10 | ~10,000 | ~1,500 | ~8,500 |

**Total**: ~54,305 images across 38 classes

### 🔧 Training with Custom Dataset

If you want to use your own dataset:

1. **Create similar directory structure**
2. **Use consistent naming**: `Plant___Disease`
3. **Ensure image quality**: Clear, well-lit images
4. **Balance classes**: Similar number of images per class
5. **Update model configuration** in `utils/model_loader.py`

### 📝 Dataset Citation

If you use this dataset in your research, please cite:

```
PlantVillage Dataset
Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.
```

### 🚀 Next Steps

1. **Download the dataset** using the instructions above
2. **Verify the structure** matches the expected format
3. **Run training**: `python train_model.py`
4. **Test the model**: `streamlit run app.py`

### ❓ Troubleshooting

**Common Issues:**
- **Wrong directory structure**: Ensure folders are named exactly as shown
- **Missing images**: Check file extensions (.JPG, .jpg, .png)
- **Permission errors**: Ensure read access to all directories
- **Memory issues**: Consider using a subset for testing

**Need Help?**
- Check the main README.md for more information
- Review the SETUP_GUIDE.md for detailed instructions
- Verify your dataset structure matches the expected format 