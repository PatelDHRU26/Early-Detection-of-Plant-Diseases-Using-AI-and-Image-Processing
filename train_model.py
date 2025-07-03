"""
Training script for plant disease detection model.
This script demonstrates how to train a CNN model on the PlantVillage dataset.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 38

def create_model():
    """
    Create a CNN model based on VGG16 with transfer learning.
    
    Returns:
        keras.Model: Compiled model
    """
    # Load pre-trained VGG16 model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_generators(data_dir):
    """
    Prepare data generators for training and validation.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        tuple: Training and validation generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def create_callbacks():
    """
    Create training callbacks.
    
    Returns:
        list: List of callbacks
    """
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        'models/plant_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, val_generator, class_names):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        val_generator: Validation data generator
        class_names: List of class names
    """
    # Get predictions
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main training function.
    """
    print("🌱 Plant Disease Detection Model Training")
    print("=" * 50)
    
    # Check if data directory exists
    data_dir = "dataset/plantvillage"
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        print("Please download the PlantVillage dataset and place it in the dataset/plantvillage directory.")
        print("Dataset structure should be:")
        print("dataset/plantvillage/")
        print("├── Apple___Apple_scab/")
        print("├── Apple___Black_rot/")
        print("├── Tomato___Bacterial_spot/")
        print("└── ...")
        return
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Create model
    print("🔧 Creating model...")
    model = create_model()
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Prepare data generators
    print("📊 Preparing data generators...")
    train_generator, val_generator = prepare_data_generators(data_dir)
    
    print(f"📈 Training samples: {train_generator.samples}")
    print(f"📊 Validation samples: {val_generator.samples}")
    print(f"🏷️ Number of classes: {len(train_generator.class_indices)}")
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Train model
    print("🚀 Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("📊 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("🔍 Evaluating model...")
    class_names = list(train_generator.class_indices.keys())
    evaluate_model(model, val_generator, class_names)
    
    # Save final model
    model.save('models/plant_disease_model_final.h5')
    print("✅ Training completed!")
    print("📁 Models saved in the models/ directory")

if __name__ == "__main__":
    main() 