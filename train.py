#!/usr/bin/env python3
"""
Training script for Traffic Violation Detection System
Trains YOLOv8 model on custom dataset
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

def setup_training_environment():
    """Setup training environment and paths"""
    print("Setting up training environment...")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Check if dataset exists
    dataset_path = Path('data.yaml')
    if not dataset_path.exists():
        print(f"❌ Dataset configuration not found at {dataset_path}")
        return False
    
    print("✅ Training environment setup complete")
    return True

def train_model():
    """Train YOLOv8 model on custom dataset"""
    print("\n🚀 Starting YOLOv8 Training...")
    
    # Load dataset configuration
    dataset_config = 'data.yaml'
    
    # Initialize YOLOv8 model
    print("📦 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Start with nano model
    
    # Training configuration - Optimized for CPU speed
    training_config = {
        'data': dataset_config,
        'epochs': 25,          # Reduced from 100 for faster training
        'imgsz': 416,          # Smaller image size for faster processing
        'batch': 8,            # Smaller batch size for CPU
        'device': 'cpu',       # Use CPU since no GPU available
        'workers': 2,          # Reduced workers for stability
        'patience': 10,        # Early stopping
        'save': True,
        'save_period': 5,      # Save more frequently
        'cache': False,        # Disable caching to save memory
        'project': 'runs/train',
        'name': 'traffic_violations',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',  # Use AdamW for better CPU performance
        'lr0': 0.001,          # Lower learning rate for stability
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 2.0,  # Reduced warmup
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'verbose': True        # Show progress
    }
    
    print("🎯 Training Configuration (Optimized for Speed):")
    print(f"   Dataset: {dataset_config}")
    print(f"   Epochs: {training_config['epochs']} (reduced for faster training)")
    print(f"   Image Size: {training_config['imgsz']} (smaller for speed)")
    print(f"   Batch Size: {training_config['batch']} (optimized for CPU)")
    print(f"   Device: {training_config['device']}")
    print(f"   Estimated Time: ~15-30 minutes (vs 3-5 hours)")
    print("💡 Training will be much faster now!")
    
    try:
        # Start training
        print("\n🔥 Starting training...")
        results = model.train(**training_config)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {results.save_dir}")
        
        # Save the trained model to models directory
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, 'models/traffic_violations_best.pt')
            print(f"📦 Best model copied to: models/traffic_violations_best.pt")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def validate_model():
    """Validate the trained model"""
    print("\n🔍 Validating trained model...")
    
    model_path = 'models/traffic_violations_best.pt'
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        
        # Validate on validation set
        dataset_config = 'data.yaml'
        results = model.val(data=dataset_config)
        
        print("✅ Validation completed!")
        print(f"📊 mAP50: {results.box.map50:.3f}")
        print(f"📊 mAP50-95: {results.box.map:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False

def main():
    """Main training function"""
    print("=" * 60)
    print("🚦 TRAFFIC VIOLATION DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Setup environment
    if not setup_training_environment():
        print("❌ Failed to setup training environment")
        return
    
    # Train model
    if not train_model():
        print("❌ Training failed")
        return
    
    # Validate model
    if not validate_model():
        print("❌ Validation failed")
        return
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 Model saved to: models/traffic_violations_best.pt")
    print("🚀 You can now run the detection system!")
    print("=" * 60)

if __name__ == "__main__":
    main()
