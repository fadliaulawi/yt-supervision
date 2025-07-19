#!/usr/bin/env python3
"""
GPU Detection and YOLO Device Configuration Test
"""

import torch
from ultralytics import YOLO
import os
import sys

def check_gpu_availability():
    """Check if GPU is available and properly configured."""
    
    print("🔍 GPU Detection and Configuration Check")
    print("=" * 50)
    
    # Check PyTorch CUDA availability
    print(f"\n1. PyTorch Configuration:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("   No CUDA-capable GPU found")
    
    # Test YOLO with device detection
    print(f"\n2. YOLO Device Configuration:")
    try:
        # Create a simple YOLO model to test device assignment
        model = YOLO('models/yolo11n.pt' if os.path.exists('models/yolo11n.pt') else 'models/yolov8n.pt')
        
        # Check what device YOLO is using
        device = next(model.model.parameters()).device
        print(f"   YOLO model device: {device}")
        
        # Try to move model to GPU if available
        if torch.cuda.is_available():
            model.to('cuda')
            device = next(model.model.parameters()).device
            print(f"   After GPU assignment: {device}")
            
            # Test inference speed
            import numpy as np
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            print(f"\n3. Performance Test:")
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
            
            results = model(test_image, verbose=False)
            
            if torch.cuda.is_available():
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                print(f"   GPU inference time: {inference_time:.2f} ms")
            else:
                print("   Running on CPU")
                
    except Exception as e:
        print(f"   Error testing YOLO: {e}")
    
    # Check environment variables
    print(f"\n4. Environment Variables:")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    print("\n" + "=" * 50)
    
    # Recommendations
    print("💡 Recommendations:")
    if torch.cuda.is_available():
        print("   ✅ GPU is available and ready for use!")
        print("   💡 Your YOLO models should automatically use GPU")
        print("   💡 For maximum performance, use models/yolo11n.pt for speed")
        print("   💡 Use models/yolo11l.pt for accuracy (if GPU memory allows)")
    else:
        print("   ⚠️ No GPU detected - running on CPU")
        print("   💡 Consider using smaller models (yolo11n.pt) for better CPU performance")
        print("   💡 Install CUDA-enabled PyTorch for GPU acceleration")

if __name__ == "__main__":
    check_gpu_availability()
