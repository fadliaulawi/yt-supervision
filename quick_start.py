"""
Quick Start Example for Video Detector
This script demonstrates basic usage of the vehicle detection system.
"""

import sys
import os

def check_installation():
    """Check if all required packages are installed."""
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        import numpy as np
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def quick_test():
    """Quick test of the video detector."""
    try:
        from video_detector import VideoDetector
        
        print("🚀 Initializing Video Detector...")
        detector = VideoDetector(confidence_threshold=0.7)
        
        print("✅ Video Detector initialized successfully!")
        print("\nDetector Configuration:")
        print(f"  - Model: {detector.model_path}")
        print(f"  - Confidence threshold: {detector.confidence_threshold}")
        print(f"  - Vehicle classes: {list(detector.class_names.values())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n" + "="*60)
    print("🎯 USAGE EXAMPLES")
    print("="*60)
    
    print("\n📹 WEBCAM DETECTION:")
    print("   python demo.py --webcam")
    
    print("\n🎬 VIDEO FILE PROCESSING:")
    print("   python demo.py --video your_video.mp4")
    
    print("\n💾 SAVE OUTPUT VIDEO:")
    print("   python demo.py --video input.mp4 --output detected_output.mp4")
    
    print("\n⚙️  ADVANCED OPTIONS:")
    print("   python video_detector.py --source video.mp4 --confidence 0.7 --model yolov8l.pt")
    
    print("\n📊 BENCHMARK PERFORMANCE:")
    print("   python video_detector.py --source test_video.mp4 --benchmark")
    
    print("\n🔧 COMMAND LINE HELP:")
    print("   python video_detector.py --help")
    
    print("\n" + "="*60)
    print("💡 TIPS:")
    print("="*60)
    print("• Press 'q' to quit during video playback")
    print("• Press 's' to save a screenshot")
    print("• Use yolov8n.pt for speed, yolov8l.pt for accuracy")
    print("• Adjust confidence threshold based on your needs")
    print("• Check logs/video_detector.log for detailed logs")

def main():
    """Main function."""
    print("🚗 Vehicle Detection System - Quick Start")
    print("="*50)
    
    # Check installation
    if not check_installation():
        return
    
    # Quick test
    if not quick_test():
        return
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 System is ready! Choose an option above to get started.")

if __name__ == "__main__":
    main()
