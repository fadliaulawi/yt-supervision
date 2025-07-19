#!/usr/bin/env python3
"""
Comprehensive System Check for YT-Supervision Vehicle Detection System

This script performs complete system validation before using demo.py or app.py:
- Hardware detection (GPU/CPU capabilities) 
- Dependency verification
- Model availability checks
- System configuration validation
- Performance benchmarks
- Usage recommendations
"""

import sys
import time
from pathlib import Path
from typing import Dict


def print_header(title: str, char: str = "=", width: int = 20) -> None:
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"🔍 {title}")
    print(f"{char * width}")


def print_status(message: str, status: str = "info") -> None:
    """Print a status message with appropriate emoji."""
    emoji_map = {
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
        "info": "ℹ️",
        "running": "🔄"
    }
    emoji = emoji_map.get(status, "•")
    print(f"   {emoji} {message}")


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    print_header("Python Version Check")
    
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == required_major and version.minor >= required_minor:
        print_status("Python version meets requirements", "success")
        return True
    else:
        print_status(f"Python {required_major}.{required_minor}+ required", "error")
        print_status("Please upgrade Python to continue", "error")
        return False


def check_dependencies() -> Dict[str, bool]:
    """Check if all required packages are installed.
    
    Note: Package names in pip often differ from import names:
    - pip install opencv-python → import cv2
    - pip install yt-dlp → import yt_dlp 
    - pip install pillow → import PIL
    """
    print_header("Dependency Check")
    
    # Core dependencies: {pip_package_name: (import_name, description)}
    core_deps = {
        "torch": ("torch", "PyTorch - Deep learning framework"),
        "ultralytics": ("ultralytics", "YOLOv11 models and utilities"), 
        "opencv-python": ("cv2", "OpenCV - Computer vision library"),
        "yt-dlp": ("yt_dlp", "YouTube video extraction"),
        "numpy": ("numpy", "Numerical computing")
    }
    
    # Dashboard dependencies 
    dashboard_deps = {
        "streamlit": ("streamlit", "Web dashboard framework"),
        "plotly": ("plotly", "Interactive visualizations"), 
        "pandas": ("pandas", "Data manipulation")
    }
    
    # Optional dependencies
    optional_deps = {
        "pillow": ("PIL", "Image processing (Pillow)"),
        "psutil": ("psutil", "System monitoring"), 
        "tqdm": ("tqdm", "Progress bars"),
        "requests": ("requests", "HTTP library for web requests"),
        "matplotlib": ("matplotlib", "Plotting and visualization")
    }
    
    results = {}
    
    def check_package_group(deps: Dict[str, tuple], group_name: str, required: bool = True):
        print(f"\n📦 {group_name} Dependencies:")
        group_results = {}
        
        for pip_name, (import_name, description) in deps.items():
            try:
                __import__(import_name)
                print_status(f"{pip_name} ({import_name}) - {description}", "success")
                group_results[pip_name] = True
            except ImportError:
                status = "error" if required else "warning"
                print_status(f"{pip_name} ({import_name}) - {description} (MISSING)", status)
                group_results[pip_name] = False
                
        return group_results
    
    # Check each group
    results.update(check_package_group(core_deps, "Core", required=True))
    results.update(check_package_group(dashboard_deps, "Dashboard", required=False))  
    results.update(check_package_group(optional_deps, "Optional", required=False))
    
    # Summary
    missing_core = [pkg for pkg, available in results.items() 
                   if pkg in core_deps and not available]
    missing_dashboard = [pkg for pkg, available in results.items()
                        if pkg in dashboard_deps and not available]
    
    if missing_core:
        print_status("Install missing core dependencies:", "error")
        print(f"     pip install {' '.join(missing_core)}")
        print(f"     # Or install all: pip install -r requirements.txt")
    
    missing_dashboard = [pkg for pkg, available in results.items()
                        if pkg in dashboard_deps and not available]
    
    if missing_dashboard:
        print_status("For dashboard functionality, install:", "info")
        print(f"     pip install {' '.join(missing_dashboard)}")
    
    if not missing_core and not missing_dashboard:
        print_status("All dependencies are available", "success")
    elif missing_dashboard and not missing_core:
        print_status("Dashboard dependencies missing - CLI will work", "warning")
    
    return results


def check_gpu_hardware() -> Dict[str, any]:
    """Check GPU availability and configuration."""
    print_header("Hardware Detection")
    
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "cuda_version": None,
        "pytorch_cuda": False
    }
    
    try:
        import torch
        
        print_status(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        gpu_info["pytorch_cuda"] = torch.cuda.is_available()
        print_status(f"PyTorch CUDA support: {gpu_info['pytorch_cuda']}")
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["count"] = torch.cuda.device_count()
            
            print_status(f"CUDA version: {gpu_info['cuda_version']}")
            print_status(f"GPU devices found: {gpu_info['count']}")
            
            # Get device information
            for i in range(gpu_info["count"]):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    "name": props.name,
                    "memory_gb": props.total_memory / 1024**3,
                    "compute_capability": f"{props.major}.{props.minor}"
                }
                gpu_info["devices"].append(device_info)
                
                print_status(f"GPU {i}: {device_info['name']} "
                           f"({device_info['memory_gb']:.1f}GB, "
                           f"Compute {device_info['compute_capability']})")
        else:
            print_status("No CUDA-capable GPU detected", "warning")
            print_status("System will run on CPU (slower processing)", "info")
            
    except ImportError:
        print_status("PyTorch not installed - cannot check GPU", "error")
    except Exception as e:
        print_status(f"Error checking GPU hardware: {e}", "error")
    
    return gpu_info


def check_models() -> Dict[str, bool]:
    """Check YOLO model availability.""" 
    print_header("Model Availability Check")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Expected models with descriptions
    expected_models = {
        "yolo11n.pt": "YOLOv11 Nano - Fastest, real-time processing",
        "yolo11l.pt": "YOLOv11 Large - Balanced accuracy/speed (default)",
        "yolo11x.pt": "YOLOv11 Extra Large - Highest accuracy"
    }
    
    model_status = {}
    
    print_status(f"Models directory: {models_dir.absolute()}")
    
    for model_file, description in expected_models.items():
        model_path = models_dir / model_file
        available = model_path.exists()
        model_status[model_file] = available
        
        if available:
            size_mb = model_path.stat().st_size / 1024**2
            print_status(f"{model_file} - {description} ({size_mb:.1f}MB)", "success")
        else:
            print_status(f"{model_file} - {description} (will download on first use)", "info")
    
    # Check for any existing models
    existing_models = list(models_dir.glob("*.pt"))
    if existing_models:
        print_status(f"Total models available: {len(existing_models)}")
    else:
        print_status("Models will be downloaded automatically on first use", "info")
    
    return model_status


def test_video_detector() -> bool:
    """Test VideoDetector initialization."""
    print_header("Video Detector Test")
    
    try:
        from modules.video_detector import VideoDetector
        
        print_status("Importing VideoDetector module...", "running")
        
        # Test initialization
        detector = VideoDetector(confidence_threshold=0.7)
        
        print_status("VideoDetector initialized successfully", "success")
        print_status(f"Model: {detector.model_path}")
        print_status(f"Confidence threshold: {detector.confidence_threshold}")
        print_status(f"Device: {detector.device}")
        
        # Show vehicle classes
        if hasattr(detector, 'class_names'):
            vehicle_classes = list(detector.class_names.values())
            print_status(f"Vehicle classes: {', '.join(vehicle_classes)}")
        
        return True
        
    except Exception as e:
        print_status(f"VideoDetector initialization failed: {e}", "error")
        print_status("Check dependencies and model files", "info")
        return False


def performance_benchmark(gpu_info: Dict[str, any]) -> Dict[str, float]:
    """Run a quick performance benchmark."""
    print_header("Performance Benchmark")
    
    benchmark_results = {}
    
    try:
        import torch
        import numpy as np
        from ultralytics import YOLO
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test with different models if available
        models_to_test = []
        if Path("models/yolo11n.pt").exists():
            models_to_test.append(("yolo11n.pt", "YOLOv11n"))
        elif Path("models/yolov8n.pt").exists():
            models_to_test.append(("yolov8n.pt", "YOLOv8n"))
        
        for model_file, model_name in models_to_test:
            print_status(f"Benchmarking {model_name}...", "running")
            
            try:
                model = YOLO(f"models/{model_file}")
                
                # Move to GPU if available
                if gpu_info["available"]:
                    model.to('cuda')
                    device_name = "GPU"
                else:
                    device_name = "CPU"
                
                # Warm-up run
                _ = model(test_image, verbose=False)
                
                # Benchmark runs
                num_runs = 5
                start_time = time.time()
                
                for _ in range(num_runs):
                    _ = model(test_image, verbose=False)
                
                end_time = time.time()
                avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
                fps = 1000 / avg_time
                
                benchmark_results[f"{model_name}_{device_name}"] = {
                    "avg_time_ms": avg_time,
                    "fps": fps
                }
                
                print_status(f"{model_name} on {device_name}: {avg_time:.1f}ms "
                           f"({fps:.1f} FPS)", "success")
                
            except Exception as e:
                print_status(f"Benchmark failed for {model_name}: {e}", "warning")
    
    except ImportError:
        print_status("Cannot run benchmark - missing dependencies", "warning")
    except Exception as e:
        print_status(f"Benchmark error: {e}", "error")
    
    return benchmark_results


def show_system_recommendations(gpu_info: Dict[str, any], deps: Dict[str, bool]) -> None:
    """Show personalized system recommendations."""
    print_header("System Recommendations", "🎯")
    
    # Hardware recommendations
    if gpu_info["available"]:
        print_status("Excellent! GPU acceleration is available", "success")
        print_status("Recommended models:")
        print("     • yolo11n.pt - Real-time processing (~25-35 FPS)")
        print("     • yolo11l.pt - Best balance (~15-25 FPS) [DEFAULT]") 
        print("     • yolo11x.pt - Highest accuracy (~5-12 FPS)")
    else:
        print_status("Running on CPU - consider GPU for better performance", "info")
        print_status("Recommended models for CPU:")
        print("     • yolo11n.pt - Best CPU performance (~8-15 FPS)")
        print("     • Lower video quality (480p-720p) for better speed")
    
    # Software recommendations
    missing_core = [pkg for pkg, available in deps.items() 
                   if pkg in ["torch", "ultralytics", "opencv-python", "yt-dlp", "numpy"] 
                   and not available]
    
    if missing_core:
        print_status("Install missing core dependencies:", "error")
        print(f"     pip install {' '.join(missing_core)}")
        print(f"     # Or install all: pip install -r requirements.txt")
    
    missing_dashboard = [pkg for pkg, available in deps.items()
                        if pkg in ["streamlit", "plotly", "pandas"] and not available]
    
    if missing_dashboard:
        print_status("For dashboard functionality, install:", "info")
        print(f"     pip install {' '.join(missing_dashboard)}")


def show_usage_guide() -> None:
    """Show comprehensive usage guide."""
    print_header("Usage Guide", "🚀")
    
    print("\n📱 DASHBOARD (Recommended):")
    print("   streamlit run app.py")
    print("   # Opens web interface at http://localhost:8501")
    
    print("\n💻 COMMAND LINE INTERFACE:")
    print("   # Webcam analysis")
    print("   python demo.py --webcam")
    print("   ")
    print("   # Video file processing") 
    print("   python demo.py --video your_video.mp4")
    print("   ")
    print("   # YouTube video analysis")
    print('   python demo.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"')
    
    print("\n⚙️ ADVANCED OPTIONS:")
    print("   # Custom model and confidence")
    print("   python demo.py --video file.mp4 --model yolo11l.pt --confidence 0.7")
    print("   ")
    print("   # YouTube with quality control")
    print('   python demo.py --youtube "URL" --quality 720p')
    
    print("\n🎯 MODEL SELECTION GUIDE:")
    print("   • yolo11n.pt - Speed priority (real-time capable)")
    print("   • yolo11l.pt - Balanced performance (recommended)")
    print("   • yolo11x.pt - Accuracy priority (research use)")
    
    print("\n💡 TIPS:")
    print("   • Press 'Q' in OpenCV window to stop analysis")
    print("   • Dashboard uses fire-and-forget processing")
    print("   • Check logs/video_detector.log for detailed logs")
    print("   • Results auto-save with timestamps")


def main():
    """Main system check function."""
    print("🚗 YT-Supervision Vehicle Detection System")
    print("🔍 Comprehensive System Check")
    print("=" * 60)
    print("This tool validates your system before using demo.py or app.py")
    
    # Collect check results for internal logic
    system_results = {
        "python_ok": False,
        "dependencies": {},
        "gpu_info": {},
        "detector_ok": False
    }
    
    # Run all checks
    system_results["python_ok"] = check_python_version()
    
    if system_results["python_ok"]:
        system_results["dependencies"] = check_dependencies()
        system_results["gpu_info"] = check_gpu_hardware()
        check_models()  # Just run the check, don't need to store results
        
        # Only test detector if core dependencies are available
        core_deps_ok = all(system_results["dependencies"].get(pkg, False) 
                          for pkg in ["torch", "ultralytics", "opencv-python", "yt-dlp"])
        
        if core_deps_ok:
            system_results["detector_ok"] = test_video_detector()
            performance_benchmark(system_results["gpu_info"])  # Just run benchmark, don't store
    
    # Show recommendations
    show_system_recommendations(system_results["gpu_info"], system_results["dependencies"])
    show_usage_guide()
    
    # Final status
    print_header("System Status Summary", "📋")
    
    core_deps_missing = sum(1 for pkg in ["torch", "ultralytics", "opencv-python", "yt-dlp", "numpy"] 
                           if not system_results["dependencies"].get(pkg, False))
    
    if not system_results["python_ok"]:
        print_status("System NOT READY - Python version too old", "error")
        ready = False
    elif core_deps_missing > 0:
        print_status(f"System NOT READY - {core_deps_missing} core dependencies missing", "error")
        ready = False
    elif not system_results["detector_ok"]:
        print_status("System NOT READY - VideoDetector failed to initialize", "error")
        ready = False
    else:
        print_status("System READY! 🎉", "success")
        
        if system_results["gpu_info"]["available"]:
            print_status("GPU acceleration enabled for optimal performance", "success")
        else:
            print_status("CPU processing mode (consider GPU for better speed)", "info")
        
        ready = True
    
    if ready:
        print("\n🚀 Ready to use:")
        print("   • Dashboard: streamlit run app.py") 
        print("   • Command line: python demo.py --help")
    else:
        print("\n🔧 Next steps:")
        print("   • Fix the issues above")
        print("   • Run this script again to verify")
    
    return ready


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
