"""
Dashboard Utilities
Helper functions and utilities for the dashboard.
"""

import sys
from pathlib import Path

class DashboardUtils:
    """Utility functions for the dashboard."""
    
    @staticmethod
    def check_dependencies():
        """Check if required dashboard dependencies are installed."""
        required_packages = ['streamlit', 'plotly', 'pandas']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("❌ Missing required packages:", ", ".join(missing_packages))
            print("📦 Install with: pip install streamlit plotly pandas")
            return False, missing_packages
        
        return True, []
    
    @staticmethod
    def get_model_recommendations():
        """Get model recommendations based on use case."""
        return {
            "real_time": {
                "model": "models/yolo11n.pt",
                "description": "Fastest processing, good for live streams",
                "use_case": "Webcam, live streaming"
            },
            "balanced": {
                "model": "models/yolo11l.pt", 
                "description": "Best balance of speed and accuracy",
                "use_case": "Most video analysis tasks"
            },
            "accuracy": {
                "model": "models/yolo11x.pt",
                "description": "Highest accuracy, slower processing", 
                "use_case": "Research, detailed analysis"
            }
        }
    
    @staticmethod
    def format_duration(seconds):
        """Format duration in seconds to human readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def get_demo_urls():
        """Get list of demo YouTube URLs for testing."""
        from .config import Config
        return Config.DEMO_YOUTUBE_URLS
    
    @staticmethod
    def validate_model_path(model_path):
        """Validate if model file exists."""
        if not Path(model_path).exists():
            print(f"⚠️ Model not found: {model_path}")
            print("📥 Model will be downloaded automatically on first use")
            return False
        return True
    
    @staticmethod
    def get_available_models():
        """Get list of available model files."""
        models_dir = Path("models")
        if not models_dir.exists():
            return []
        
        model_files = list(models_dir.glob("*.pt"))
        return [str(model) for model in model_files]
    
    @staticmethod  
    def print_dashboard_info():
        """Print dashboard information and usage."""
        print("🚗 Vehicle Detection Dashboard")
        print("=" * 40)
        print("📺 Supports YouTube videos/streams")
        print("📁 Supports video file upload")
        print("🎥 Supports webcam detection")
        print("📊 Real-time interactive charts")
        print("🎛️ Configurable models and settings")
        print()
        print("🚀 Launch: streamlit run app.py")
        print("🌐 URL: http://localhost:8501")
        print()
        
    @staticmethod
    def get_system_info():
        """Get system information for debugging."""
        import platform
        import cv2
        
        info = {
            "platform": platform.system(),
            "python_version": sys.version,
            "opencv_version": cv2.__version__,
        }
        
        # Try to get GPU info
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch_version"] = "Not installed"
            info["cuda_available"] = False
        
        return info
