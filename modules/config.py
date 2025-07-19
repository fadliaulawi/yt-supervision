"""
Unified Configuration
Complete configuration settings for the vehicle detection system including
both core detection settings and dashboard-specific configurations.
"""

class Config:
    """Unified configuration class for the entire system."""
    
    # ====== CORE SYSTEM CONFIGURATION ======
    
    # Model Configuration
    DEFAULT_MODEL = "models/yolo11l.pt"  # Primary: models/yolo11l.pt, Fallback: models/yolov8n.pt
    CONFIDENCE_THRESHOLD = 0.7    # Minimum confidence for detections (0.0 - 1.0)

    # Vehicle Classes (COCO dataset class IDs)
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    # Detection Colors (BGR format for OpenCV)
    DETECTION_COLORS = {
        'car': (0, 255, 0),        # Green
        'motorcycle': (255, 0, 0), # Blue  
        'bus': (0, 0, 255),        # Red
        'truck': (255, 0, 255),    # Magenta
        'default': (255, 255, 255) # White
    }

    # Display Settings
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2
    LABEL_PADDING = 10

    # Processing Settings
    MAX_PROCESSING_TIME = 0.1  # Maximum time per frame (seconds)
    FRAME_SKIP = 1             # Process every nth frame (1 = all frames)

    # Logging Configuration
    LOG_LEVEL = "INFO"         # DEBUG, INFO, WARNING, ERROR
    LOG_FILE = "logs/video_detector.log"
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB

    # Output Settings
    DEFAULT_OUTPUT_FPS = 30
    OUTPUT_CODEC = "mp4v"
    SAVE_SCREENSHOTS = True
    SCREENSHOT_QUALITY = 95

    # Statistics Settings
    SAVE_STATISTICS = True
    STATS_UPDATE_INTERVAL = 100  # frames

    # Performance Settings
    USE_GPU = True             # Use GPU acceleration if available
    BATCH_SIZE = 1             # Number of frames to process together
    NUM_THREADS = 4            # Number of threads for CPU processing

    # Video Input Settings
    WEBCAM_INDEX = 0           # Default webcam index
    BUFFER_SIZE = 1            # Video capture buffer size
    READ_TIMEOUT = 5           # Seconds to wait for frame

    # Alert Thresholds
    HIGH_DENSITY_THRESHOLD = 10  # Alert when more than X vehicles detected
    LOW_FPS_THRESHOLD = 15       # Alert when FPS drops below X

    # File Paths
    MODELS_DIR = "models"
    OUTPUTS_DIR = "outputs"
    LOGS_DIR = "logs"
    TEMP_DIR = "temp"
    
    # ====== DASHBOARD CONFIGURATION ======
    
    # Example YouTube URLs for testing
    DEMO_YOUTUBE_URLS = [
        "https://www.youtube.com/live/6QL0RHNtOlo",  # Live traffic stream placeholder
    ]

    # Model performance recommendations
    MODEL_RECOMMENDATIONS = {
        "real_time": "models/yolo11n.pt",    # Fastest, good for live streams
        "balanced": "models/yolo11s.pt",     # Good balance of speed/accuracy
        "accurate": "models/yolo11l.pt",     # High accuracy, slower
        "research": "models/yolo11x.pt",     # Maximum accuracy
    }

    # Quality settings by use case
    QUALITY_SETTINGS = {
        "speed_priority": "480p",
        "balanced": "720p",
        "accuracy_priority": "1080p",
        "maximum": "best"
    }

    # Confidence threshold recommendations
    CONFIDENCE_SETTINGS = {
        "permissive": 0.3,      # Catch everything, more false positives
        "balanced": 0.6,        # Good balance
        "strict": 0.7,          # High confidence only
        "very_strict": 0.9      # Very high confidence
    }

    # Dashboard themes
    DASHBOARD_THEMES = {
        "light": {
            "base": "light",
            "primaryColor": "#1f77b4",
            "backgroundColor": "#ffffff",
            "secondaryBackgroundColor": "#f0f2f6",
            "textColor": "#262730"
        },
        "dark": {
            "base": "dark", 
            "primaryColor": "#ff6b6b",
            "backgroundColor": "#0e1117",
            "secondaryBackgroundColor": "#262730",
            "textColor": "#ffffff"
        }
    }
    
    # Dashboard default settings
    DEFAULT_QUALITY = "720p"
    DEFAULT_THEME = "light"
    
    # ====== UTILITY METHODS ======
    
    @classmethod
    def get_model_by_performance(cls, performance_type="balanced"):
        """Get recommended model path by performance type."""
        return cls.MODEL_RECOMMENDATIONS.get(performance_type, cls.DEFAULT_MODEL)
    
    @classmethod
    def get_confidence_by_setting(cls, setting="balanced"):
        """Get confidence threshold by setting name."""
        return cls.CONFIDENCE_SETTINGS.get(setting, cls.CONFIDENCE_THRESHOLD)
    
    @classmethod
    def get_quality_by_priority(cls, priority="balanced"):
        """Get quality setting by priority."""
        return cls.QUALITY_SETTINGS.get(priority, cls.DEFAULT_QUALITY)
