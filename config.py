"""
Configuration file for the video detector system.
Modify these settings to customize detection behavior.
"""

# Model Configuration
DEFAULT_MODEL = "models/yolo11n.pt"  # Primary: models/yolo11n.pt, Fallback: models/yolov8n.pt
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
