# 📖 Vehicle Detection System - Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Installation & Setup](#installation--setup)
- [Dashboard Guide](#dashboard-guide)
- [Command Line Usage](#command-line-usage)
- [Directional Traffic Analysis](#directional-traffic-analysis)
- [YouTube Analysis](#youtube-analysis)
- [API Reference](#api-reference)
- [Model Configuration](#model-configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## System Overview

YT-Supervision is a Python application for real-time vehicle detection using YOLOv11 models. Features a Streamlit web dashboard and command-line interface supporting multiple video sources.

### Architecture
- **🔄 Fire-and-Forget Processing**: Independent subprocess execution prevents UI blocking
- **📊 Centralized Status Management**: Real-time process tracking via `status_manager.py`
- **🎯 Modular Design**: Clean separation with unified configuration system
- **🔧 Cross-Platform**: Windows, macOS, and Linux support

### Key Features
- **Multi-Source Input**: YouTube, local files, webcam with auto-detection
- **Directional Traffic Analysis**: Center line division with left/right vehicle counting
- **Unique Vehicle Tracking**: SORT algorithm prevents duplicate counting across frames
- **Real-time Processing**: Live analysis with interactive directional visualizations
- **AI Models**: YOLOv11n/l/x variants with automatic GPU acceleration
- **Professional Output**: Broadcasting-quality annotations with traffic flow analytics

## Installation & Setup

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU**: CUDA-compatible GPU recommended (10x speed improvement)
- **Network**: Internet for YouTube analysis and model downloads

### Installation Steps

1. **Setup Repository**
   ```bash
   git clone <repository-url>
   cd yt-supervision
   ```

2. **Create Environment** (Recommended)
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/macOS: source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **System Validation** ⭐ **IMPORTANT**
   ```bash
   python system_check.py
   # Validates Python, dependencies, GPU, models, and provides personalized recommendations
   ```

## Dashboard Guide

### Launch
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Fire-and-Forget Architecture
The dashboard launches `demo.py` as an independent subprocess:
1. **Process Spawning**: Analysis runs separately from UI
2. **Status Monitoring**: Real-time tracking via `status_manager.py`
3. **OpenCV Window**: Direct analysis control (press 'Q' to stop)
4. **UI Responsiveness**: Dashboard stays responsive during processing

### Components
- **Control Panel**: Mode selection (YouTube/File/Webcam), model choice, confidence settings
- **Status Display**: Real-time analysis state with progress indicators
- **Analytics**: Vehicle counts, directional traffic charts, interactive flow visualizations
- **Directional Stats**: Left/right vehicle tracking with unique count prioritization
- **Results**: Session-based data persistence with directional analysis export

### Usage Flow
1. Select input source and model
2. Click "Start Analysis" 
3. Control analysis via OpenCV window
4. Results automatically load in dashboard when complete

## Command Line Usage

### Basic Commands
```bash
# YouTube analysis
python demo.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"

# Local video file
python demo.py --video "path/to/video.mp4"

# Webcam feed
python demo.py --webcam
```

### Advanced Options
```bash
# Model selection
python demo.py --youtube "URL" --model yolo11n.pt  # Fastest
python demo.py --video "file.mp4" --model yolo11l.pt  # Balanced (default)
python demo.py --webcam --model yolo11x.pt  # Most accurate

# Confidence and quality control
python demo.py --youtube "URL" --confidence 0.8 --quality 720p
```

### Architecture
- **Independent Process**: Runs separately from dashboard
- **Status Integration**: Updates tracked via `status_manager.py`
- **OpenCV Display**: Direct visual feedback with directional center line
- **Results Export**: Timestamped JSON output with directional statistics (`analysis_YYYYMMDD_HHMMSS.json`)

## Directional Traffic Analysis

### Overview
The system provides comprehensive directional traffic analysis by dividing the video frame with a vertical center line and tracking vehicles moving left vs. right.

### Key Features
- **🎯 Center Line Division**: Automatic vertical line at frame center (50% width)
- **↔️ Directional Counting**: Separate vehicle counts for left and right directions
- **🚗 Unique Vehicle Tracking**: SORT algorithm prevents duplicate counting
- **📊 Real-time Statistics**: Live directional charts in dashboard
- **🎨 Visual Overlay**: Cyan dashed center line with L/R labels

### How It Works
```python
# Directional logic
frame_center = frame_width // 2
vehicle_center_x = (bbox_x1 + bbox_x2) // 2

if vehicle_center_x < frame_center:
    direction = "left"   # Left side of center line
else:
    direction = "right"  # Right side of center line
```

### Visual Elements
- **Center Line**: Cyan dashed vertical line at 50% frame width
- **Direction Labels**: "L" and "R" markers at top of center line
- **Vehicle Bounding Boxes**: Color-coded by vehicle type
- **Tracking IDs**: Unique numbers for each tracked vehicle (when enabled)

### Statistics Output
```json
{
  "directional_counts": {
    "left": {"car": 15, "truck": 3, "bus": 1, "motorcycle": 7},
    "right": {"car": 22, "truck": 5, "bus": 2, "motorcycle": 4}
  },
  "unique_directional_counts": {
    "left": {"car": 12, "truck": 2, "bus": 1, "motorcycle": 5},
    "right": {"car": 18, "truck": 4, "bus": 1, "motorcycle": 3}
  }
}
```

### Dashboard Integration
- **Directional Charts**: Side-by-side bar charts comparing left vs. right traffic
- **Traffic Flow Pie Chart**: Visual proportion of directional movement
- **Unique Count Priority**: Dashboard prioritizes unique counts over general detections
- **Real-time Updates**: Charts update automatically during analysis

### Usage Examples
```bash
# All analysis modes include directional tracking by default

# YouTube with directional analysis
python demo.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"

# Local video with tracking enabled (recommended for unique counts)
python demo.py --video "traffic.mp4" --model yolo11l.pt

# Webcam with real-time directional analysis
python demo.py --webcam --confidence 0.7
```

## YouTube Analysis

### Supported Sources
```python
# Standard formats
"https://www.youtube.com/watch?v=VIDEO_ID"
"https://youtu.be/VIDEO_ID"
"https://www.youtube.com/live/LIVE_ID"  # Live streams
"https://www.youtube.com/watch?v=VIDEO_ID&t=120s"  # Timestamped
```

### Quality & Performance
| Quality | Resolution | Bandwidth | Processing Speed | Use Case |
|---------|------------|-----------|------------------|----------|
| **480p** | 854×480 | ~1 Mbps | ⚡ ~25-35 FPS | Fast, live streams |
| **720p** | 1280×720 | ~2.5 Mbps | ⚖️ ~15-25 FPS | **Recommended** |
| **1080p** | 1920×1080 | ~5 Mbps | 🎯 ~8-15 FPS | High accuracy |
| **best** | Variable | Variable | 🔍 Variable | Maximum quality |

### Usage Examples
```bash
# Balanced processing
python demo.py --youtube "URL" --quality 720p --model yolo11l.pt

# Fast live stream processing  
python demo.py --youtube "LIVE_URL" --quality 480p --model yolo11n.pt

# High accuracy analysis
python demo.py --youtube "URL" --quality 1080p --model yolo11x.pt --confidence 0.8
```

## API Reference

### Core Modules
```python
# Configuration & Status
from modules.config import Config
from modules.status_manager import set_analysis_starting, get_analysis_status

# Video Processing
from modules.video_detector import VideoDetector
from modules.utils import validate_video_source, export_analysis_results
```

### VideoDetector Class
```python
# Initialize detector with tracking enabled
detector = VideoDetector(
    model_path="models/yolo11l.pt",
    confidence_threshold=0.7,
    device="auto",  # Auto GPU detection
    enable_tracking=True  # Enable unique vehicle tracking
)

# Core methods
detector.detect_vehicles(frame)              # Single frame detection
detector.detect_and_track_vehicles(frame)    # Detection with tracking
detector.process_video_source(source)        # Full video processing
detector.draw_detections(frame, results)     # Annotate frame with bounding boxes
detector._draw_center_line(frame)            # Add directional center line
detector.add_info_overlay(frame, detections, fps)  # Complete overlay
detector.get_performance_stats()             # FPS and timing metrics
detector.update_statistics(detections, frame_width)  # Update directional stats
```

### Status Manager
```python
# Status management functions
set_analysis_starting(metadata=None)
set_analysis_running(current_frame, total_frames, fps)
set_analysis_completed(results_file, summary)
set_analysis_error(error_message, details=None)
get_analysis_status()  # Returns current status dict
```

### Configuration System
```python
from modules.config import Config

# System settings
Config.MODELS              # Available YOLO models
Config.USE_GPU            # Auto-detected GPU availability  
Config.VEHICLE_CLASSES    # ["car", "truck", "bus", "motorcycle"]
Config.CONFIDENCE_PRESETS # Common confidence thresholds
Config.YOUTUBE_QUALITIES  # Available quality options

# Directional analysis settings
Config.ENABLE_TRACKING    # Enable unique vehicle tracking (default: True)
Config.CENTER_LINE_COLOR  # Directional line color (default: cyan)
Config.DIRECTION_LABELS   # Left/Right labels (default: ["L", "R"])
```

## Model Configuration

### Available Models
| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| **yolo11n.pt** | Very Fast | Good | ~6 MB | Real-time, live streams |
| **yolo11l.pt** | Medium | Very High | ~87 MB | **Default balanced choice** |
| **yolo11x.pt** | Slow | Maximum | ~136 MB | Research, maximum precision |

### Selection Guide
```python
# Choose based on use case
USE_CASES = {
    "live_webcam": "yolo11n.pt",       # Speed priority
    "youtube_analysis": "yolo11l.pt",   # Accuracy priority  
    "batch_processing": "yolo11x.pt",   # Maximum accuracy
    "production": "yolo11l.pt"          # Reliable default
}
```

Models auto-download to `models/` directory on first use.

## Performance Optimization

### Hardware-Aware Configuration
```python
# Automatic optimization based on available hardware
performance_matrix = {
    "GPU + YOLOv11n": "~25-35 FPS (1080p)",
    "GPU + YOLOv11l": "~15-25 FPS (1080p)",   # Recommended
    "GPU + YOLOv11x": "~5-12 FPS (1080p)",
    "CPU + YOLOv11n": "~8-15 FPS (720p)",     # CPU fallback
}
```

### Optimization Strategies
- **Real-time**: Use `yolo11n.pt` with 720p resolution and confidence 0.7+
- **High-accuracy**: Use `yolo11x.pt` with 1080p resolution and confidence 0.4+
- **Balanced**: Use `yolo11l.pt` with 720p resolution (recommended)
- **Directional Tracking**: Enable tracking for accurate unique counts (slight performance impact)

### Memory Management
- Automatic GPU memory optimization
- Periodic cleanup every 100 frames
- Buffer limits to prevent memory growth
- Tracking memory management for unique vehicle IDs

## Troubleshooting

### Quick Diagnostics

#### System Check ⭐ **START HERE**
```bash
python system_check.py
# Provides complete system analysis:
# - Hardware detection, dependency verification
# - Model availability, performance benchmarking  
# - Personalized recommendations and troubleshooting
```

### Common Issues

#### 🚫 Dashboard Issues
```bash
# Dashboard won't start
pip install -r requirements.txt
streamlit run app.py --server.port 8502  # Different port

# Dashboard freezes during analysis (expected behavior)
# Use OpenCV window for control, press 'Q' to stop
```

#### 🤖 Model Problems
```bash
# Models not found - auto-download on first use
python demo.py --video sample.mp4

# Out of memory - use smaller model
python demo.py --youtube "URL" --model yolo11n.pt --quality 480p
```

#### 📺 YouTube Issues
```bash
# Video unavailable - check URL and update yt-dlp
pip install -U yt-dlp

# Connection timeouts - use lower quality
python demo.py --youtube "URL" --quality 480p
```

#### ⚡ Performance Issues
```bash
# Slow processing - check GPU acceleration
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA PyTorch if needed
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 🔧 Process Issues
```bash
# Analysis stuck - press 'Q' in OpenCV window
# Clear stuck status
rm status.json

# Check logs for detailed errors
tail -f logs/video_detector.log
```

### Error Solutions
- **Model download failed**: Check internet, manually download if needed
- **CUDA errors**: Update GPU drivers, reinstall CUDA PyTorch
- **FFmpeg missing**: Install via system package manager
- **Streamlit port conflict**: Use `--server.port 8502`
- **Webcam issues**: Check permissions, try different index

### Emergency Reset
```bash
rm status.json
pip install --force-reinstall -r requirements.txt
python system_check.py
```

## Project Structure

### File Organization
```
📂 yt-supervision/
├── 🚀 app.py                      # Streamlit dashboard launcher
├── 🎬 demo.py                     # Unified CLI interface
├── 🔍 system_check.py             # Comprehensive system validation
├── 📁 modules/                    # Core system modules
│   ├── ⚙️ config.py               # Unified configuration
│   ├── 🎥 video_detector.py       # YOLO detection engine with directional analysis
│   ├── 📊 status_manager.py       # Centralized status tracking
│   ├── 🖥️ dashboard_core.py       # Dashboard logic with directional charts
│   ├── 📺 youtube_watcher.py      # YouTube processor with traffic flow tracking
│   └── 🛠️ utils.py                # Helper utilities
├── 🤖 models/                     # YOLO model storage
│   ├── ⚡ yolo11n.pt              # Fastest model (auto-downloaded)
│   ├── ⚖️ yolo11l.pt              # Balanced model (default)
│   └── 🎯 yolo11x.pt              # Most accurate model
├── 📊 logs/                       # Analysis logs & debugging
├── 📊 analysis_*.json             # Timestamped analysis results
├── ⚙️ status.json                # Temporary status file
├── 📖 README.md                   # Quick start guide
├── 📚 DOCS.md                     # This documentation
├── 📦 requirements.txt            # Dependencies
└── 🎬 sample.mp4                  # Demo video
```

### Architecture Components

#### Core Principles
- **Process Isolation**: `demo.py` runs independently from dashboard
- **Status Centralization**: `status_manager.py` coordinates all components
- **Configuration Unification**: Single `config.py` for all settings
- **Session Persistence**: Results available within dashboard sessions

#### Module Dependencies
```python
# Dependency hierarchy
"app.py": ["modules.dashboard_core", "streamlit"]
"demo.py": ["modules.video_detector", "modules.status_manager", "modules.config"]
"modules.dashboard_core": ["modules.status_manager", "subprocess", "streamlit"]
"modules.video_detector": ["ultralytics", "opencv-python", "yt-dlp"]
```

### Data Flow
1. **Initialization**: Dashboard or CLI validates configuration and enables tracking
2. **Process Launch**: Independent subprocess spawned for analysis with directional features
3. **Status Tracking**: Real-time updates via `status_manager.py` with directional statistics
4. **Analysis**: OpenCV window displays results with center line, user controls execution
5. **Completion**: Results saved as timestamped JSON files with directional and unique counts
6. **Integration**: Dashboard automatically loads completed analysis with directional charts

---

## Summary

This documentation covers the **YT-Supervision Vehicle Detection System** featuring:

- ✅ **Fire-and-Forget Architecture**: Independent processing prevents UI blocking
- ✅ **Directional Traffic Analysis**: Center line division with left/right vehicle counting
- ✅ **Unique Vehicle Tracking**: SORT algorithm prevents duplicate counting across frames
- ✅ **Unified Interface**: Single `demo.py` for all analysis types with directional features
- ✅ **Centralized Management**: `status_manager.py` coordinates components
- ✅ **Auto-Configuration**: GPU detection, model management, tracking enablement
- ✅ **Session-Based Results**: Dashboard persistence with directional analytics

For quick start instructions, see **[README.md](README.md)**.

---
*Documentation version: 2025-07-19 | Architecture: Fire-and-Forget Processing*
