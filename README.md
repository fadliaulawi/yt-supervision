# 🚗 Vehicle Detection System

A comprehensive Python application for detecting and analyzing vehicles (cars, trucks, buses, motorcycles) in video streams using state-of-the-art YOLOv11 deep learning models. Features both an interactive web dashboard and powerful command-line tools.

## ✨ Key Features

### 🌐 Interactive Web Dashboard
- **📱 Modern UI**: Clean Streamlit interface with real-time updates
- **📊 Live Analytics**: Interactive charts showing detection trends and directional traffic distribution
- **↔️ Directional Analysis**: Left/right traffic flow visualization with unique vehicle tracking
- **🎯 Smart Controls**: Model selection, confidence adjustment, and quality settings
- **📈 Performance Metrics**: Real-time FPS, processing time, and detection statistics
- **💾 Session Management**: Results persistence within dashboard session

### 🎥 Multi-Source Analysis
- **📺 YouTube Integration**: Analyze any YouTube video or live stream with `yt-dlp`
- **📁 File Processing**: Support for MP4, AVI, MOV, MKV, and FLV video formats
- **🎥 Live Webcam**: Real-time detection from camera feed
- **🔥 Fire-and-Forget Processing**: Independent analysis processes with status tracking

### 🎯 Directional Traffic Analysis
- **↔️ Center Line Division**: Automatic frame splitting with vertical center line visualization
- **📊 Left/Right Counting**: Separate vehicle counts for each direction
- **🚗 Unique Vehicle Tracking**: SORT algorithm prevents duplicate counting across frames
- **📈 Directional Statistics**: Real-time charts showing traffic flow distribution
- **🎨 Visual Overlays**: Cyan dashed center line with L/R directional labels

### 🤖 Advanced AI Models  
- **YOLOv11n**: Ultra-fast processing for real-time applications
- **YOLOv11l**: Balanced accuracy and speed (default recommendation)
- **YOLOv11x**: Maximum precision for research and critical applications
- **⚡ GPU Acceleration**: CUDA support with automatic fallback to CPU
- **🎯 Smart Detection**: Cars, trucks, buses, and motorcycles with confidence scoring

### 🛠️ Technical Excellence
- **📊 Centralized Status Management**: Real-time process tracking and error handling
- **⚙️ Unified Configuration**: Single source for all system and dashboard settings
- **🔧 Modular Architecture**: Clean separation of concerns with reusable components
- **📝 Comprehensive Logging**: Detailed analysis results with timestamped JSON exports

## 🚀 Quick Start

### 1. System Check (Recommended)
```bash
# Run comprehensive system validation
python system_check.py

# This will check:
# - Python version compatibility
# - Required dependencies installation
# - GPU/CUDA configuration
# - YOLO model availability  
# - Performance benchmarking
# - Personalized recommendations
```

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd yt-supervision

# Install dependencies
pip install -r requirements.txt

# Verify installation 
python system_check.py
```

### 3. Launch Web Dashboard
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

**Dashboard Features:**
- 🎛️ **Control Panel**: Choose analysis mode, model, and settings
- 📊 **Live Monitoring**: Real-time status updates and progress tracking  
- 📈 **Interactive Charts**: Vehicle distribution, directional traffic flow, and performance analytics
- 🎥 **Live Preview**: See analysis progress with directional center line in independent OpenCV window
- ↔️ **Directional Analysis**: Left/right traffic statistics with unique vehicle counting
- 💾 **Results Export**: Download analysis data and directional statistics

### 4. Command Line Usage

⚠️ **Pre-flight Check**: Run `python system_check.py` first to verify your system is ready!

**Analyze YouTube video/stream:**
```bash
python demo.py --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
python demo.py --youtube "https://youtu.be/VIDEO_ID" --quality 720p
```

**Process local video file:**
```bash
python demo.py --video path/to/video.mp4
```

**Live webcam detection:**
```bash
python demo.py --webcam
```

**Advanced options:**
```bash
# Custom model and confidence
python demo.py --video file.mp4 --model yolo11l.pt --confidence 0.7

# YouTube with specific quality
python demo.py --youtube "URL" --quality 1080p --model yolo11x.pt
```
- `--output`: Output file path for processed video

## 🎛️ Dashboard Architecture

### Fire-and-Forget Processing
The dashboard uses an innovative **independent process architecture**:

1. **🚀 Process Launching**: Dashboard spawns independent `demo.py` subprocess
2. **🔄 Status Tracking**: Centralized status management via JSON files  
3. **🖥️ OpenCV Window**: Analysis runs in separate OpenCV window (press 'Q' to stop)
4. **📊 Real-time Updates**: Dashboard monitors progress and displays results
5. **💾 Session Results**: Analysis data persists within dashboard session

**Key Benefits:**
- ✅ **No Performance Impact**: Streamlit UI remains responsive during analysis
- ✅ **Process Isolation**: Analysis crashes won't affect dashboard
- ✅ **Cross-Platform**: Works on Windows, macOS, and Linux
- ✅ **User Control**: Direct interaction with analysis via OpenCV window

### Dashboard Components
- **📺 YouTube Input**: Paste any YouTube URL, set quality and duration
- **📁 File Upload**: Drag & drop video files with automatic format detection
- **🎥 Webcam Mode**: Live detection from camera with real-time preview
- **📊 Analytics**: Vehicle distribution charts, directional traffic flow, and detection timelines
- **↔️ Directional Stats**: Left/right vehicle counts with unique tracking visualization
- **⚙️ Model Selection**: Choose between speed and accuracy (YOLOv11n/l/x)
- **🎚️ Confidence Control**: Adjust detection sensitivity (0.1 - 1.0)
- **⚡ Performance Display**: FPS monitoring and frame processing stats

## 🔧 System Architecture

### Core Components
```
📂 yt-supervision/
├── 🚀 app.py                     # Main dashboard entry point
├── 🎬 demo.py                    # Unified command-line interface  
├── ⚙️ check_gpu.py              # GPU compatibility checker
├── 📚 modules/                   # Core system modules
│   ├── 🎯 video_detector.py     # YOLO detection engine with directional analysis
│   ├── 📺 youtube_watcher.py    # YouTube stream processor with traffic flow tracking
│   ├── 🎛️ dashboard_core.py     # Web interface logic with directional charts
│   ├── 📊 status_manager.py     # Centralized status tracking
│   ├── ⚙️ config.py             # Unified configuration
│   └── 🛠️ utils.py              # Helper utilities
├── 🤖 models/                    # YOLO model storage (auto-download)
├── 📊 logs/                      # Analysis results and logs
└── 📋 requirements.txt          # Python dependencies
```

### Model Management
Models are automatically downloaded to `models/` directory on first use:

| Model | Speed | Accuracy | Size | Best For |
|-------|--------|----------|------|----------|
| **yolo11n.pt** | Very Fast | Good | ~6MB | Real-time streams, live webcam |
| **yolo11l.pt** | Medium | High | ~87MB | **Default balanced choice** |
| **yolo11x.pt** | Slow | Maximum | ~136MB | Research, maximum precision |

### Configuration System
The system uses a **unified configuration approach**:

```python
from modules.config import Config

# Access system settings
model = Config.get_model_by_performance("balanced")  # -> yolo11l.pt
confidence = Config.get_confidence_by_setting("strict")  # -> 0.7
quality = Config.get_quality_by_priority("balanced")  # -> 720p
```

## 📖 Documentation

For detailed documentation, configuration options, and troubleshooting, see [DOCS.md](DOCS.md).

## ❗ Requirements

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **RAM**: 4GB minimum, 8GB+ recommended  
- **Storage**: 1GB for models and temporary files
- **GPU**: CUDA-compatible GPU optional (significant speed boost)
- **Network**: Internet connection for YouTube analysis and model downloads

### Python Dependencies
```
# Core Processing
ultralytics>=8.0.0      # YOLO models and training
opencv-python>=4.8.0    # Computer vision and video processing  
torch>=1.13.0           # PyTorch deep learning framework
numpy<2.0.0             # Numerical computing (pinned for compatibility)

# Web Dashboard  
streamlit>=1.28.0       # Web interface framework
plotly>=5.15.0          # Interactive charts and visualizations
pandas>=1.5.0           # Data manipulation and analysis

# Media Processing
yt-dlp>=2023.7.6        # YouTube video downloading and streaming
Pillow>=9.0.0           # Image processing and manipulation
matplotlib>=3.5.0       # Basic plotting and visualization

# Utilities
PyYAML>=6.0             # Configuration file parsing
tqdm>=4.64.0            # Progress bars and status updates
```

All dependencies are automatically installed with:
```bash
pip install -r requirements.txt
```

---
*For comprehensive documentation and advanced usage, see [DOCS.md](DOCS.md)*
