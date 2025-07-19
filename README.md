# Video Detector - Vehicle Detection System

A comprehensive Python application for detecting vehicles (cars, trucks, buses, motorcycles) in videos using YOLO (You Only Look Once) deep learning models. 

**For Smooth Playback:**
1. **Use 720p quality** for best balance
2. **Choose models/yolo11s.pt model** for speed
3. **Increase confidence threshold** to 0.7+
4. **Ensure stable internet connection**

**For High Accuracy:**
1. **Use 1080p quality** when possible
2. **Choose models/yolo11l.pt model** for accuracy (default)
3. **Lower confidence threshold** to 0.4-0.5
4. **Use powerful hardware with GPU**

## Features

- **🎬 YouTube Video Analysis**: Watch and analyze YouTube videos/live streams from viewer perspective
- **📺 Live Stream Monitoring**: Real-time analysis of YouTube live traffic cameras
- **🚗 Real-time Detection**: Process webcam feed, video files, or streams  
- **🎯 Multiple Vehicle Types**: Detects cars, trucks, buses, and motorcycles
- **⚡ High Performance**: Optimized for speed with configurable confidence thresholds
- **🎨 Professional Overlays**: Broadcasting-quality overlays with video information
- **📊 Statistics Tracking**: Detailed analytics and performance metrics
- **💾 Export Capabilities**: Save processed videos and detection statistics
- **🏃 Benchmarking**: Performance testing tools included
- **📁 Organized Structure**: Models in `models/` directory, logs in `logs/` directory

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [🎬 YouTube Video Analysis](#-youtube-video-analysis-recommended)
  - [📹 Local Video Processing](#-local-video-processing)
- [📺 YouTube Video Analysis Guide](#-youtube-video-analysis-guide)
  - [Supported YouTube URLs](#supported-youtube-urls)
  - [Video Quality Options](#video-quality-options)
  - [YouTube Command Options](#youtube-command-options)
  - [YouTube Examples](#youtube-examples)
  - [Performance Tips for YouTube](#performance-tips-for-youtube)
- [Advanced Usage](#advanced-usage)
- [YOLO Model Variants](#yolo-model-variants)
- [API Usage](#api-usage)
- [Output Information](#output-information)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
  - [YouTube-Specific Issues](#youtube-specific-issues)
  - [General Issues](#general-issues)
- [Supported Formats](#supported-formats)
- [System Requirements](#system-requirements)
- [GPU Acceleration](#gpu-acceleration)
- [Legal and Ethical Considerations](#legal-and-ethical-considerations)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **The YOLO model will be automatically downloaded on first run**

## Quick Start

### 🎬 YouTube Video Analysis (Recommended!)

**Watch and analyze any YouTube video:**
```bash
python demo.py --youtube-watch "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Analyze YouTube live streams:**
```bash
python demo.py --youtube-watch "https://www.youtube.com/live/LIVE_ID"
```

**High-quality analysis:**
```bash
python demo.py --youtube-watch "https://youtu.be/VIDEO_ID" --quality 1080p
```

**Time-limited analysis:**
```bash
# Analyze for 5 minutes
python demo.py --youtube-watch "YOUTUBE_URL" --duration 5
```

### 📹 Local Video Processing

**Webcam detection:**
```bash
python demo.py --webcam
```

**Video file processing:**
```bash
python demo.py --video path/to/your/video.mp4
```

**Save output video:**
```bash
python demo.py --video input.mp4 --output detected_output.mp4
```

## 📺 YouTube Video Analysis Guide

### Supported YouTube URLs

The system supports various YouTube URL formats:

**Regular Videos:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

**Live Streams:**
- `https://www.youtube.com/live/LIVE_ID`
- `https://www.youtube.com/watch?v=LIVE_ID` (live videos)
- `https://www.youtube.com/@CHANNEL/live`
- `https://www.youtube.com/c/CHANNEL/live`

### Video Quality Options

| Quality | Resolution | Use Case | Bandwidth |
|---------|------------|----------|-----------|
| `480p` | 854×480 | Fast analysis, low bandwidth | ~1-2 Mbps |
| `720p` | 1280×720 | **Recommended balance** | ~2-4 Mbps |
| `1080p` | 1920×1080 | High quality analysis | ~4-8 Mbps |
| `best` | Highest available | Maximum quality | Varies |

### YouTube Command Options

```bash
python youtube_watcher.py [URL] [OPTIONS]
```

**YouTube-specific Options:**
- `--confidence, -c`: Detection confidence threshold (0.0-1.0, default: 0.6)
- `--model, -m`: YOLO model path (default: models/yolo11l.pt)
- `--quality, -q`: Video quality (480p, 720p, 1080p, best)
- `--output, -o`: Save processed video to file
- `--no-display`: Run without showing video (analysis only)
- `--max-duration, -d`: Maximum analysis time in seconds

### YouTube Examples

**High accuracy analysis:**
```bash
python youtube_watcher.py "https://youtu.be/VIDEO_ID" \
  --model models/yolo11l.pt \
  --confidence 0.4 \
  --quality 1080p
```

**Fast analysis for live monitoring:**
```bash
python youtube_watcher.py "https://youtube.com/live/LIVE_ID" \
  --model models/yolo11n.pt \
  --confidence 0.7 \
  --quality 720p
```

**Save analysis to file:**
```bash
python demo.py --youtube-watch "YOUTUBE_URL" --output analyzed_traffic.mp4
```

**Headless analysis (no display):**
```bash
python youtube_watcher.py "YOUTUBE_URL" --no-display --max-duration 300
```

### YouTube Features

**Real-Time Analysis:**
- Vehicle detection with colored bounding boxes
- Confidence scores for each detection
- Real-time vehicle counting by type
- FPS monitoring and performance metrics

**Professional Overlay:**
- 📺 Video title and channel name
- 🔴 Live stream indicator
- 🚗 Real-time vehicle statistics
- ⏱️ Timestamp and FPS counter
- 🎯 Vehicle type breakdown with colors

**On-Screen Display Example:**
```
📺 Live Traffic Camera - Downtown Intersection
by City Traffic Monitoring 🔴 LIVE
🚗 AI Vehicle Analysis

Vehicles: 12  Cars: 8  Trucks: 3  Bus: 1    FPS: 28.5
                                            12:34:56
```

### Controls During YouTube Analysis
- **'q'**: Quit analysis
- **'s'**: Save screenshot
- **Ctrl+C**: Stop analysis (terminal)

### Finding Traffic/Vehicle Content
Search YouTube for:
- "Traffic live cam"
- "Highway traffic live"
- "City traffic monitoring"
- "Intersection live camera"
- "Traffic jam live"

### Performance Tips for YouTube

**For Smooth Playback:**
1. **Use 720p quality** for best balance
2. **Choose models/yolo11n.pt model** for speed
3. **Increase confidence threshold** to 0.7+
4. **Ensure stable internet connection**

**For High Accuracy:**
1. **Use 1080p quality** when possible
2. **Choose models/yolo11l.pt model** for accuracy
3. **Lower confidence threshold** to 0.4-0.5
4. **Use powerful hardware with GPU**

### Bandwidth Requirements
- **480p**: ~1-2 Mbps download
- **720p**: ~2-4 Mbps download  
- **1080p**: ~4-8 Mbps download
- **Live streams**: Add 20-30% buffer

### Common YouTube Use Cases

**Traffic Monitoring:**
```bash
# Monitor highway traffic
python demo.py --youtube-watch "https://youtu.be/HIGHWAY_CAM" --quality 720p

# City intersection analysis  
python demo.py --youtube-watch "https://youtu.be/INTERSECTION_CAM" --duration 30
```

**Research and Analysis:**
```bash
# Detailed analysis with YOLOv11 (latest)
python youtube_watcher.py "RESEARCH_VIDEO_URL" \
  --model models/yolo11l.pt \
  --quality 1080p \
  --output detailed_analysis.mp4 \
  --confidence 0.3

# Try YOLOv10 (end-to-end detection)
python youtube_watcher.py "RESEARCH_VIDEO_URL" \
  --model models/yolov10m.pt \
  --quality 1080p \
  --confidence 0.4
```

**Live Event Monitoring:**
```bash
# Real-time live stream analysis
python demo.py --youtube-watch "https://youtube.com/live/EVENT_ID" --quality 720p
```

> 📺 **YouTube Features:**
> - **Watching**: For YouTube video analysis guide, see [YOUTUBE_WATCHING.md](YOUTUBE_WATCHING.md)

## Advanced Usage

### Command Line Interface

```bash
python video_detector.py [OPTIONS]
```

**Options:**
- `--source, -s`: Video source (file path, webcam index, or stream URL)
- `--model, -m`: YOLO model variant (models/yolo11n.pt, models/yolo11s.pt, models/yolo11m.pt, models/yolo11l.pt, models/yolo11x.pt)
- `--confidence, -c`: Confidence threshold (0.0-1.0, default: 0.5)
- `--output, -o`: Output video file path
- `--no-display`: Disable real-time video display
- `--benchmark`: Run performance benchmark
- `--benchmark-frames`: Number of frames for benchmarking (default: 100)

### Examples

**Process video with high confidence threshold:**
```bash
python video_detector.py --source traffic.mp4 --confidence 0.7 --output detected_traffic.mp4
```

**Use larger, more accurate model:**
```bash
python video_detector.py --source 0 --model models/yolo11l.pt
```

**Benchmark performance:**
```bash
python video_detector.py --source test_video.mp4 --benchmark --benchmark-frames 200
```

## YOLO Model Variants (Updated to YOLOv11 - Latest!)

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| models/yolo11n.pt | ~6MB | Fastest | Good | Real-time applications, limited hardware |
| models/yolo11s.pt | ~22MB | Fast | Better | Balanced performance |
| models/yolo11m.pt | ~52MB | Medium | High | High accuracy requirements |
| models/yolo11l.pt | ~87MB | Slow | Higher | **Maximum accuracy [DEFAULT]** |
| models/yolo11x.pt | ~136MB | Slowest | Highest | Research, offline processing |

> **⚡ Now using YOLOv11 (2024) - Latest release!**  
> - Improved efficiency over YOLOv8
> - Better speed-accuracy trade-off
> - Enhanced real-time performance
> - Automatic model download on first use

### Alternative Models Available:
- **YOLOv10**: `yolov10n.pt`, `yolov10s.pt`, etc. (End-to-end detection, no NMS)
- **YOLOv9**: `yolov9c.pt`, `yolov9e.pt` (Improved accuracy over YOLOv8)
- **YOLOv8**: `yolov8n.pt`, `yolov8s.pt`, etc. (Previous stable version)

## API Usage

### Basic Usage

```python
from video_detector import VideoDetector

# Initialize detector
detector = VideoDetector(
    model_path="models/yolo11l.pt",
    confidence_threshold=0.7
)

# Process video
stats = detector.process_video(
    source="path/to/video.mp4",
    output_path="output.mp4",
    display=True
)

print(f"Detected {stats['total_vehicle_detections']} vehicles")
```

### Advanced Configuration

```python
# Custom vehicle classes and colors
detector = VideoDetector()
detector.vehicle_classes = [2, 3, 5, 7]  # COCO class IDs
detector.colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 255),
    'bus': (0, 0, 255),
    'motorcycle': (255, 0, 0)
}

# Process with custom settings
stats = detector.process_video(
    source="traffic_cam_stream.rtsp",
    display=False,
    save_stats=True
)
```

### Single Frame Detection

```python
import cv2

# Load frame
frame = cv2.imread("traffic_image.jpg")

# Detect vehicles
detections = detector.detect_vehicles(frame)

# Draw results
annotated_frame = detector.draw_detections(frame, detections)

# Save result
cv2.imwrite("detected_image.jpg", annotated_frame)
```

## Output Information

### Real-time Display
- **Bounding boxes**: Colored rectangles around detected vehicles
- **Labels**: Vehicle type and confidence score
- **Statistics overlay**: FPS, total count, breakdown by vehicle type
- **YouTube overlay**: Video title, channel name, live indicator (for YouTube videos)

### YouTube Terminal Output
```
🎬 Starting YouTube video analysis...
   URL: https://youtu.be/VIDEO_ID
   Quality: 720p
   Model: yolo11l.pt
   Confidence: 0.6

✅ Analysis completed!
   Video: Live Traffic Camera - Downtown Intersection
   Channel: City Traffic Monitoring
   Total vehicles detected: 1,247
   Vehicle breakdown: {'car': 892, 'truck': 234, 'bus': 67, 'motorcycle': 54}
```

### Saved Statistics
The system automatically saves detection statistics in JSON format:

```json
{
  "video_info": {
    "title": "Live Traffic Camera - Downtown",
    "uploader": "Traffic Monitor",
    "is_live": true,
    "duration": 0,
    "view_count": 15432
  },
  "total_frames_processed": 8640,
  "average_fps": 28.8,
  "total_vehicle_detections": 1247,
  "vehicle_counts_by_type": {
    "car": 892,
    "truck": 234, 
    "bus": 67,
    "motorcycle": 54
  }
}
```

## Performance Optimization

### Hardware Acceleration
- **GPU Support**: Automatically uses CUDA if available (see [GPU Acceleration](#gpu-acceleration))
- **CPU Optimization**: Multi-threading for CPU-only systems
- **Memory Management**: Automatic VRAM optimization for different GPU sizes

### Speed vs Accuracy Trade-offs
1. **Maximum Speed**: Use `models/yolo11n.pt` with confidence 0.6+ (GPU: ~200-300 FPS)
2. **Balanced**: Use `models/yolo11s.pt` with confidence 0.5 (GPU: ~150-200 FPS)
3. **Maximum Accuracy**: Use `models/yolo11l.pt` or `models/yolo11x.pt` with confidence 0.3-0.4 (GPU: ~50-80 FPS) **[DEFAULT]**

### Tips for Better Performance
- **Enable GPU acceleration**: 10-12x speed improvement (see [GPU Acceleration](#gpu-acceleration))
- Lower input resolution for faster processing
- Increase confidence threshold to reduce false positives
- Process every nth frame for real-time applications

## Troubleshooting

### YouTube-Specific Issues

**"Invalid YouTube URL":**
- Verify the URL is correct and accessible
- Try different URL formats (youtu.be vs youtube.com)
- Check if video is public or region-locked

**"Failed to extract video information":**
- Video might be private, deleted, or region-restricted
- Try updating yt-dlp: `pip install --upgrade yt-dlp`
- Check internet connection

**"Failed to open video stream":**
- Video format might not be supported
- Try different quality setting
- Check if video requires age verification

**Low FPS or choppy YouTube playback:**
- Use lower quality (720p → 480p)
- Use faster model (models/yolo11l.pt → models/yolo11n.pt)
- Increase confidence threshold
- Check internet speed

**No detections on YouTube videos:**
- Lower confidence threshold (0.6 → 0.4)
- Check if video actually contains vehicles
- Try different YOLO model
- Verify video quality and lighting

**Unicode/Emoji errors in video titles:**
- This is automatically handled by safe logging functions
- Emoji characters are replaced with text equivalents (🔴 → [LIVE])
- UTF-8 encoding is used for log files to support international characters

**YouTube Error Recovery:**
```bash
# Update yt-dlp to latest version
pip install --upgrade yt-dlp

# Test with simple command
python youtube_watcher.py "https://youtu.be/SHORT_VIDEO_ID" --max-duration 30

# Check video info only
python -c "import yt_dlp; ydl = yt_dlp.YoutubeDL(); print(ydl.extract_info('YOUR_URL', download=False)['title'])"
```

### General Issues

**Model Download Failed:**
```bash
# Manually download model to models/ directory
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -P models/
```

**Low FPS:**
- Enable GPU acceleration (see [GPU Acceleration](#gpu-acceleration))
- Use faster model (models/yolo11s.pt or models/yolo11n.pt)
- Increase confidence threshold

**CUDA Out of Memory:**
- Use smaller model variant (yolo11l.pt → yolo11s.pt → yolo11n.pt)
- Reduce input resolution
- Lower batch size (see [GPU Memory Management](#gpu-memory-management))

**No Detections:**
- Lower confidence threshold
- Check lighting conditions
- Verify vehicle types in frame

## Supported Formats

### Input Formats
- **Video Files**: MP4, AVI, MOV, MKV, FLV
- **Image Sequences**: JPG, PNG, BMP
- **Streams**: RTSP, HTTP, webcam
- **Devices**: USB cameras, IP cameras

### Output Formats
- **Video**: MP4 (H.264)
- **Images**: JPG, PNG
- **Data**: JSON statistics, CSV logs

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- CPU: Any modern processor

### Recommended Requirements
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with CUDA support
- SSD storage

## GPU Acceleration

🚀 **GPU acceleration provides 10-12x faster performance for real-time video analysis!**

### Check GPU Status

Run the GPU check script to see your current configuration:
```bash
python check_gpu.py
```

**Expected Output with GPU:**
```
✅ GPU acceleration enabled: NVIDIA GeForce RTX 3070 (8.0 GB)
✅ CUDA available: True
✅ YOLO model device: cuda:0
```

**Output without GPU:**
```
⚠️ GPU requested but not available - falling back to CPU
💡 Install CUDA-enabled PyTorch for GPU acceleration
```

### Enable GPU Acceleration

#### Step 1: Verify NVIDIA GPU
```bash
nvidia-smi
```

#### Step 2: Install CUDA-Enabled PyTorch
```bash
# Uninstall CPU-only version
pip uninstall torch torchvision torchaudio

# Install CUDA version (CUDA 12.1 - recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Step 3: Configure GPU Usage
GPU acceleration is enabled by default in `config.py`:
```python
USE_GPU = True  # Set to False to force CPU usage
```

### Performance Comparison

| Model | CPU (Intel i7) | GPU (RTX 3070) | Speedup |
|-------|----------------|----------------|---------|
| yolo11n.pt | ~20-30 FPS | ~200-300 FPS | **10x faster** |
| yolo11s.pt | ~15-20 FPS | ~150-200 FPS | **10x faster** |
| yolo11m.pt | ~8-12 FPS | ~80-120 FPS | **10x faster** |
| yolo11l.pt | ~4-8 FPS | ~50-80 FPS | **12x faster** |

### GPU Memory Management

**If you get CUDA out of memory errors:**
- Use smaller models: `yolo11l.pt` → `yolo11s.pt` → `yolo11n.pt`
- Reduce video resolution
- Lower batch size in config.py
- Process fewer frames simultaneously

**GPU Memory Usage by Model:**
- `yolo11n.pt`: ~1-2 GB VRAM
- `yolo11s.pt`: ~2-3 GB VRAM
- `yolo11m.pt`: ~3-4 GB VRAM
- `yolo11l.pt`: ~4-6 GB VRAM **[DEFAULT]**
- `yolo11x.pt`: ~6-8 GB VRAM

### Troubleshooting GPU Issues

**No NVIDIA GPU detected:**
- Install NVIDIA drivers: https://www.nvidia.com/drivers
- Restart after driver installation

**CUDA installation issues:**
- Check CUDA version: `nvcc --version`
- Use matching PyTorch version
- Visit: https://pytorch.org/get-started/locally/

**GPU not being used:**
- Verify `USE_GPU = True` in config.py
- Check GPU status with `python check_gpu.py`
- Restart application after PyTorch installation

## Legal and Ethical Considerations

When using YouTube video analysis features:

- ✅ **Allowed**: Watching public YouTube videos for analysis
- ✅ **Allowed**: Analyzing traffic cameras and public streams  
- ✅ **Allowed**: Educational and research purposes
- ⚠️ **Check**: Copyright restrictions for saving/redistribution
- ⚠️ **Check**: Terms of service for automated access
- ❌ **Avoid**: Downloading copyrighted content
- ❌ **Avoid**: Analyzing private or restricted content

**Remember to respect content creators and follow YouTube's terms of service!** 🚗📺

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Review the logs in `logs/video_detector.log`
3. Open an issue with detailed information about your setup and the problem
