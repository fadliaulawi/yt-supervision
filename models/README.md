# Models Directory

This directory contains the YOLO model files used for vehicle detection.

## Available Models

The following YOLO v11 models will be automatically downloaded when first used:

| Model File | Size | Speed | Accuracy | Description |
|------------|------|-------|----------|-------------|
| `yolo11n.pt` | ~6MB | Fastest | Good | Nano model - best for real-time applications |
| `yolo11s.pt` | ~22MB | Fast | Better | Small model - good balance of speed/accuracy |
| `yolo11m.pt` | ~52MB | Medium | High | Medium model - higher accuracy |
| `yolo11l.pt` | ~87MB | Slow | Higher | Large model - best for accuracy |
| `yolo11x.pt` | ~136MB | Slowest | Highest | Extra large model - research/offline use |

## Alternative YOLO Versions

While YOLOv11 is the recommended default (latest 2024 release), other versions are also supported:

### YOLOv10 (End-to-End Detection)
- `yolov10n.pt`, `yolov10s.pt`, `yolov10m.pt`, `yolov10l.pt`, `yolov10x.pt`
- No NMS (Non-Maximum Suppression) required
- Faster inference for some use cases

### YOLOv9 (Improved Accuracy)
- `yolov9c.pt`, `yolov9e.pt`
- Better accuracy than YOLOv8 
- Good for research applications

### YOLOv8 (Previous Stable)
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- Well-tested and stable
- Broad compatibility

## Model Selection Guide

### For Real-Time Applications
- **Webcam detection**: Use `yolo11n.pt`
- **Live stream analysis**: Use `yolo11n.pt` or `yolo11s.pt`
- **Mobile/embedded**: Use `yolo11n.pt`

### For High Accuracy
- **Research analysis**: Use `yolo11l.pt` or `yolo11x.pt`
- **Offline processing**: Use `yolo11l.pt` or `yolo11x.pt`
- **Professional analysis**: Use `yolo11m.pt` or `yolo11l.pt`

### For Balanced Performance
- **General purpose**: Use `yolo11s.pt`
- **YouTube video analysis**: Use `yolo11s.pt` or `yolo11m.pt`
- **Mixed workloads**: Use `yolo11s.pt`

## Automatic Download

Models are automatically downloaded from the official Ultralytics repository when first used. The system will:

1. Check if the model file exists in this directory
2. If not found, download from: `https://github.com/ultralytics/assets/releases/`
3. Cache the model locally for future use

## Manual Download

To manually download models:

```bash
# Download nano model (fastest)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -P models/

# Download small model (balanced)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt -P models/

# Download large model (most accurate)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt -P models/
```

## Model Performance

Performance varies based on hardware:

### CPU Performance (Intel i7)
- `yolo11n.pt`: ~50-80 FPS
- `yolo11s.pt`: ~30-50 FPS  
- `yolo11m.pt`: ~15-25 FPS
- `yolo11l.pt`: ~8-15 FPS

### GPU Performance (RTX 3070)
- `yolo11n.pt`: ~200-300 FPS
- `yolo11s.pt`: ~150-200 FPS
- `yolo11m.pt`: ~80-120 FPS  
- `yolo11l.pt`: ~50-80 FPS

## Vehicle Detection Classes

All models detect the same vehicle classes from the COCO dataset:

- **Class 2**: Car
- **Class 3**: Motorcycle  
- **Class 5**: Bus
- **Class 7**: Truck

## Model Updates

Models are periodically updated by Ultralytics. To get the latest version:

1. Delete the existing model file from this directory
2. Run the application - it will download the latest version
3. Or manually download the latest version from the official repository

## Troubleshooting

**Model download fails:**
- Check internet connection
- Verify firewall/proxy settings
- Try manual download with wget/curl

**Out of memory errors:**
- Use smaller model (yolo11l.pt → yolo11n.pt)
- Reduce input resolution
- Lower batch size

**Poor detection accuracy:**
- Use larger model (yolo11n.pt → yolo11l.pt)
- Lower confidence threshold
- Ensure good video quality and lighting
