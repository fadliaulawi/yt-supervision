# Models Directory

This directory contains the YOLO model files used for vehicle detection.

## Available Models

The following YOLO v8 models will be automatically downloaded when first used:

| Model File | Size | Speed | Accuracy | Description |
|------------|------|-------|----------|-------------|
| `yolov8n.pt` | ~6MB | Fastest | Good | Nano model - best for real-time applications |
| `yolov8s.pt` | ~22MB | Fast | Better | Small model - good balance of speed/accuracy |
| `yolov8m.pt` | ~52MB | Medium | High | Medium model - higher accuracy |
| `yolov8l.pt` | ~87MB | Slow | Higher | Large model - best for accuracy |
| `yolov8x.pt` | ~136MB | Slowest | Highest | Extra large model - research/offline use |

## Model Selection Guide

### For Real-Time Applications
- **Webcam detection**: Use `yolov8n.pt`
- **Live stream analysis**: Use `yolov8n.pt` or `yolov8s.pt`
- **Mobile/embedded**: Use `yolov8n.pt`

### For High Accuracy
- **Research analysis**: Use `yolov8l.pt` or `yolov8x.pt`
- **Offline processing**: Use `yolov8l.pt` or `yolov8x.pt`
- **Professional analysis**: Use `yolov8m.pt` or `yolov8l.pt`

### For Balanced Performance
- **General purpose**: Use `yolov8s.pt`
- **YouTube video analysis**: Use `yolov8s.pt` or `yolov8m.pt`
- **Mixed workloads**: Use `yolov8s.pt`

## Automatic Download

Models are automatically downloaded from the official Ultralytics repository when first used. The system will:

1. Check if the model file exists in this directory
2. If not found, download from: `https://github.com/ultralytics/assets/releases/`
3. Cache the model locally for future use

## Manual Download

To manually download models:

```bash
# Download nano model (fastest)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/

# Download small model (balanced)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -P models/

# Download large model (most accurate)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt -P models/
```

## Model Performance

Performance varies based on hardware:

### CPU Performance (Intel i7)
- `yolov8n.pt`: ~50-80 FPS
- `yolov8s.pt`: ~30-50 FPS  
- `yolov8m.pt`: ~15-25 FPS
- `yolov8l.pt`: ~8-15 FPS

### GPU Performance (RTX 3070)
- `yolov8n.pt`: ~200-300 FPS
- `yolov8s.pt`: ~150-200 FPS
- `yolov8m.pt`: ~80-120 FPS  
- `yolov8l.pt`: ~50-80 FPS

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
- Use smaller model (yolov8l.pt → yolov8n.pt)
- Reduce input resolution
- Lower batch size

**Poor detection accuracy:**
- Use larger model (yolov8n.pt → yolov8l.pt)
- Lower confidence threshold
- Ensure good video quality and lighting
