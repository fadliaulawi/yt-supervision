# Logs Directory

This directory contains log files and analysis results from the vehicle detection system.

## Log Files

### `video_detector.log`
Main application log containing:
- System initialization messages
- Model loading status
- Processing statistics
- Error messages and warnings
- Performance metrics

### Detection Statistics Files
JSON files with detailed analysis results:
- `detection_stats_[timestamp].json` - Processing statistics
- Video information and metadata
- Vehicle detection counts by type
- Performance metrics (FPS, processing time)
- Frame-by-frame analysis data

## Screenshot Files

Screenshots saved during analysis:
- `youtube_analysis_[timestamp].jpg` - YouTube video analysis screenshots
- Manual screenshots saved with 's' key during playback

## Log Rotation

Logs are automatically managed:
- Maximum log file size: 10MB
- Old logs are archived when size limit is reached
- Statistics files are saved with timestamps to avoid overwrites

## Log Analysis

### View Recent Activity
```bash
# View last 50 lines of main log
tail -n 50 logs/video_detector.log

# Monitor logs in real-time
tail -f logs/video_detector.log
```

### Search for Errors
```bash
# Find error messages
grep "ERROR" logs/video_detector.log

# Find warnings
grep "WARNING" logs/video_detector.log
```

### Analyze Detection Statistics
Statistics files contain JSON data that can be analyzed:

```python
import json

# Load statistics file
with open('logs/detection_stats_1234567890.json', 'r') as f:
    stats = json.load(f)

print(f"Total vehicles: {stats['total_vehicle_detections']}")
print(f"Average FPS: {stats['average_fps']:.2f}")
print(f"Vehicle types: {stats['vehicle_counts_by_type']}")
```

## Common Log Messages

### Normal Operation
- `Model loaded successfully` - YOLO model initialization
- `Processing completed` - Successful video analysis
- `Stream URL obtained` - YouTube video stream ready
- `Analysis completed` - Finished processing

### Warnings
- `Failed to read frame` - Input source issues
- `Low FPS` - Performance warnings
- `Model download` - Automatic model downloading

### Errors
- `Failed to load model` - Model file issues
- `Failed to open video` - Input source problems
- `CUDA out of memory` - GPU memory issues
- `Invalid YouTube URL` - URL format problems

## Log Levels

The system uses standard Python logging levels:

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about system operation
- **WARNING**: Something unexpected happened but system continues
- **ERROR**: Serious problem that prevented operation
- **CRITICAL**: Very serious error that may cause system to stop

## Configuration

Log settings can be modified in `config.py`:

```python
# Logging Configuration
LOG_LEVEL = "INFO"         # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "logs/video_detector.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
```

## Troubleshooting with Logs

### Performance Issues
Look for:
- FPS measurements in log messages
- Processing time statistics
- GPU/CUDA related messages
- Memory usage warnings

### Detection Problems
Check for:
- Model loading messages
- Confidence threshold settings
- Input video quality issues
- Detection count statistics

### YouTube Issues
Monitor:
- URL validation messages
- Stream extraction status
- Network-related errors
- Video quality selection

## Privacy and Cleanup

### Automatic Cleanup
The system automatically manages disk space:
- Old log files are rotated
- Statistics files older than 30 days can be auto-deleted
- Screenshot files can be automatically cleaned up

### Manual Cleanup
```bash
# Remove old statistics files (Windows)
forfiles /p logs /m *.json /d -30 /c "cmd /c del @path"

# Remove old statistics files (Linux/Mac)
find logs/ -name "*.json" -mtime +30 -delete

# Clear main log file
> logs/video_detector.log
```

### Sensitive Data
- Logs may contain YouTube URLs
- Statistics files include video titles and channel names
- No personal data or credentials are logged
- Screenshots may contain video content

## Integration

### External Log Analysis
Logs can be integrated with external tools:
- **ELK Stack**: For log aggregation and analysis
- **Grafana**: For real-time monitoring dashboards  
- **Splunk**: For enterprise log management
- **Custom scripts**: For automated analysis

### API Access
Statistics files provide programmatic access to analysis results for:
- Research data collection
- Performance monitoring
- Automated reporting
- Integration with other systems
