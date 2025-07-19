#!/usr/bin/env python3
"""
Real-time Vehicle Detection Demo
A simplified simplified demo script for quick testing of the vehicle detection system.
Includes YouTube video watching and analysis capability.
"""

from modules.video_detector import VideoDetector
from modules.youtube_watcher import YouTubeVideoWatcher
from modules.status_manager import set_analysis_running, set_analysis_completed, set_analysis_error
import argparse
import json
from datetime import datetime
from pathlib import Path

def save_analysis_results(stats, config, start_time=None):
    """Save analysis results to JSON file for dashboard integration."""
    try:
        Path("logs").mkdir(exist_ok=True)
        
        # Use provided start_time or fall back to current time
        analysis_start = start_time if start_time else datetime.now()
        analysis_end = datetime.now()
        
        # Extract relevant data from stats
        results = {
            'analysis_start_time': analysis_start.isoformat(),
            'analysis_end_time': analysis_end.isoformat(),
            'duration_seconds': stats.get('analysis_duration', 0) or stats.get('total_analysis_time', 0) or (analysis_end - analysis_start).total_seconds(),
            'total_frames': stats.get('total_frames_processed', 0),
            'detection_counts': stats.get('vehicle_counts_by_type', {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}),
            'total_detections': stats.get('total_vehicle_detections', 0),
            'average_fps': stats.get('average_fps', 0),
            'config': config
        }
        
        # Save to timestamped file with readable timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/analysis_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update status to completed
        set_analysis_completed(log_file=log_file)
        
        print(f"📊 Results saved to {log_file}")
        
    except Exception as e:
        print(f"Failed to save results: {e}")
        set_analysis_error(str(e))

def demo_webcam():
    """Demo function for webcam detection."""
    print("Starting webcam vehicle detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    set_analysis_running(mode='webcam', model='yolo11l.pt')
    start_time = datetime.now()
    
    try:
        detector = VideoDetector(confidence_threshold=0.7)
        stats = detector.process_video(source="0", display=True)
        
        print(f"\n✅ Demo completed! Processed {stats['total_frames_processed']} frames")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Total vehicles detected: {stats['total_vehicle_detections']}")
        
        # Save results for dashboard integration
        save_analysis_results(stats, {
            'mode': 'webcam',
            'model_path': 'models/yolov8n.pt',
            'confidence': 0.7
        }, start_time)
        
    except Exception as e:
        print(f"❌ Webcam analysis failed: {e}")
        set_analysis_error(str(e))

def demo_video_file(video_path: str, output_path: str = None):
    """Demo function for video file detection."""
    print(f"📹 Processing video file: {video_path}")
    
    set_analysis_running(mode='file', video_path=video_path, model='yolo11l.pt')
    start_time = datetime.now()
    
    try:
        detector = VideoDetector(confidence_threshold=0.7)
        stats = detector.process_video(
            source=video_path, 
            output_path=output_path,
            display=True
        )
        
        print(f"\n✅ Processing completed!")
        print(f"Processed {stats['total_frames_processed']} frames")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Total vehicles detected: {stats['total_vehicle_detections']}")
        print(f"Vehicle breakdown: {stats['vehicle_counts_by_type']}")
        
        # Save results for dashboard integration
        save_analysis_results(stats, {
            'mode': 'file',
            'video_path': video_path,
            'output_path': output_path,
            'model_path': 'models/yolov8n.pt',
            'confidence': 0.7
        }, start_time)
        
    except Exception as e:
        print(f"❌ Video file analysis failed: {e}")
        set_analysis_error(str(e))

def demo_youtube_watch(youtube_url: str, quality: str = "720p", max_duration: int = None):
    """Demo function for watching and analyzing YouTube videos."""
    print("🎬 Starting YouTube video analysis...")
    print("Press 'q' to stop, 's' to save screenshot")
    
    set_analysis_running(mode='youtube', url=youtube_url, quality=quality, model='yolo11l.pt')
    start_time = datetime.now()
    
    try:
        watcher = YouTubeVideoWatcher(
            confidence_threshold=0.7,
            model_path="models/yolo11l.pt",
            quality=quality
        )
        
        print(f"📺 Analyzing YouTube video/stream...")
        print(f"   URL: {youtube_url}")
        print(f"   Quality: {quality}")
        print(f"   Model: YOLOv11l (high accuracy)")
        if max_duration:
            print(f"   Max duration: {max_duration} seconds")
        
        stats = watcher.watch_and_analyze(
            youtube_url=youtube_url,
            display=True,
            max_duration=max_duration
        )
        
        print(f"\n✅ Analysis completed!")
        print(f"Video: {stats['video_info'].get('title', 'Unknown')}")
        print(f"Channel: {stats['video_info'].get('uploader', 'Unknown')}")
        print(f"Processed {stats['total_frames_processed']} frames")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Total vehicles detected: {stats['total_vehicle_detections']}")
        print(f"Vehicle breakdown: {stats['vehicle_counts_by_type']}")
        
        # Save results for dashboard integration
        save_analysis_results(stats, {
            'mode': 'youtube',
            'url': youtube_url,
            'quality': quality,
            'max_duration': max_duration,
            'model_path': 'models/yolo11l.pt',
            'confidence': 0.7
        }, start_time)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Make sure you have:")
        print("  - Valid YouTube URL")
        print("  - yt-dlp installed (pip install yt-dlp)")
        print("  - Stable internet connection")
        
        # Update status to error
        set_analysis_error(str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detection Demo")
    parser.add_argument('--video', '-v', type=str, help='Path to video file')
    parser.add_argument('--output', '-o', type=str, help='Output video path')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam')
    parser.add_argument('--youtube-watch', type=str, help='YouTube video/live URL (for watching)')
    parser.add_argument('--quality', '-q', type=str, default='720p',
                       choices=['480p', '720p', '1080p', 'best'],
                       help='Video quality for YouTube watching')
    parser.add_argument('--duration', '-d', type=int,
                       help='Duration limit in minutes (converted to seconds for YouTube watching)')
    
    args = parser.parse_args()
    
    if args.youtube_watch:
        # Convert duration from minutes to seconds for consistency
        max_duration = args.duration * 60 if args.duration else None
        demo_youtube_watch(args.youtube_watch, args.quality, max_duration)
    elif args.webcam:
        demo_webcam()
    elif args.video:
        demo_video_file(args.video, args.output)
    else:
        print("Please specify one of the following options:")
        print("\n🎥 LOCAL OPTIONS:")
        print("  --webcam                         Use webcam for local detection")
        print("  --video <path>                   Process video file")
        print("\n📺 YOUTUBE WATCHING:")
        print("  --youtube-watch <url>            Watch & analyze YouTube video/live stream")
        print("\n📺 YouTube Watching Examples:")
        print("  python demo.py --youtube-watch 'https://www.youtube.com/watch?v=VIDEO_ID'")
        print("  python demo.py --youtube-watch 'https://youtu.be/VIDEO_ID' --quality 1080p")
        print("  python demo.py --youtube-watch 'https://youtube.com/live/LIVE_ID' --duration 10")
        print("\n Tips:")
        print("  - Use 720p quality for good balance of speed/quality")
        print("  - Duration is in minutes, converted to seconds for analysis")
        print("  - Live streams can be analyzed in real-time")
        print("  - Press 'q' to quit, 's' to save screenshots")
