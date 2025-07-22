#!/usr/bin/env python3
"""
Real-time Vehicle Detection Demo
A simplified script for quick testing of the vehicle detection system.
Includes YouTube video watching and analysis capability.
"""

from modules.video_detector import VideoDetector
from modules.youtube_watcher import YouTubeVideoWatcher
from modules.status_manager import set_analysis_running, set_analysis_completed, set_analysis_error
import argparse
import json
import glob
from datetime import datetime
from pathlib import Path

def get_daily_summary(date=None):
    """Get summary of analyses for today from single daily log file."""
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    
    # Single daily log file
    logs_dir = Path("logs")
    daily_log_file = logs_dir / f"analysis_{date}.json"
    
    # Format date for display
    try:
        date_obj = datetime.strptime(date, "%Y%m%d")
        date_formatted = date_obj.strftime("%A, %B %d, %Y")
    except ValueError:
        date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    
    if not daily_log_file.exists():
        return {
            'date': date,
            'date_formatted': date_formatted,
            'total_analyses': 0,
            'total_detections': 0,
            'total_frames': 0,
            'total_duration': 0.0,
            'analyses': []
        }
    
    try:
        with open(daily_log_file, 'r') as f:
            daily_data = json.load(f)
        
        # Return summary data
        summary = daily_data.get('summary', {})
        return {
            'date': date,
            'date_formatted': summary.get('date_formatted', date_formatted),
            'total_analyses': summary.get('total_analyses', len(daily_data.get('analyses', []))),
            'total_detections': summary.get('total_detections', 0),
            'total_frames': summary.get('total_frames', 0),
            'total_duration': summary.get('total_duration', 0.0),
            'analyses': daily_data.get('analyses', [])
        }
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        return {
            'date': date,
            'date_formatted': date_formatted,
            'total_analyses': 0,
            'total_detections': 0,
            'total_frames': 0,
            'total_duration': 0.0,
            'analyses': []
        }

def get_next_daily_counter():
    """Get the next analysis counter for today."""
    today = datetime.now().strftime("%Y%m%d")
    logs_dir = Path("logs")
    daily_log_file = logs_dir / f"analysis_{today}.json"
    
    if not daily_log_file.exists():
        return 1
    
    try:
        with open(daily_log_file, 'r') as f:
            daily_data = json.load(f)
        return len(daily_data.get('analyses', [])) + 1
    except (json.JSONDecodeError, FileNotFoundError):
        return 1

def show_daily_progress():
    """Display daily analysis progress and summary information."""
    try:
        today_summary = get_daily_summary()
        
        print("=" * 60)
        print("📊 DAILY ANALYSIS TRACKER")
        print("=" * 60)
        print(f"📅 Date: {today_summary['date_formatted']}")
        
        if today_summary['total_analyses'] > 0:
            print(f"🚗 Total detections today: {today_summary['total_detections']:,}")
            print(f"🎬 Total frames processed: {today_summary['total_frames']:,}")
            print(f"⏱️ Total analysis time: {today_summary['total_duration']:.1f}s")
            
            # Show last analysis info
            if today_summary['analyses']:
                last = today_summary['analyses'][-1]
                last_time = last['analysis_start_time'][11:19] if len(last['analysis_start_time']) > 19 else 'Unknown'
                last_mode = last.get('config', {}).get('mode', 'unknown')
                print(f"🕒 Last analysis: {last_time} ({last_mode} mode)")
        else:
            print("🌅 No analyses completed today yet!")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"⚠️ Could not load daily progress: {e}")
        print("🔄 Starting fresh analysis...")

def save_analysis_results(stats, config, start_time=None):
    """Save analysis results to daily log file, appending to existing analyses."""
    try:
        # Use simple daily naming convention: analysis_YYYYMMDD.json
        today = datetime.now().strftime("%Y%m%d")
        log_file = f"logs/analysis_{today}.json"
        
        # Use provided start_time or fall back to current time
        analysis_start = start_time if start_time else datetime.now()
        analysis_end = datetime.now()
        
        # Prepare new analysis data
        new_analysis = {
            'analysis_start_time': analysis_start.isoformat(),
            'analysis_end_time': analysis_end.isoformat(),
            'duration_seconds': stats.get('analysis_duration', 0) or stats.get('total_analysis_time', 0) or (analysis_end - analysis_start).total_seconds(),
            'total_frames': stats.get('total_frames_processed', 0),
            'detection_counts': stats.get('vehicle_counts_by_type', {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}),
            'total_detections': stats.get('total_vehicle_detections', 0),
            'average_fps': stats.get('average_fps', 0),
            'config': config
        }
        
        # Add tracking statistics if available
        if stats.get('tracking_enabled', False):
            new_analysis.update({
                'tracking_enabled': True,
                'unique_vehicles_total': stats.get('unique_vehicles_total', 0),
                'unique_vehicle_counts_by_type': stats.get('unique_vehicle_counts_by_type', {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}),
                'active_tracks': stats.get('active_tracks', 0)
            })
        else:
            new_analysis['tracking_enabled'] = False
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Load existing data or create new structure
        if Path(log_file).exists():
            try:
                with open(log_file, 'r') as f:
                    daily_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                daily_data = {'date': today, 'analyses': [], 'summary': {}}
        else:
            daily_data = {'date': today, 'analyses': [], 'summary': {}}
        
        # Add new analysis
        daily_counter = len(daily_data['analyses']) + 1
        new_analysis['daily_counter'] = daily_counter
        daily_data['analyses'].append(new_analysis)
        
        # Update summary statistics
        total_detections = sum(analysis['total_detections'] for analysis in daily_data['analyses'])
        total_frames = sum(analysis['total_frames'] for analysis in daily_data['analyses'])
        total_duration = sum(analysis['duration_seconds'] for analysis in daily_data['analyses'])
        
        # Combine detection counts
        combined_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        combined_unique_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        total_unique_vehicles = 0
        tracking_analyses = 0
        
        for analysis in daily_data['analyses']:
            # Regular detection counts
            for vehicle_type in combined_counts:
                combined_counts[vehicle_type] += analysis['detection_counts'].get(vehicle_type, 0)
            
            # Unique tracking counts if available
            if analysis.get('tracking_enabled', False):
                tracking_analyses += 1
                unique_counts = analysis.get('unique_vehicle_counts_by_type', {})
                for vehicle_type in combined_unique_counts:
                    combined_unique_counts[vehicle_type] += unique_counts.get(vehicle_type, 0)
                total_unique_vehicles += analysis.get('unique_vehicles_total', 0)
        
        daily_summary = {
            'date_formatted': datetime.strptime(today, "%Y%m%d").strftime("%A, %B %d, %Y"),
            'total_analyses': len(daily_data['analyses']),
            'total_detections': total_detections,
            'total_frames': total_frames,
            'total_duration': total_duration,
            'combined_detection_counts': combined_counts,
            'last_updated': analysis_end.isoformat()
        }
        
        # Add tracking summary if any analysis used tracking
        if tracking_analyses > 0:
            daily_summary.update({
                'tracking_enabled_analyses': tracking_analyses,
                'total_unique_vehicles': total_unique_vehicles,
                'combined_unique_counts': combined_unique_counts
            })
        
        daily_data['summary'] = daily_summary
        
        # Save updated daily data
        with open(log_file, 'w') as f:
            json.dump(daily_data, f, indent=2)
        
        # Update status to completed
        set_analysis_completed(log_file=log_file)
        
        print(f"📊 Results saved to {log_file}")
        
        return daily_data
        
    except Exception as e:
        print(f"Failed to save results: {e}")
        set_analysis_error(str(e))
        return None

def demo_webcam():
    """Demo function for webcam detection."""
    print("🎥 Starting webcam vehicle detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Show daily progress
    show_daily_progress()
    
    set_analysis_running(mode='webcam', model='yolo11l.pt')
    start_time = datetime.now()
    
    try:
        detector = VideoDetector(confidence_threshold=0.7)
        stats = detector.process_video(source="0", display=True)
        
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
    
    # Show daily progress
    show_daily_progress()
    
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
    
    # Show daily progress
    show_daily_progress()
    
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
