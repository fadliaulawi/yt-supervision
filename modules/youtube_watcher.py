#!/usr/bin/env python3
"""
YouTube Video Watcher and Analyzer
Watch and analyze YouTube videos/streams for vehicle detection from viewer perspective.
"""

import cv2
import yt_dlp
import time
import logging
import re
from typing import Optional, Dict
from .video_detector import VideoDetector

def safe_log_message(message: str) -> str:
    """
    Safely encode a message for logging, removing or replacing problematic Unicode characters.
    """
    try:
        # Try to encode to Windows-1252 (cp1252) to test compatibility
        message.encode('cp1252')
        return message
    except UnicodeEncodeError:
        # Replace problematic Unicode characters with safe alternatives
        safe_replacements = {
            '🔴': '[LIVE]',
            '📺': '[VIDEO]',
            '🎬': '[MOVIE]',
            '🚗': '[CAR]',
            '🏃': '[RUNNING]',
            '📊': '[CHART]',
            '💾': '[SAVE]',
            '⚡': '[FAST]',
            '🎯': '[TARGET]',
            '🎨': '[ART]',
        }
        
        safe_message = message
        for unicode_char, replacement in safe_replacements.items():
            safe_message = safe_message.replace(unicode_char, replacement)
        
        # Remove any remaining non-ASCII characters as fallback
        safe_message = safe_message.encode('ascii', errors='ignore').decode('ascii')
        
        return safe_message

class YouTubeVideoWatcher:
    """
    Watch and analyze YouTube videos/live streams for vehicle detection.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 model_path: str = "models/yolo11n.pt",
                 quality: str = "720p"):
        """
        Initialize YouTube video watcher.
        
        Args:
            confidence_threshold: Detection confidence threshold
            model_path: YOLO model path
            quality: Video quality preference (480p, 720p, 1080p, best)
        """
        self.detector = VideoDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.quality = quality
        self.logger = logging.getLogger(__name__)
        
        # Setup yt-dlp options
        self.ydl_opts = {
            'format': self._get_format_selector(),
            'quiet': True,
            'no_warnings': True,
            'extractaudio': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
    
    def _get_format_selector(self):
        """Get format selector based on quality preference."""
        if self.quality == "best":
            return "best[ext=mp4]"
        elif self.quality == "1080p":
            return "best[height<=1080][ext=mp4]/best[ext=mp4]"
        elif self.quality == "720p":
            return "best[height<=720][ext=mp4]/best[ext=mp4]"
        elif self.quality == "480p":
            return "best[height<=480][ext=mp4]/best[ext=mp4]"
        else:
            return "best[ext=mp4]"
    
    def is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtube\.com/live/[\w-]+',
            r'https?://youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/c/[\w-]+/live',
            r'https?://(?:www\.)?youtube\.com/@[\w-]+/live'
        ]
        
        return any(re.match(pattern, url) for pattern in youtube_patterns)
    
    def get_video_info(self, url: str) -> Dict:
        """
        Get video information from YouTube URL.
        
        Args:
            url: YouTube video/stream URL
            
        Returns:
            Dictionary with video information
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'is_live': info.get('is_live', False),
                    'description': info.get('description', ''),
                    'url': info.get('url', url),
                    'thumbnail': info.get('thumbnail', ''),
                    'upload_date': info.get('upload_date', ''),
                    'formats': len(info.get('formats', []))
                }
        except Exception as e:
            self.logger.error(f"Failed to extract video info: {e}")
            return {}
    
    def get_stream_url(self, youtube_url: str) -> Optional[str]:
        """
        Get direct stream URL from YouTube URL.
        
        Args:
            youtube_url: YouTube video/stream URL
            
        Returns:
            Direct stream URL or None if failed
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Get the best format URL
                if 'url' in info:
                    return info['url']
                elif 'formats' in info and info['formats']:
                    # Find the best format
                    formats = info['formats']
                    for fmt in formats:
                        if fmt.get('ext') == 'mp4' and fmt.get('url'):
                            return fmt['url']
                    
                    # Fallback to first available format
                    if formats[0].get('url'):
                        return formats[0]['url']
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to extract stream URL: {e}")
            return None
    
    def watch_and_analyze(self, 
                         youtube_url: str,
                         output_path: Optional[str] = None,
                         display: bool = True,
                         max_duration: Optional[int] = None) -> Dict:
        """
        Watch and analyze YouTube video/stream.
        
        Args:
            youtube_url: YouTube video/stream URL
            output_path: Optional output video path
            display: Whether to display video in real-time
            max_duration: Maximum duration in seconds (None for unlimited)
            
        Returns:
            Analysis statistics
        """
        if not self.is_valid_youtube_url(youtube_url):
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")
        
        # Get video information
        self.logger.info("Extracting video information...")
        video_info = self.get_video_info(youtube_url)
        
        if not video_info:
            raise ValueError("Failed to extract video information")
        
        # Use safe logging for Unicode-containing titles
        title = video_info.get('title', 'Unknown')
        uploader = video_info.get('uploader', 'Unknown')
        
        self.logger.info(f"Title: {safe_log_message(title)}")
        self.logger.info(f"Uploader: {safe_log_message(uploader)}")
        self.logger.info(f"Live stream: {video_info.get('is_live', False)}")
        
        # Get stream URL
        self.logger.info("Getting stream URL...")
        stream_url = self.get_stream_url(youtube_url)
        
        if not stream_url:
            raise ValueError("Failed to get stream URL")
        
        self.logger.info(f"Stream URL obtained, starting analysis...")
        
        # Open video stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError("Failed to open video stream")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Stream properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Analysis loop
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of stream reached")
                    break
                
                frame_count += 1
                
                # Detect vehicles
                detections = self.detector.detect_vehicles(frame)
                
                # Draw detections
                annotated_frame = self.detector.draw_detections(frame, detections)
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Add YouTube info overlay
                overlay_frame = self.add_youtube_overlay(
                    annotated_frame, detections, current_fps, video_info
                )
                
                # Display frame
                if display:
                    cv2.imshow(f'YouTube Analysis: {video_info.get("title", "Unknown")[:50]}', 
                             overlay_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Analysis stopped by user")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"logs/youtube_analysis_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, overlay_frame)
                        self.logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Write to output video
                if out:
                    out.write(overlay_frame)
                
                # Update statistics
                self.detector.update_statistics(detections)
                
                # Check duration limit
                if max_duration and elapsed_time > max_duration:
                    self.logger.info(f"Duration limit reached ({max_duration}s)")
                    break
                
                # Progress logging for live streams
                if frame_count % 300 == 0:  # Every 10 seconds at 30fps
                    total_detections = sum(self.detector.detection_stats['vehicle_counts'].values())
                    elapsed_formatted = self._format_time(elapsed_time)
                    self.logger.info(f"Processed {frame_count} frames in {elapsed_formatted}, "
                                   f"FPS: {current_fps:.1f}, "
                                   f"Total vehicles: {total_detections}")
        
        except KeyboardInterrupt:
            self.logger.info("Analysis interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        stats = {
            'video_info': video_info,
            'total_frames_processed': frame_count,
            'total_analysis_time': total_time,
            'time_elapsed_formatted': self._format_time(total_time),
            'average_fps': avg_fps,
            'total_vehicle_detections': sum(self.detector.detection_stats['vehicle_counts'].values()),
            'vehicle_counts_by_type': self.detector.detection_stats['vehicle_counts'].copy(),
            'avg_detections_per_frame': (
                sum(self.detector.detection_stats['vehicle_counts'].values()) / frame_count 
                if frame_count > 0 else 0
            ),
            'analysis_start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'analysis_end_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_rate': frame_count / 60 if total_time > 60 else frame_count  # frames per minute or total frames
        }
        
        self.logger.info("Analysis completed:")
        self.logger.info(f"  Video: {safe_log_message(video_info.get('title', 'Unknown'))}")
        self.logger.info(f"  Time elapsed: {stats['time_elapsed_formatted']}")
        self.logger.info(f"  Frames processed: {stats['total_frames_processed']}")
        self.logger.info(f"  Average FPS: {stats['average_fps']:.2f}")
        self.logger.info(f"  Total vehicles detected: {stats['total_vehicle_detections']}")
        self.logger.info(f"  Vehicle breakdown: {stats['vehicle_counts_by_type']}")
        
        return stats
    
    def add_youtube_overlay(self, frame, detections, fps, video_info):
        """
        Add YouTube-specific overlay to the frame.
        
        Args:
            frame: Input frame
            detections: Current frame detections
            fps: Current FPS
            video_info: YouTube video information
            
        Returns:
            Frame with YouTube overlay
        """
        height, width = frame.shape[:2]
        
        # Count vehicles by type
        vehicle_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Top banner for YouTube info
        banner_height = 90
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
        
        # Bottom info bar
        info_height = 70
        cv2.rectangle(overlay, (0, height - info_height), (width, height), (0, 0, 0), -1)
        
        # Blend overlay
        alpha = 0.75
        frame = cv2.addWeighted(frame, alpha, overlay, 1 - alpha, 0)
        
        # YouTube video title
        title = video_info.get('title', 'Unknown Video')[:60] + '...' if len(video_info.get('title', '')) > 60 else video_info.get('title', 'Unknown')
        cv2.putText(frame, f"📺 {title}", (20, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Channel name and live indicator
        uploader = video_info.get('uploader', 'Unknown Channel')
        is_live = video_info.get('is_live', False)
        live_text = " 🔴 LIVE" if is_live else ""
        cv2.putText(frame, f"by {uploader}{live_text}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Analysis info
        cv2.putText(frame, "🚗 AI Vehicle Analysis", (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Bottom info bar
        info_y_base = height - 45
        
        # Vehicle count
        total_vehicles = len(detections)
        cv2.putText(frame, f"Vehicles: {total_vehicles}", (20, info_y_base), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Vehicle breakdown
        x_offset = 130
        for vehicle_type, count in vehicle_counts.items():
            color = self.detector.colors.get(vehicle_type, (255, 255, 255))
            text = f"{vehicle_type.title()}: {count}"
            cv2.putText(frame, text, (x_offset, info_y_base), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            x_offset += text_width + 20
        
        # FPS and timestamp
        fps_text = f"FPS: {fps:.1f}"
        time_text = time.strftime("%H:%M:%S")
        
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        cv2.putText(frame, fps_text, (width - fps_size[0] - 20, info_y_base), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, time_text, (width - time_size[0] - fps_size[0] - 40, info_y_base - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def _format_time(self, seconds: float) -> str:
        """
        Format time duration in human-readable format.
        
        Args:
            seconds: Time duration in seconds
            
        Returns:
            Formatted time string (e.g., "1h 23m 45s" or "2m 30s" or "45s")
        """
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch and Analyze YouTube Videos for Vehicle Detection")
    parser.add_argument('url', type=str, help='YouTube video/stream URL')
    parser.add_argument('--confidence', '-c', type=float, default=0.6,
                       help='Detection confidence threshold')
    parser.add_argument('--model', '-m', type=str, default='models/yolo11n.pt',
                       help='YOLO model path')
    parser.add_argument('--quality', '-q', type=str, default='720p',
                       choices=['480p', '720p', '1080p', 'best'],
                       help='Video quality preference')
    parser.add_argument('--output', '-o', type=str,
                       help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--max-duration', '-d', type=int,
                       help='Maximum analysis duration in seconds')
    
    args = parser.parse_args()
    
    # Setup logging with UTF-8 encoding for Unicode support
    import os
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/video_detector.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Initialize watcher
        watcher = YouTubeVideoWatcher(
            confidence_threshold=args.confidence,
            model_path=args.model,
            quality=args.quality
        )
        
        print(f"🎬 Starting YouTube video analysis...")
        print(f"   URL: {args.url}")
        print(f"   Quality: {args.quality}")
        print(f"   Model: {args.model}")
        print(f"   Confidence: {args.confidence}")
        if args.max_duration:
            print(f"   Max duration: {args.max_duration}s")
        print(f"\nPress 'q' to stop, 's' to save screenshot...")
        
        # Start analysis
        stats = watcher.watch_and_analyze(
            youtube_url=args.url,
            output_path=args.output,
            display=not args.no_display,
            max_duration=args.max_duration
        )
        
        print(f"\n✅ Analysis completed!")
        print(f"   Video: {stats['video_info'].get('title', 'Unknown')}")
        print(f"   Time elapsed: {stats['time_elapsed_formatted']}")
        print(f"   Frames processed: {stats['total_frames_processed']}")
        print(f"   Average FPS: {stats['average_fps']:.2f}")
        print(f"   Total vehicles detected: {stats['total_vehicle_detections']}")
        print(f"   Vehicle breakdown: {stats['vehicle_counts_by_type']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Analysis error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
