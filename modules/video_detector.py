"""
Video Detection Module
Advanced vehicle detection system using YOLO models with comprehensive
video processing capabilities including real-time detection, file processing,
and performance benchmarking.
"""

import argparse
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import URLError

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import Config

class VideoDetector:
    """
    Advanced vehicle detection system using YOLO models.
    
    Features:
    - Real-time detection from webcam, video files, or streams
    - GPU acceleration with automatic fallback to CPU
    - Automatic model downloading and fallback handling
    - Comprehensive statistics tracking and benchmarking
    - Configurable detection parameters and output options
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 confidence_threshold: float = None):
        """
        Initialize the VideoDetector with configurable parameters.
        
        Args:
            model_path: Path to YOLO model file (defaults to Config.DEFAULT_MODEL)
            confidence_threshold: Detection confidence threshold (defaults to Config.CONFIDENCE_THRESHOLD)
        """
        # Use config defaults if not specified
        self.model_path = model_path or Config.DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        self.model = None
        
        # Load configuration from Config class
        self.vehicle_classes = list(Config.VEHICLE_CLASSES.keys())
        self.class_names = Config.VEHICLE_CLASSES
        self.colors = Config.DETECTION_COLORS
        
        # Initialize components
        self.setup_device()
        self.setup_logging()
        self.initialize_statistics()
        
        # Load model
        self.ensure_model_available()
        self.load_model()

    def setup_device(self):
        """Configure GPU/CPU device for optimal performance."""
        if Config.USE_GPU and torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU acceleration enabled: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self.device = 'cpu'
            if Config.USE_GPU:
                print("⚠️ GPU requested but not available - falling back to CPU")
                print("💡 Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("🖥️ Using CPU for inference")

    def setup_logging(self):
        """Initialize logging with UTF-8 encoding for Unicode support."""
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_statistics(self):
        """Initialize detection statistics tracking."""
        self.detection_stats = {
            'total_frames': 0,
            'detections_per_frame': [],
            'processing_times': [],
            'vehicle_counts': {cls: 0 for cls in self.class_names.values()}
        }

    def ensure_model_available(self):
        """
        Ensure model file exists with intelligent fallback handling.
        Downloads models automatically if not available locally.
        """
        model_file = Path(self.model_path)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        
        if model_file.exists():
            self.logger.info(f"Model found: {self.model_path}")
            return
        
        self.logger.info(f"Model not found: {self.model_path}")
        
        # Attempt to download the requested model
        if self._download_model(self.model_path):
            self.logger.info(f"Successfully downloaded: {self.model_path}")
            return
        
        # Try fallback to YOLOv8 equivalent
        fallback_model = self._get_fallback_model(self.model_path)
        if fallback_model and self._download_model(fallback_model):
            self.logger.info(f"Using fallback model: {fallback_model}")
            self.model_path = fallback_model
            return
        
        # Final fallback - rely on YOLO auto-download
        self.logger.warning("Will rely on YOLO auto-download functionality")

    def _get_fallback_model(self, model_path: str) -> Optional[str]:
        """Get fallback model path for YOLOv11 -> YOLOv8 conversion."""
        fallback_path = model_path.replace("yolo11", "yolov8")
        return fallback_path if fallback_path != model_path else None

    def _download_model(self, model_path: str) -> bool:
        """
        Download YOLO model with progress indication.
        
        Args:
            model_path: Local path to save the model
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            model_name = Path(model_path).name
            download_url = self._get_download_url(model_name)
            
            if not download_url:
                self.logger.error(f"Unknown model format: {model_name}")
                return False
            
            self.logger.info(f"Downloading {model_name}...")
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0 and block_num % 50 == 0:
                    percent = min(100, (block_num * block_size / total_size) * 100)
                    print(f"\rDownloading {model_name}: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(download_url, model_path, reporthook=show_progress)
            print(f"\n✅ Successfully downloaded {model_name}")
            
            # Verify download integrity
            return self._verify_downloaded_model(model_path)
                
        except URLError as e:
            self.logger.error(f"Network error downloading {model_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            return False

    def _get_download_url(self, model_name: str) -> Optional[str]:
        """Get the appropriate download URL for a model."""
        base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
        
        # Support for different YOLO versions
        supported_prefixes = ["yolo11", "yolov10", "yolov9", "yolov8"]
        
        for prefix in supported_prefixes:
            if model_name.startswith(prefix):
                return f"{base_url}/{model_name}"
        
        return None

    def _verify_downloaded_model(self, model_path: str) -> bool:
        """Verify that the downloaded model file is valid."""
        try:
            file_size = Path(model_path).stat().st_size
            if file_size < 1_000_000:  # Less than 1MB is suspicious
                self.logger.error(f"Downloaded file seems too small: {file_size} bytes")
                Path(model_path).unlink(missing_ok=True)
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error verifying downloaded model: {e}")
            return False
    
    def load_model(self):
        """Load and configure the YOLO model."""
        try:
            self.logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Configure device
            if hasattr(self, 'device'):
                self.model.to(self.device)
                self.logger.info(f"Model loaded successfully on {self.device}")
            else:
                self.logger.info("Model loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame with performance tracking.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        start_time = time.time()
        
        # Run YOLO inference with configured parameters
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = self._parse_detection_results(results)
        
        # Track performance
        processing_time = time.time() - start_time
        self.detection_stats['processing_times'].append(processing_time)
        
        return detections

    def _parse_detection_results(self, results) -> List[Dict]:
        """Parse YOLO detection results into standardized format."""
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                
                # Filter for vehicle classes only
                if class_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self.class_names[class_id]
                    }
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels with consistent styling.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame with detection overlays
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            self._draw_single_detection(annotated_frame, detection)
        
        return annotated_frame

    def _draw_single_detection(self, frame: np.ndarray, detection: Dict):
        """Draw a single detection box and label."""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        color = self.colors.get(class_name, self.colors['default'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, Config.BOX_THICKNESS)
        
        # Draw label with background
        label = f"{class_name} {confidence:.2f}"
        self._draw_label_with_background(frame, label, (x1, y1), color)

    def _draw_label_with_background(self, frame: np.ndarray, label: str, position: tuple, color: tuple):
        """Draw text label with contrasting background."""
        x, y = position
        
        # Calculate label dimensions
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, Config.FONT_THICKNESS
        )
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x, y - label_height - Config.LABEL_PADDING),
            (x + label_width, y),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            Config.FONT_SCALE,
            (255, 255, 255),  # White text
            Config.FONT_THICKNESS
        )
    
    def add_info_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float) -> np.ndarray:
        """
        Add comprehensive information overlay to the frame.
        
        Args:
            frame: Input frame
            detections: Current frame detections
            fps: Current FPS
            
        Returns:
            Frame with information overlay
        """
        # Count vehicles by type
        vehicle_counts = self._count_vehicles_by_type(detections)
        
        # Prepare information lines
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Total Vehicles: {len(detections)}",
            *[f"{vehicle_type.capitalize()}s: {count}" 
              for vehicle_type, count in vehicle_counts.items()]
        ]
        
        self._draw_info_panel(frame, info_lines)
        return frame

    def _count_vehicles_by_type(self, detections: List[Dict]) -> Dict[str, int]:
        """Count detections by vehicle type."""
        vehicle_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
        return vehicle_counts

    def _draw_info_panel(self, frame: np.ndarray, info_lines: List[str]):
        """Draw information panel with background."""
        panel_height = len(info_lines) * 25 + 10
        panel_width = 250
        
        # Draw panel background
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
        
        # Draw information text
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(
                frame, line, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE,
                (255, 255, 255), Config.FONT_THICKNESS
            )

    def update_statistics(self, detections: List[Dict]):
        """Update comprehensive detection statistics."""
        self.detection_stats['total_frames'] += 1
        self.detection_stats['detections_per_frame'].append(len(detections))
        
        # Update vehicle counts
        for detection in detections:
            class_name = detection['class_name']
            self.detection_stats['vehicle_counts'][class_name] += 1
    
    def process_video(self, 
                     source: str, 
                     output_path: Optional[str] = None,
                     display: bool = True) -> Dict:
        """
        Process video from various sources with comprehensive options.
        
        Args:
            source: Video source (file path, webcam index, or stream URL)
            output_path: Optional output video path
            display: Whether to display video in real-time
            
        Returns:
            Dictionary with comprehensive processing statistics
        """
        # Initialize video capture
        cap = self._initialize_video_source(source)
        
        # Get video properties and setup output writer
        video_info = self._get_video_properties(cap, source)
        out = self._setup_output_writer(output_path, video_info) if output_path else None
        
        # Process video frames
        try:
            processing_stats = self._process_video_frames(cap, out, display, video_info)
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            processing_stats = self._get_current_stats()
        finally:
            self._cleanup_resources(cap, out, display)
        
        # Generate final statistics
        final_stats = self._generate_final_statistics(processing_stats)
        
        self._log_processing_summary(final_stats)
        return final_stats

    def _initialize_video_source(self, source: str) -> cv2.VideoCapture:
        """Initialize video capture from source."""
        if source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        return cap

    def _get_video_properties(self, cap: cv2.VideoCapture, source) -> Dict:
        """Extract video properties for processing configuration."""
        return {
            'fps': cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_OUTPUT_FPS,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else -1
        }

    def _setup_output_writer(self, output_path: str, video_info: Dict) -> cv2.VideoWriter:
        """Setup video writer for output recording."""
        fourcc = cv2.VideoWriter_fourcc(*Config.OUTPUT_CODEC)
        return cv2.VideoWriter(
            output_path, fourcc, video_info['fps'],
            (video_info['width'], video_info['height'])
        )

    def _process_video_frames(self, cap: cv2.VideoCapture, out: Optional[cv2.VideoWriter], 
                             display: bool, video_info: Dict) -> Dict:
        """Process video frames with detection and annotation."""
        frame_count = 0
        start_time = time.time()
        
        self.logger.info(f"Video properties: {video_info['width']}x{video_info['height']} @ {video_info['fps']} FPS")
        if video_info['total_frames'] > 0:
            self.logger.info(f"Total frames: {video_info['total_frames']}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect and annotate
            detections = self.detect_vehicles(frame)
            annotated_frame = self.draw_detections(frame, detections)
            
            # Add information overlay
            current_fps = frame_count / (time.time() - start_time)
            annotated_frame = self.add_info_overlay(annotated_frame, detections, current_fps)
            
            # Update statistics
            self.update_statistics(detections)
            
            # Handle display and user input
            if display and self._handle_frame_display(annotated_frame, bool(out)):
                break
            
            # Write to output video
            if out:
                out.write(annotated_frame)
            
            # Progress logging
            self._log_progress(frame_count, video_info['total_frames'])
        
        return {
            'frames_processed': frame_count,
            'processing_time': time.time() - start_time,
            'start_time': start_time
        }

    def _handle_frame_display(self, frame: np.ndarray, has_output_writer: bool) -> bool:
        """Handle frame display and user input. Returns True if should quit."""
        cv2.imshow('Vehicle Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return True
        elif key == ord('s') and not has_output_writer:
            self._save_screenshot(frame)
        
        return False

    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot."""
        screenshot_path = f"detection_screenshot_{int(time.time())}.jpg"
        cv2.imwrite(screenshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, Config.SCREENSHOT_QUALITY])
        self.logger.info(f"Screenshot saved: {screenshot_path}")

    def _log_progress(self, frame_count: int, total_frames: int):
        """Log processing progress periodically."""
        if total_frames > 0 and frame_count % Config.STATS_UPDATE_INTERVAL == 0:
            progress = (frame_count / total_frames) * 100
            self.logger.info(f"Processing: {frame_count}/{total_frames} ({progress:.1f}%)")

    def _cleanup_resources(self, cap: cv2.VideoCapture, out: Optional[cv2.VideoWriter], display: bool):
        """Clean up video capture and display resources."""
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()

    def _get_current_stats(self) -> Dict:
        """Get current processing statistics."""
        return {
            'frames_processed': self.detection_stats['total_frames'],
            'processing_time': 0,
            'start_time': time.time()
        }

    def _generate_final_statistics(self, processing_stats: Dict) -> Dict:
        """Generate comprehensive final statistics."""
        total_time = processing_stats['processing_time']
        frames_processed = processing_stats['frames_processed']
        
        return {
            'total_frames_processed': frames_processed,
            'total_processing_time': total_time,
            'average_fps': frames_processed / total_time if total_time > 0 else 0,
            'average_detection_time': np.mean(self.detection_stats['processing_times']) if self.detection_stats['processing_times'] else 0,
            'total_vehicle_detections': sum(self.detection_stats['vehicle_counts'].values()),
            'vehicle_counts_by_type': self.detection_stats['vehicle_counts'].copy(),
            'avg_detections_per_frame': np.mean(self.detection_stats['detections_per_frame']) if self.detection_stats['detections_per_frame'] else 0
        }

    def _log_processing_summary(self, stats: Dict):
        """Log comprehensive processing summary."""
        self.logger.info("Processing completed:")
        self.logger.info(f"  Frames processed: {stats['total_frames_processed']}")
        self.logger.info(f"  Average FPS: {stats['average_fps']:.2f}")
        self.logger.info(f"  Total vehicles detected: {stats['total_vehicle_detections']}")
        self.logger.info(f"  Vehicle breakdown: {stats['vehicle_counts_by_type']}")
    
    def benchmark_model(self, test_video: str, num_frames: int = 100) -> Dict:
        """
        Comprehensive model performance benchmarking.
        
        Args:
            test_video: Path to test video file
            num_frames: Number of frames to process for benchmarking
            
        Returns:
            Detailed benchmark results including timing and detection metrics
        """
        self.logger.info(f"Benchmarking model on {num_frames} frames from {test_video}")
        
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            raise ValueError(f"Failed to open test video: {test_video}")
        
        processing_times = []
        detection_counts = []
        
        try:
            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Could only process {i} frames (video ended)")
                    break
                
                # Time the detection process
                start_time = time.time()
                detections = self.detect_vehicles(frame)
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
                detection_counts.append(len(detections))
        
        finally:
            cap.release()
        
        if not processing_times:
            raise ValueError("No frames could be processed for benchmarking")
        
        # Calculate comprehensive benchmark metrics
        benchmark_results = self._calculate_benchmark_metrics(processing_times, detection_counts)
        self._log_benchmark_results(benchmark_results)
        
        return benchmark_results

    def _calculate_benchmark_metrics(self, processing_times: List[float], detection_counts: List[int]) -> Dict:
        """Calculate comprehensive benchmark metrics."""
        processing_times = np.array(processing_times)
        detection_counts = np.array(detection_counts)
        
        return {
            'frames_processed': len(processing_times),
            'avg_processing_time_ms': np.mean(processing_times) * 1000,
            'min_processing_time_ms': np.min(processing_times) * 1000,
            'max_processing_time_ms': np.max(processing_times) * 1000,
            'std_processing_time_ms': np.std(processing_times) * 1000,
            'avg_fps': 1.0 / np.mean(processing_times),
            'max_fps': 1.0 / np.min(processing_times),
            'min_fps': 1.0 / np.max(processing_times),
            'avg_detections_per_frame': np.mean(detection_counts),
            'max_detections_per_frame': np.max(detection_counts),
            'min_detections_per_frame': np.min(detection_counts),
            'total_detections': np.sum(detection_counts),
            'device': self.device,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold
        }

    def _log_benchmark_results(self, results: Dict):
        """Log detailed benchmark results."""
        self.logger.info("=" * 50)
        self.logger.info("BENCHMARK RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {results['model_path']}")
        self.logger.info(f"Device: {results['device']}")
        self.logger.info(f"Confidence Threshold: {results['confidence_threshold']}")
        self.logger.info(f"Frames Processed: {results['frames_processed']}")
        self.logger.info("-" * 25)
        self.logger.info("Performance Metrics:")
        self.logger.info(f"  Average FPS: {results['avg_fps']:.2f}")
        self.logger.info(f"  FPS Range: {results['min_fps']:.2f} - {results['max_fps']:.2f}")
        self.logger.info(f"  Average Processing Time: {results['avg_processing_time_ms']:.2f} ms")
        self.logger.info(f"  Processing Time Range: {results['min_processing_time_ms']:.2f} - {results['max_processing_time_ms']:.2f} ms")
        self.logger.info("-" * 25)
        self.logger.info("Detection Metrics:")
        self.logger.info(f"  Total Detections: {results['total_detections']}")
        self.logger.info(f"  Average per Frame: {results['avg_detections_per_frame']:.2f}")
        self.logger.info(f"  Detection Range: {results['min_detections_per_frame']} - {results['max_detections_per_frame']}")
        self.logger.info("=" * 50)


def main():
    """
    Main function providing command-line interface for vehicle detection.
    Supports video processing, webcam input, and model benchmarking.
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize detector with configuration
        detector = VideoDetector(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        if args.benchmark:
            run_benchmark(detector, args)
        else:
            run_video_processing(detector, args)
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Error in main: {e}", exc_info=True)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced Vehicle Detection using YOLO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source 0                          # Use webcam
  %(prog)s --source video.mp4 --output out.mp4 # Process video file
  %(prog)s --source video.mp4 --benchmark      # Benchmark model
  %(prog)s --model models/yolo11x.pt --confidence 0.8  # High accuracy mode
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--source', '-s', type=str, default='0',
                       help='Video source: file path, webcam index (0), or stream URL')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (optional)')
    
    # Model configuration
    parser.add_argument('--model', '-m', type=str, default=Config.DEFAULT_MODEL,
                       help=f'YOLO model path (default: {Config.DEFAULT_MODEL})')
    parser.add_argument('--confidence', '-c', type=float, default=Config.CONFIDENCE_THRESHOLD,
                       help=f'Confidence threshold (0.0-1.0, default: {Config.CONFIDENCE_THRESHOLD})')
    
    # Display options
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display (useful for headless processing)')
    
    # Benchmark options
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark instead of processing')
    parser.add_argument('--benchmark-frames', type=int, default=100,
                       help='Number of frames for benchmarking (default: 100)')
    
    return parser


def run_benchmark(detector: VideoDetector, args) -> None:
    """Run model benchmarking with the given arguments."""
    if args.source == '0':
        print("❌ Cannot benchmark with webcam. Please provide a video file.")
        return
    
    print(f"🔥 Starting benchmark with {args.benchmark_frames} frames...")
    benchmark_results = detector.benchmark_model(args.source, args.benchmark_frames)
    print("\n✅ Benchmark completed successfully!")
    
    # Save benchmark results
    benchmark_file = Path(Config.LOGS_DIR) / f"benchmark_results_{int(time.time())}.json"
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"📊 Benchmark results saved to: {benchmark_file}")


def run_video_processing(detector: VideoDetector, args) -> None:
    """Run video processing with the given arguments."""
    print(f"🎥 Starting video processing...")
    print(f"📹 Source: {args.source}")
    if args.output:
        print(f"💾 Output: {args.output}")
    
    stats = detector.process_video(
        source=args.source,
        output_path=args.output,
        display=not args.no_display
    )
    print("\n✅ Video processing completed successfully!")
    print(f"📈 Processed {stats['total_frames_processed']} frames at {stats['average_fps']:.2f} FPS")
    print(f"🚗 Detected {stats['total_vehicle_detections']} total vehicles")


if __name__ == "__main__":
    main()
