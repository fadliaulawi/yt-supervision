import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os
from pathlib import Path
import logging
from typing import Optional, List, Dict, Tuple
import json
import requests
import urllib.request
from urllib.error import URLError

class VideoDetector:
    """
    A comprehensive video detector for vehicles using YOLO models.
    Supports real-time detection, video file processing, and webcam input.
    """
    
    def __init__(self, model_path: str = "models/yolo11n.pt", confidence_threshold: float = 0.7):
        """
        Initialize the VideoDetector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO dataset
        self.class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        self.colors = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0), # Blue
            'bus': (0, 0, 255),      # Red
            'truck': (255, 0, 255)   # Magenta
        }
        
        # Statistics tracking
        self.detection_stats = {
            'total_frames': 0,
            'detections_per_frame': [],
            'processing_times': [],
            'vehicle_counts': {cls: 0 for cls in self.class_names.values()}
        }
        
        self.setup_logging()
        self.ensure_model_available()
        self.load_model()

    def ensure_model_available(self):
        """
        Check if the model exists, and download it if not.
        Falls back to YOLOv8 if YOLOv11 is not available.
        """
        model_file = Path(self.model_path)
        
        # Create models directory if it doesn't exist
        model_file.parent.mkdir(parents=True, exist_ok=True)
        
        if model_file.exists():
            self.logger.info(f"Model found: {self.model_path}")
            return
        
        self.logger.info(f"Model not found: {self.model_path}")
        
        # Try to download the model
        if self.download_model(self.model_path):
            self.logger.info(f"Successfully downloaded: {self.model_path}")
            return
        
        # If download fails, fall back to YOLOv8
        fallback_model = self.model_path.replace("yolo11", "yolov8")
        fallback_path = Path(fallback_model)
        
        if fallback_path.exists():
            self.logger.warning(f"Using fallback model: {fallback_model}")
            self.model_path = fallback_model
            return
        
        # Try to download YOLOv8 fallback
        if self.download_model(fallback_model):
            self.logger.info(f"Successfully downloaded fallback model: {fallback_model}")
            self.model_path = fallback_model
            return
        
        # Final fallback - let YOLO handle auto-download
        self.logger.warning("Will rely on YOLO auto-download functionality")

    def download_model(self, model_path: str) -> bool:
        """
        Download a YOLO model from the official repository.
        
        Args:
            model_path: Path where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = Path(model_path).name
            
            # Determine download URL based on model name
            if "yolo11" in model_name:
                # YOLOv11 models (latest)
                base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
            elif "yolov10" in model_name:
                # YOLOv10 models
                base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
            elif "yolov9" in model_name:
                # YOLOv9 models
                base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
            else:
                # YOLOv8 models (fallback)
                base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
            
            download_url = f"{base_url}/{model_name}"
            
            self.logger.info(f"Downloading {model_name} from {download_url}")
            
            # Download with progress indication
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size / total_size) * 100)
                    if block_num % 50 == 0:  # Update every 50 blocks
                        print(f"\rDownloading {model_name}: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(download_url, model_path, reporthook=show_progress)
            print(f"\n✅ Successfully downloaded {model_name}")
            
            # Verify the downloaded file
            if Path(model_path).stat().st_size > 1000000:  # At least 1MB
                return True
            else:
                self.logger.error(f"Downloaded file seems too small: {model_path}")
                Path(model_path).unlink(missing_ok=True)  # Remove incomplete file
                return False
                
        except URLError as e:
            self.logger.error(f"Network error downloading {model_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to download {model_name}: {e}")
            return False
    
    def setup_logging(self):
        """Setup logging configuration with UTF-8 encoding for Unicode support."""
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/video_detector.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the YOLO model."""
        try:
            self.logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        start_time = time.time()
        
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for vehicle classes only
                    if class_id in self.vehicle_classes:
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names[class_id]
                        }
                        detections.append(detection)
        
        processing_time = time.time() - start_time
        self.detection_stats['processing_times'].append(processing_time)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name} {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def add_info_overlay(self, frame: np.ndarray, detections: List[Dict], fps: float) -> np.ndarray:
        """
        Add information overlay to the frame.
        
        Args:
            frame: Input frame
            detections: Current frame detections
            fps: Current FPS
            
        Returns:
            Frame with info overlay
        """
        height, width = frame.shape[:2]
        
        # Count vehicles by type
        vehicle_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
        
        # Prepare info text
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Total Vehicles: {len(detections)}",
        ]
        
        for vehicle_type, count in vehicle_counts.items():
            info_lines.append(f"{vehicle_type.capitalize()}s: {count}")
        
        # Draw info background
        info_height = len(info_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (250, info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, info_height), (255, 255, 255), 2)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def update_statistics(self, detections: List[Dict]):
        """Update detection statistics."""
        self.detection_stats['total_frames'] += 1
        self.detection_stats['detections_per_frame'].append(len(detections))
        
        for detection in detections:
            class_name = detection['class_name']
            self.detection_stats['vehicle_counts'][class_name] += 1
    
    def process_video(self, 
                     source: str, 
                     output_path: Optional[str] = None,
                     display: bool = True,
                     save_stats: bool = True) -> Dict:
        """
        Process video from file, webcam, or stream.
        
        Args:
            source: Video source (file path, webcam index, or stream URL)
            output_path: Optional output video path
            display: Whether to display video in real-time
            save_stats: Whether to save detection statistics
            
        Returns:
            Dictionary with processing statistics
        """
        # Open video source
        if source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else -1
        
        self.logger.info(f"Video properties: {width}x{height} @ {fps} FPS")
        if total_frames > 0:
            self.logger.info(f"Total frames: {total_frames}")
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect vehicles
                detections = self.detect_vehicles(frame)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Add info overlay
                annotated_frame = self.add_info_overlay(annotated_frame, detections, current_fps)
                
                # Update statistics
                self.update_statistics(detections)
                
                # Display frame
                if display:
                    cv2.imshow('Vehicle Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and output_path is None:
                        # Save screenshot
                        screenshot_path = f"detection_screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        self.logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Write to output video
                if out:
                    out.write(annotated_frame)
                
                # Progress logging
                if total_frames > 0 and frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Processing: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
        
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
        avg_processing_time = np.mean(self.detection_stats['processing_times']) if self.detection_stats['processing_times'] else 0
        
        stats = {
            'total_frames_processed': frame_count,
            'total_processing_time': total_time,
            'average_fps': avg_fps,
            'average_detection_time': avg_processing_time,
            'total_vehicle_detections': sum(self.detection_stats['vehicle_counts'].values()),
            'vehicle_counts_by_type': self.detection_stats['vehicle_counts'].copy(),
            'avg_detections_per_frame': np.mean(self.detection_stats['detections_per_frame']) if self.detection_stats['detections_per_frame'] else 0
        }
        
        self.logger.info("Processing completed:")
        self.logger.info(f"  Frames processed: {stats['total_frames_processed']}")
        self.logger.info(f"  Average FPS: {stats['average_fps']:.2f}")
        self.logger.info(f"  Total vehicles detected: {stats['total_vehicle_detections']}")
        self.logger.info(f"  Vehicle breakdown: {stats['vehicle_counts_by_type']}")
        
        # Save statistics to file
        if save_stats:
            stats_file = f"logs/detection_stats_{int(time.time())}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            self.logger.info(f"Statistics saved to: {stats_file}")
        
        return stats
    
    def benchmark_model(self, test_video: str, num_frames: int = 100) -> Dict:
        """
        Benchmark the model performance on a test video.
        
        Args:
            test_video: Path to test video
            num_frames: Number of frames to process for benchmarking
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Benchmarking model on {num_frames} frames...")
        
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened():
            raise ValueError(f"Failed to open test video: {test_video}")
        
        processing_times = []
        detection_counts = []
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            detections = self.detect_vehicles(frame)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            detection_counts.append(len(detections))
        
        cap.release()
        
        if not processing_times:
            raise ValueError("No frames could be processed")
        
        benchmark_results = {
            'frames_processed': len(processing_times),
            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'avg_fps': 1.0 / np.mean(processing_times),
            'avg_detections_per_frame': np.mean(detection_counts)
        }
        
        self.logger.info("Benchmark Results:")
        for key, value in benchmark_results.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return benchmark_results


def main():
    """Main function to run the video detector."""
    parser = argparse.ArgumentParser(description="Vehicle Detection in Videos using YOLO")
    parser.add_argument('--source', '-s', type=str, default='0', 
                       help='Video source (file path, webcam index, or stream URL)')
    parser.add_argument('--model', '-m', type=str, default='models/yolo11n.pt',
                       help='YOLO model path (models/yolo11n.pt, models/yolo11s.pt, models/yolo11m.pt, models/yolo11l.pt, models/yolo11x.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark mode')
    parser.add_argument('--benchmark-frames', type=int, default=100,
                       help='Number of frames for benchmarking')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = VideoDetector(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        if args.benchmark:
            # Run benchmark
            if args.source == '0':
                print("Error: Cannot benchmark with webcam. Please provide a video file.")
                return
            
            benchmark_results = detector.benchmark_model(args.source, args.benchmark_frames)
            print("\nBenchmark completed successfully!")
        else:
            # Process video
            stats = detector.process_video(
                source=args.source,
                output_path=args.output,
                display=not args.no_display
            )
            print("\nVideo processing completed successfully!")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Error in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()
