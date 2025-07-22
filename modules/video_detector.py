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
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import Config


class KalmanBoxTracker:
    """
    Kalman filter-based tracker for bounding box tracking.
    Represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        """
        # Define constant velocity model
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            # Fallback implementation without filterpy
            self.use_simple_tracker = True
            self.bbox = bbox
            self.time_since_update = 0
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1
            self.history = []
            self.hits = 1
            self.hit_streak = 1
            self.age = 1
            return

        self.use_simple_tracker = False
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 1

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        if self.use_simple_tracker:
            self.bbox = bbox
        else:
            self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if self.use_simple_tracker:
            self.age += 1
            if self.time_since_update > 0:
                self.hit_streak = 0
            self.time_since_update += 1
            self.history.append(self._convert_x_to_bbox(self.bbox))
            return self.history[-1]
        
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        if self.use_simple_tracker:
            return self._convert_x_to_bbox(self.bbox)
        return self._convert_x_to_bbox(self.kf.x)

    def _convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def _convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        if isinstance(x, list):
            # Simple tracker case
            return x
        
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def linear_assignment(cost_matrix):
    """
    Simple linear assignment implementation as fallback for scipy.optimize.linear_sum_assignment
    """
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost_matrix)
    except ImportError:
        # Simple greedy assignment as fallback
        assignments = []
        cost_matrix = cost_matrix.copy()
        
        for _ in range(min(cost_matrix.shape)):
            # Find minimum cost
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            assignments.append(min_idx)
            
            # Set row and column to infinity
            cost_matrix[min_idx[0], :] = np.inf
            cost_matrix[:, min_idx[1]] = np.inf
        
        if assignments:
            return np.array(assignments).T
        else:
            return np.array([[], []], dtype=int)


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return o  


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    SORT: Simple, online, and realtime tracking of multiple objects in a video sequence.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections 
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


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
                 confidence_threshold: float = None,
                 enable_tracking: bool = True):
        """
        Initialize the VideoDetector with configurable parameters.
        
        Args:
            model_path: Path to YOLO model file (defaults to Config.DEFAULT_MODEL)
            confidence_threshold: Detection confidence threshold (defaults to Config.CONFIDENCE_THRESHOLD)
            enable_tracking: Enable object tracking for unique vehicle identification
        """
        # Use config defaults if not specified
        self.model_path = model_path or Config.DEFAULT_MODEL
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        self.model = None
        self.enable_tracking = enable_tracking
        
        # Load configuration from Config class
        self.vehicle_classes = list(Config.VEHICLE_CLASSES.keys())
        self.class_names = Config.VEHICLE_CLASSES
        self.colors = Config.DETECTION_COLORS
        
        # Initialize tracking system
        if self.enable_tracking:
            self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
            self.track_colors = {}  # Store colors for each track ID
            self.unique_vehicle_count = 0
            self.tracked_vehicles = set()  # Set to store unique vehicle IDs
        
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
        """Initialize detection statistics tracking with tracking support."""
        self.detection_stats = {
            'total_frames': 0,
            'detections_per_frame': [],
            'processing_times': [],
            'vehicle_counts': {cls: 0 for cls in self.class_names.values()}
        }
        
        # Initialize tracking-specific statistics if tracking is enabled
        if self.enable_tracking:
            self.detection_stats.update({
                'unique_vehicle_counts': {cls: 0 for cls in self.class_names.values()},
                'tracked_vehicles_by_class': {cls: set() for cls in self.class_names.values()}
            })

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

    def detect_and_track_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and track vehicles in a single frame with unique ID assignment.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, class info, and track ID
        """
        start_time = time.time()
        
        # Run YOLO inference with configured parameters
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = self._parse_detection_results(results)
        
        if self.enable_tracking and detections:
            # Convert detections to format expected by SORT tracker
            dets = self._convert_detections_to_sort_format(detections)
            
            # Update tracker with new detections
            tracked_objects = self.tracker.update(dets)
            
            # Convert tracked objects back to detection format with track IDs
            detections = self._convert_tracked_objects_to_detections(tracked_objects, detections)
            
            # Update unique vehicle tracking
            for detection in detections:
                if 'track_id' in detection:
                    if detection['track_id'] not in self.tracked_vehicles:
                        self.tracked_vehicles.add(detection['track_id'])
                        self.unique_vehicle_count += 1
        else:
            # Update tracker even with empty detections to maintain state
            if self.enable_tracking:
                self.tracker.update(np.empty((0, 5)))
        
        # Track performance
        processing_time = time.time() - start_time
        self.detection_stats['processing_times'].append(processing_time)
        
        return detections

    def _convert_detections_to_sort_format(self, detections: List[Dict]) -> np.ndarray:
        """Convert detection list to SORT-compatible numpy array format."""
        if not detections:
            return np.empty((0, 5))
        
        sort_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            sort_detections.append([x1, y1, x2, y2, confidence])
        
        return np.array(sort_detections)

    def _convert_tracked_objects_to_detections(self, tracked_objects: np.ndarray, 
                                              original_detections: List[Dict]) -> List[Dict]:
        """Convert SORT tracked objects back to detection dictionary format."""
        if len(tracked_objects) == 0:
            return []
        
        tracked_detections = []
        
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Find the closest original detection to associate class information
            best_detection = self._find_closest_detection(
                [x1, y1, x2, y2], original_detections
            )
            
            if best_detection:
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': best_detection['confidence'],
                    'class_id': best_detection['class_id'],
                    'class_name': best_detection['class_name'],
                    'track_id': track_id
                }
                tracked_detections.append(detection)
        
        return tracked_detections

    def _find_closest_detection(self, bbox: List[float], 
                               detections: List[Dict]) -> Optional[Dict]:
        """Find the detection with highest IoU to the given bbox."""
        if not detections:
            return None
        
        max_iou = 0
        best_detection = None
        
        for detection in detections:
            iou = self._calculate_iou(bbox, detection['bbox'])
            if iou > max_iou:
                max_iou = iou
                best_detection = detection
        
        return best_detection if max_iou > 0.1 else detections[0]  # Fallback to first detection

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame with optional tracking.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries with bbox, confidence, class info, and optional track ID
        """
        if self.enable_tracking:
            return self.detect_and_track_vehicles(frame)
        
        # Original detection without tracking
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
        """Draw a single detection box and label with optional track ID."""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Get track ID if available
        track_id = detection.get('track_id', None)
        
        # Use track-specific color if tracking is enabled
        if self.enable_tracking and track_id is not None:
            color = self._get_track_color(track_id, class_name)
        else:
            color = self.colors.get(class_name, self.colors['default'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, Config.BOX_THICKNESS)
        
        # Create label with track ID
        if track_id is not None:
            label = f"{class_name} ID:{track_id} {confidence:.2f}"
        else:
            label = f"{class_name} {confidence:.2f}"
        
        self._draw_label_with_background(frame, label, (x1, y1), color)

    def _get_track_color(self, track_id: int, class_name: str) -> Tuple[int, int, int]:
        """Get a consistent color for a track ID."""
        if track_id not in self.track_colors:
            # Generate a unique color based on track ID
            np.random.seed(track_id)
            base_color = self.colors.get(class_name, self.colors['default'])
            
            # Add some variation to the base color
            variation = 50
            r = max(0, min(255, base_color[2] + np.random.randint(-variation, variation)))
            g = max(0, min(255, base_color[1] + np.random.randint(-variation, variation)))
            b = max(0, min(255, base_color[0] + np.random.randint(-variation, variation)))
            
            self.track_colors[track_id] = (b, g, r)  # BGR format for OpenCV
        
        return self.track_colors[track_id]

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
        Add comprehensive information overlay to the frame with tracking stats.
        
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
            f"Current Detections: {len(detections)}",
            *[f"{vehicle_type.capitalize()}s: {count}" 
              for vehicle_type, count in vehicle_counts.items()]
        ]
        
        # Add tracking information
        if self.enable_tracking:
            active_tracks = len([d for d in detections if 'track_id' in d])
            
            # Get current unique counts from detection_stats
            unique_counts = self.detection_stats.get('unique_vehicle_counts', {})
            total_unique = sum(unique_counts.values()) if unique_counts else 0
            
            info_lines.extend([
                "-" * 15,
                f"Active Tracks: {active_tracks}",
                f"Unique Vehicles: {total_unique}"
            ])
            
            # Add unique counts by type if any exist
            if unique_counts:
                for vehicle_type, count in unique_counts.items():
                    if count > 0:
                        info_lines.append(f"Unique {vehicle_type.capitalize()}s: {count}")
        
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
        """Update comprehensive detection statistics with tracking support."""
        self.detection_stats['total_frames'] += 1
        self.detection_stats['detections_per_frame'].append(len(detections))
        
        # Update vehicle counts
        for detection in detections:
            class_name = detection['class_name']
            self.detection_stats['vehicle_counts'][class_name] += 1
            
            # Update tracking statistics if tracking is enabled and track_id exists
            if self.enable_tracking and 'track_id' in detection:
                track_id = detection['track_id']
                
                # Check if this track ID is new for this vehicle class
                if track_id not in self.detection_stats['tracked_vehicles_by_class'][class_name]:
                    self.detection_stats['tracked_vehicles_by_class'][class_name].add(track_id)
                    self.detection_stats['unique_vehicle_counts'][class_name] += 1
                    self.logger.debug(f"New {class_name} tracked: ID {track_id}. Total unique {class_name}s: {self.detection_stats['unique_vehicle_counts'][class_name]}")
    
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
        """Generate comprehensive final statistics with tracking information."""
        total_time = processing_stats['processing_time']
        frames_processed = processing_stats['frames_processed']
        
        stats = {
            'total_frames_processed': frames_processed,
            'total_processing_time': total_time,
            'average_fps': frames_processed / total_time if total_time > 0 else 0,
            'average_detection_time': np.mean(self.detection_stats['processing_times']) if self.detection_stats['processing_times'] else 0,
            'total_vehicle_detections': sum(self.detection_stats['vehicle_counts'].values()),
            'vehicle_counts_by_type': self.detection_stats['vehicle_counts'].copy(),
            'avg_detections_per_frame': np.mean(self.detection_stats['detections_per_frame']) if self.detection_stats['detections_per_frame'] else 0
        }
        
        # Add tracking statistics
        if self.enable_tracking:
            unique_counts = self.detection_stats.get('unique_vehicle_counts', {})
            unique_total = sum(unique_counts.values()) if unique_counts else self.unique_vehicle_count
            
            stats.update({
                'tracking_enabled': True,
                'unique_vehicles_total': unique_total,
                'unique_vehicle_counts_by_type': unique_counts,
                'active_tracks': len(self.tracker.trackers) if hasattr(self.tracker, 'trackers') else 0
            })
        else:
            stats['tracking_enabled'] = False
        
        return stats

    def _log_processing_summary(self, stats: Dict):
        """Log comprehensive processing summary with tracking information."""
        self.logger.info("Processing completed:")
        self.logger.info(f"  Frames processed: {stats['total_frames_processed']}")
        self.logger.info(f"  Average FPS: {stats['average_fps']:.2f}")
        
        if stats.get('tracking_enabled', False):
            self.logger.info(f"  Total detections: {stats['total_vehicle_detections']}")
            self.logger.info(f"  Unique vehicles tracked: {stats['unique_vehicles_total']}")
            self.logger.info(f"  Detection breakdown: {stats['vehicle_counts_by_type']}")
            
            # Log unique vehicle breakdown with better formatting
            unique_breakdown = stats.get('unique_vehicle_counts_by_type', {})
            if unique_breakdown:
                # Filter out zero counts for cleaner display
                non_zero_unique = {k: v for k, v in unique_breakdown.items() if v > 0}
                if non_zero_unique:
                    self.logger.info(f"  Unique vehicle breakdown: {non_zero_unique}")
                else:
                    self.logger.info("  Unique vehicle breakdown: No unique vehicles tracked yet")
            
            self.logger.info(f"  Active tracks at end: {stats['active_tracks']}")
        else:
            self.logger.info(f"  Total vehicles detected: {stats['total_vehicle_detections']}")
            self.logger.info(f"  Vehicle breakdown: {stats['vehicle_counts_by_type']}")
            self.logger.info("  Note: Tracking was disabled - counts may include duplicates")
    
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
            confidence_threshold=args.confidence,
            enable_tracking=not args.no_tracking
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
  %(prog)s --source 0                          # Use webcam with tracking
  %(prog)s --source video.mp4 --output out.mp4 # Process video file with tracking
  %(prog)s --source video.mp4 --no-tracking    # Process without tracking
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
    
    # Tracking options
    parser.add_argument('--no-tracking', action='store_true',
                       help='Disable object tracking (may cause duplicate counting)')
    
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
