"""
Dashboard Core Module
Core StreamlitDashboard class for vehicle detection dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

import cv2
import time
import threading
import os
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

# Import detection modules
from .video_detector import VideoDetector
from .youtube_watcher import YouTubeVideoWatcher
from .status_manager import set_analysis_starting, set_analysis_running, set_analysis_error


class StreamlitDashboard:
    """Main dashboard class for Streamlit interface."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.detector = None
        self.watcher = None
        self.video_source = None
        self.detection_history = []
        self.fps_history = []
        self.time_stamps = []
        self.analysis_running = False  # Instance-level flag for thread communication
    
    def initialize_session_state(self):
        """Initialize session state variables - must be called from main thread."""
        if 'detection_counts' not in st.session_state:
            st.session_state.detection_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        if 'total_detections' not in st.session_state:
            st.session_state.total_detections = 0
        if 'analysis_start_time' not in st.session_state:
            st.session_state.analysis_start_time = None
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False
        if 'current_session_results' not in st.session_state:
            st.session_state.current_session_results = None
    
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Vehicle Detection Dashboard",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-running {
            color: #28a745;
            font-weight: bold;
        }
        .status-stopped {
            color: #dc3545;
            font-weight: bold;
        }
        .video-container {
            border: 2px solid #1f77b4;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Create sidebar with controls."""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # Analysis mode selection
            mode = st.selectbox(
                "Analysis Mode",
                ["📺 YouTube Video/Stream", "📹 Video File", "🎥 Webcam"],
                help="Choose your input source"
            )
            
            # Model selection
            model_options = {
                "YOLOv8n (Fastest)": "models/yolov8n.pt",
                "YOLOv11l (High Accuracy)": "models/yolo11l.pt",
                "YOLOv11x (Research)": "models/yolo11x.pt"
            }
            
            selected_model = st.selectbox(
                "YOLO Model",
                list(model_options.keys()),
                index=1,  # Default to YOLOv11l
                help="Choose model based on speed vs accuracy needs"
            )
            
            model_path = model_options[selected_model]
            
            # Confidence threshold
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Higher values = fewer but more confident detections"
            )
            
            st.divider()
            
            # Mode-specific settings
            if mode == "📺 YouTube Video/Stream":
                youtube_url = st.text_input(
                    "YouTube URL",
                    value="https://www.youtube.com/live/6QL0RHNtOlo",
                    help="Enter YouTube video or live stream URL"
                )
                
                quality = st.selectbox(
                    "Video Quality",
                    ["480p", "720p", "1080p", "best"],
                    index=1,  # Default to 720p
                    help="Higher quality = better accuracy but slower processing"
                )
                
                max_duration = st.number_input(
                    "Max Duration (minutes)",
                    min_value=1,
                    max_value=180,
                    value=30,
                    help="Maximum analysis duration"
                )
                
                return {
                    'mode': 'youtube',
                    'url': youtube_url,
                    'quality': quality,
                    'max_duration': max_duration * 60,
                    'model_path': model_path,
                    'confidence': confidence
                }
                
            elif mode == "📹 Video File":
                video_file = st.file_uploader(
                    "Upload Video File",
                    type=['mp4', 'avi', 'mov', 'mkv', 'flv'],
                    help="Upload a video file for analysis"
                )
                
                return {
                    'mode': 'file',
                    'file': video_file,
                    'model_path': model_path,
                    'confidence': confidence
                }
                
            else:  # Webcam
                return {
                    'mode': 'webcam',
                    'model_path': model_path,
                    'confidence': confidence
                }
    
    def create_main_dashboard(self):
        """Create the main dashboard interface."""
        # Header
        st.markdown('<h1 class="main-header">🚗 Vehicle Detection Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Status indicator
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Dynamic status indicator - check from status file
            status = self.check_analysis_status()
            
            if status['status'] == 'running':
                st.markdown('<p class="status-running">🟢 Analysis Running in Independent Process</p>', 
                           unsafe_allow_html=True)
            elif status['status'] == 'completed':
                st.markdown('<p class="status-running">✅ Analysis Completed - Check Results</p>', 
                           unsafe_allow_html=True)
            elif status['status'] == 'error':
                st.markdown('<p class="status-stopped">❌ Analysis Error</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-stopped">🔴 Ready for Analysis</p>', 
                           unsafe_allow_html=True)
        
        with col2:
            if status['status'] == 'running':
                start_button = st.button("📊 Check Progress")
            else:
                start_button = st.button("▶️ Start Analysis")
        
        with col3:
            stop_button = st.button("🧹 Clear Results")
        
        return start_button, stop_button
    
    def create_metrics_display(self):
        """Create real-time metrics display."""
        st.subheader("📊 Real-time Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🚗 Cars",
                value=st.session_state.detection_counts['car'],
                delta=None
            )
        
        with col2:
            st.metric(
                label="🚛 Trucks",
                value=st.session_state.detection_counts['truck'],
                delta=None
            )
        
        with col3:
            st.metric(
                label="🚌 Buses", 
                value=st.session_state.detection_counts['bus'],
                delta=None
            )
        
        with col4:
            st.metric(
                label="🏍️ Motorcycles",
                value=st.session_state.detection_counts['motorcycle'],
                delta=None
            )
        
        # Total and time elapsed
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                label="🎯 Total Vehicles",
                value=st.session_state.total_detections,
                delta=None
            )
        
        with col6:
            if st.session_state.analysis_start_time:
                elapsed = datetime.now() - st.session_state.analysis_start_time
                elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
                st.metric(
                    label="⏱️ Time Elapsed",
                    value=elapsed_str,
                    delta=None
                )
        
        with col7:
            if len(self.fps_history) > 0:
                current_fps = self.fps_history[-1] if self.fps_history else 0
                st.metric(
                    label="⚡ Current FPS",
                    value=f"{current_fps:.1f}",
                    delta=None
                )
    
    def create_charts(self):
        """Create real-time charts."""
        st.subheader("📈 Analytics")
        
        if len(self.detection_history) > 0:
            # Create two columns for charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Vehicle breakdown pie chart
                counts = st.session_state.detection_counts
                if sum(counts.values()) > 0:
                    fig_pie = px.pie(
                        values=list(counts.values()),
                        names=list(counts.keys()),
                        title="🚗 Vehicle Type Distribution",
                        color_discrete_map={
                            'car': '#2E8B57',
                            'truck': '#FF6347', 
                            'bus': '#4169E1',
                            'motorcycle': '#FFD700'
                        }
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Detection timeline
                if len(self.time_stamps) > 0:
                    df_timeline = pd.DataFrame({
                        'Time': self.time_stamps,
                        'Detections': self.detection_history,
                        'FPS': self.fps_history[:len(self.time_stamps)]
                    })
                    
                    fig_timeline = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Detections Over Time', 'FPS Over Time'),
                        vertical_spacing=0.15
                    )
                    
                    # Detections line
                    fig_timeline.add_trace(
                        go.Scatter(
                            x=df_timeline['Time'],
                            y=df_timeline['Detections'],
                            mode='lines+markers',
                            name='Detections',
                            line=dict(color='#1f77b4', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # FPS line
                    fig_timeline.add_trace(
                        go.Scatter(
                            x=df_timeline['Time'],
                            y=df_timeline['FPS'],
                            mode='lines+markers', 
                            name='FPS',
                            line=dict(color='#ff7f0e', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig_timeline.update_layout(height=400, showlegend=False)
                    fig_timeline.update_xaxes(title_text="Time", row=2, col=1)
                    fig_timeline.update_yaxes(title_text="Count", row=1, col=1)
                    fig_timeline.update_yaxes(title_text="FPS", row=2, col=1)
                    
                    st.plotly_chart(fig_timeline, use_container_width=True)
    
    def create_video_display(self):
        """Create video display area."""
        st.subheader("🎥 Live Video Feed")
        
        # Video container
        video_container = st.empty()
        
        if not st.session_state.is_running:
            video_container.info("📺 Click 'Start Analysis' to begin vehicle detection")
        elif not hasattr(st.session_state, 'current_frame') or st.session_state.current_frame is None:
            video_container.info("⏳ Initializing video feed...")
        
        return video_container
    
    def initialize_video_source(self, config):
        """Initialize video source based on configuration."""
        try:
            # Check if model file exists
            model_path = Path(config['model_path'])
            if not model_path.exists():
                st.error(f"❌ Model file not found: {config['model_path']}")
                st.info("Available models in models/ folder:")
                models_dir = Path("models")
                if models_dir.exists():
                    for model_file in models_dir.glob("*.pt"):
                        st.info(f"  - {model_file.name}")
                return False
            
            if config['mode'] == 'youtube':
                st.info("🔄 Initializing YouTube stream...")
                
                # Initialize watcher with error handling
                try:
                    self.watcher = YouTubeVideoWatcher(
                        confidence_threshold=config['confidence'],
                        model_path=config['model_path'],
                        quality=config['quality']
                    )
                    st.info("✅ YouTube watcher created")
                except Exception as e:
                    st.error(f"❌ Failed to create YouTube watcher: {str(e)}")
                    return False
                
                # Verify watcher initialization
                if self.watcher is None:
                    st.error("❌ Watcher is None after initialization")
                    return False
                    
                if not hasattr(self.watcher, 'detector'):
                    st.error("❌ Watcher does not have detector attribute")
                    return False
                    
                if self.watcher.detector is None:
                    st.error("❌ Watcher detector is None")
                    return False
                
                st.info("✅ YouTube watcher detector verified")
                
                # Get stream URL
                try:
                    stream_url = self.watcher.get_stream_url(config['url'])
                    if not stream_url:
                        st.error("❌ Failed to get YouTube stream URL")
                        return False
                    st.info(f"✅ Stream URL obtained: {stream_url[:50]}...")
                except Exception as e:
                    st.error(f"❌ Error getting stream URL: {str(e)}")
                    return False
                
                # Initialize video capture
                try:
                    st.session_state.video_source = cv2.VideoCapture(stream_url)
                    if not st.session_state.video_source.isOpened():
                        st.error("❌ Failed to open YouTube stream")
                        return False
                    st.success("✅ YouTube stream initialized!")
                except Exception as e:
                    st.error(f"❌ Error opening video stream: {str(e)}")
                    return False
                
            elif config['mode'] == 'webcam':
                st.info("🔄 Initializing webcam...")
                try:
                    self.detector = VideoDetector(
                        model_path=config['model_path'],
                        confidence_threshold=config['confidence']
                    )
                    
                    # Verify detector initialization
                    if self.detector is None:
                        st.error("❌ Failed to initialize detector")
                        return False
                    
                    st.session_state.video_source = cv2.VideoCapture(0)
                    if not st.session_state.video_source.isOpened():
                        st.error("❌ Failed to open webcam")
                        return False
                        
                    st.success("✅ Webcam initialized!")
                except Exception as e:
                    st.error(f"❌ Error initializing webcam: {str(e)}")
                    return False
                
            elif config['mode'] == 'file':
                if config['file'] is None:
                    st.error("❌ Please upload a video file")
                    return False
                    
                st.info("🔄 Initializing video file...")
                try:
                    self.detector = VideoDetector(
                        model_path=config['model_path'],
                        confidence_threshold=config['confidence']
                    )
                    
                    # Verify detector initialization
                    if self.detector is None:
                        st.error("❌ Failed to initialize detector")
                        return False
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_{int(time.time())}.mp4"
                    with open(temp_path, "wb") as f:
                        f.write(config['file'].getbuffer())
                    
                    st.session_state.video_source = cv2.VideoCapture(temp_path)
                    if not st.session_state.video_source.isOpened():
                        st.error("❌ Failed to open video file")
                        return False
                        
                    st.success("✅ Video file initialized!")
                except Exception as e:
                    st.error(f"❌ Error initializing video file: {str(e)}")
                    return False
            
            # Store config in session state for processing
            st.session_state.video_config = config
            return True
            
        except Exception as e:
            st.error(f"❌ Error initializing video source: {str(e)}")
            return False

    def check_analysis_status(self):
        """Check analysis status from status file."""
        try:
            status_file = "logs/analysis_status.json"
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                return status_data
            return {'status': 'none'}
        except:
            return {'status': 'none'}
    
    def run_video_analysis_fire_and_forget(self, config):
        """Run video analysis completely independently using demo.py subprocess."""
        
        try:
            # Mark analysis as starting
            set_analysis_starting(
                mode=config['mode'],
                model_path=config['model_path'],
                confidence=config['confidence']
            )
            
            # Build command arguments for demo.py
            cmd_args = [sys.executable, "demo.py"]
            
            # Add mode-specific arguments
            if config['mode'] == 'youtube':
                cmd_args.extend(["--youtube-watch", config['url']])
                cmd_args.extend(["--quality", config['quality']])
                
                if 'max_duration' in config and config['max_duration']:
                    # Convert seconds to minutes for demo.py
                    duration_minutes = int(config['max_duration'] / 60)
                    cmd_args.extend(["--duration", str(duration_minutes)])
                    
            elif config['mode'] == 'webcam':
                cmd_args.append("--webcam")
                
            elif config['mode'] == 'file':
                if config.get('file'):
                    # Save uploaded file temporarily
                    temp_path = f"temp_{int(time.time())}.mp4"
                    with open(temp_path, "wb") as f:
                        f.write(config['file'].getbuffer())
                    cmd_args.extend(["--video", temp_path])
                else:
                    cmd_args.extend(["--video", "sample.mp4"])  # fallback to sample file
            
            # Add output path if processing video file
            if config['mode'] in ['file', 'webcam']:
                output_path = f"logs/demo_output_{int(time.time())}.mp4"
                cmd_args.extend(["--output", output_path])
            
            # Set up environment for subprocess to handle encoding properly
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '1'  # For Windows compatibility
            
            # Launch demo.py as independent process with proper encoding
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    cmd_args,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NEW_CONSOLE,
                    env=env,
                    cwd=os.getcwd(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Unix/Linux
                process = subprocess.Popen(
                    cmd_args,
                    env=env,
                    cwd=os.getcwd(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Update status to running
            set_analysis_running(
                process_id=process.pid,
                command=" ".join(cmd_args),
                start_time=time.time()
            )
            
            st.session_state.analysis_running = True
            return True
            
        except Exception as e:
            st.error(f"Failed to start demo.py analysis: {str(e)}")
            # Update status to error
            set_analysis_error(str(e))
            return False

    def display_analysis_results(self, results):
        """Display analysis results from session state."""
        if not results:
            st.info("📋 No analysis results available in current session.")
            st.info("💡 Results will appear here after completing an analysis.")
            return
        
        st.success("✅ Displaying results from current session")
        
        # Summary info
        col1, col2, col3 = st.columns(3)
        with col1:
            # Safely get analysis start time with fallback
            start_time_str = "Unknown"
            if 'analysis_start_time' in results:
                try:
                    start_time_str = datetime.fromisoformat(results['analysis_start_time']).strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    start_time_str = "Invalid Date"
            
            st.metric("📅 Analysis Date", start_time_str)
        with col2:
            duration = results.get('duration_seconds', 0)
            st.metric("⏱️ Duration", 
                     f"{duration:.1f}s")
        with col3:
            total_frames = results.get('total_frames', 0)
            st.metric("🎬 Total Frames", 
                     total_frames)
        
        # Detection metrics
        st.subheader("📊 Detection Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🚗 Cars",
                value=results['detection_counts']['car'],
                delta=None
            )
        
        with col2:
            st.metric(
                label="🚛 Trucks",
                value=results['detection_counts']['truck'],
                delta=None
            )
        
        with col3:
            st.metric(
                label="🚌 Buses", 
                value=results['detection_counts']['bus'],
                delta=None
            )
        
        with col4:
            st.metric(
                label="🏍️ Motorcycles",
                value=results['detection_counts']['motorcycle'],
                delta=None
            )
        
        # Total detections
        st.metric(
            label="🎯 Total Vehicles Detected",
            value=results['total_detections'],
            delta=None
        )
        
        # Configuration details
        with st.expander("🔧 Analysis Configuration"):
            config = results['config']
            st.write(f"**Mode:** {config['mode']}")
            st.write(f"**Model:** {config['model_path']}")
            st.write(f"**Confidence:** {config['confidence']}")
            if config.get('url'):
                st.write(f"**Source URL:** {config['url']}")

    def cleanup_video_source(self):
        """Clean up video source and reset results."""
        cv2.destroyAllWindows()
        if hasattr(st.session_state, 'video_source') and st.session_state.video_source:
            st.session_state.video_source.release()
            st.session_state.video_source = None
        
        # Reset all session state results
        st.session_state.detection_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        st.session_state.total_detections = 0
        st.session_state.analysis_start_time = None
        st.session_state.current_session_results = None
        
        # Clear instance variables
        self.detection_history.clear()
        self.fps_history.clear()
        self.time_stamps.clear()
    
    def reset_stats(self):
        """Reset all statistics."""
        st.session_state.detection_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        st.session_state.total_detections = 0
        st.session_state.analysis_start_time = None
        st.session_state.analysis_error = None
        st.session_state.current_frame = None
        self.detection_history.clear()
        self.fps_history.clear()
        self.time_stamps.clear()
    
    def run(self):
        """Main dashboard run method - simplified OpenCV window approach."""
        self.setup_page()
        
        # Initialize session state (must be called from main thread)
        self.initialize_session_state()
        
        # Get configuration from sidebar
        config = self.create_sidebar()
        
        # Create main dashboard
        start_button, stop_button = self.create_main_dashboard()
        
        # Handle start/stop buttons
        if start_button:
            # Check current status
            status = self.check_analysis_status()
            
            if status['status'] == 'running':
                # Show running info
                st.info("🟢 Analysis is currently running in OpenCV window. Press 'Q' in the video window to stop.")
                if 'frames_processed' in status:
                    st.info(f"� Frames processed: {status['frames_processed']}, Detections: {status.get('detections', 0)}")
            else:
                # Start new analysis - validate configuration first
                if config['mode'] == 'youtube' and not config.get('url'):
                    st.error("❌ Please enter a YouTube URL")
                elif config['mode'] == 'file' and not config.get('file'):
                    st.error("❌ Please upload a video file")
                else:
                    # Check if model file exists
                    model_path = Path(config['model_path'])
                    if not model_path.exists():
                        st.error(f"❌ Model file not found: {config['model_path']}")
                        st.info("Available models in models/ folder:")
                        models_dir = Path("models")
                        if models_dir.exists():
                            for model_file in models_dir.glob("*.pt"):
                                st.info(f"  - {model_file.name}")
                    else:
                        # Clear previous session results when starting new analysis
                        st.session_state.current_session_results = None
                        
                        # Launch fire-and-forget analysis
                        st.info("🚀 Starting independent analysis process...")
                        st.info("💡 **Instructions:** OpenCV window will open shortly. Press 'Q' in the video window to stop.")
                        
                        if self.run_video_analysis_fire_and_forget(config):
                            st.success("✅ Analysis process launched successfully!")
                            st.info("🔄 Click 'Refresh Status' to check progress")
                            time.sleep(2)
                            st.rerun()
        
        if stop_button:
            status = self.check_analysis_status()
            
            if status['status'] == 'running':
                st.info("🛑 To stop analysis: Press 'Q' in the OpenCV video window")
            else:
                # Clear results
                self.cleanup_video_source()
                # Clear status file
                status_file = "logs/analysis_status.json"
                if os.path.exists(status_file):
                    os.remove(status_file)
                st.success("🧹 Results cleared and status reset")
                st.rerun()
        
        # Load and display session results
        st.divider()
        
        # Status section
        # Show current analysis status from file
        status = self.check_analysis_status()
        
        if status['status'] == 'running':
            st.write("🟢 **Status:** Analysis running independently")
            if 'frames_processed' in status:
                st.write(f"📊 Processed: {status['frames_processed']} frames")
        elif status['status'] == 'completed':
            st.write("✅ **Status:** Analysis completed - check results below")
            # Load results into session when analysis is completed
            if not st.session_state.current_session_results:
                try:
                    import glob
                    # Look for analysis result files, not status files
                    log_files = glob.glob("logs/analysis_????????_??????.json")
                    if log_files:
                        latest_file = max(log_files, key=lambda x: Path(x).stat().st_mtime)
                        with open(latest_file, 'r') as f:
                            results_data = json.load(f)
                            # Validate that this is actually a results file (has detection_counts)
                            if 'detection_counts' in results_data:
                                st.session_state.current_session_results = results_data
                except Exception:
                    pass
        elif status['status'] == 'error':
            st.write(f"❌ **Status:** Error - {status.get('error', 'Unknown error')}")
        else:
            st.write("🔴 **Status:** Ready for analysis")
        
        # Display results from current session
        self.display_analysis_results(st.session_state.current_session_results)
        
        # Show charts if we have data in current session
        if st.session_state.current_session_results and st.session_state.current_session_results['total_detections'] > 0:
            st.subheader("📈 Analysis Charts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Vehicle breakdown pie chart
                counts = st.session_state.current_session_results['detection_counts']
                if sum(counts.values()) > 0:
                    fig_pie = px.pie(
                        values=list(counts.values()),
                        names=list(counts.keys()),
                        title="🚗 Vehicle Type Distribution",
                        color_discrete_map={
                            'car': '#2E8B57',
                            'truck': '#FF6347', 
                            'bus': '#4169E1',
                            'motorcycle': '#FFD700'
                        }
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Summary stats chart
                vehicles = list(counts.keys())
                counts_list = list(counts.values())
                
                fig_bar = px.bar(
                    x=vehicles,
                    y=counts_list,
                    title="🎯 Vehicle Detection Summary",
                    color=vehicles,
                    color_discrete_map={
                        'car': '#2E8B57',
                        'truck': '#FF6347', 
                        'bus': '#4169E1',
                        'motorcycle': '#FFD700'
                    }
                )
                fig_bar.update_layout(
                    height=400,
                    xaxis_title="Vehicle Type",
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Check for completed analysis
        status = self.check_analysis_status()
        if status['status'] == 'completed':
            st.success("✅ Analysis completed! Results are displayed above.")
