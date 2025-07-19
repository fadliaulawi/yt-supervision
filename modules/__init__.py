"""
Vehicle Detection Dashboard Modules
Core functionality organized into modular components.
"""

from .config import Config
from .utils import DashboardUtils
from .video_detector import VideoDetector
from .youtube_watcher import YouTubeVideoWatcher
from .dashboard_core import StreamlitDashboard

__all__ = ['StreamlitDashboard', 'Config', 'DashboardUtils', 'VideoDetector', 'YouTubeVideoWatcher']