"""
Status Management Utilities
Centralized status tracking for analysis processes.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path


def update_analysis_status(status, error=None, **kwargs):
    """Update analysis status file with comprehensive information."""
    try:
        Path("logs").mkdir(exist_ok=True)
        status_file = "logs/analysis_status.json"
        
        # Base status data
        status_data = {
            "status": status,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat()
        }
        
        # Add error if provided
        if error:
            status_data["error"] = str(error)
        
        # Add any additional data
        status_data.update(kwargs)
        
        # Add status-specific data
        if status == "starting":
            status_data.update({
                "message": "Analysis initialization in progress..."
            })
        elif status == "running":
            status_data.update({
                "message": "Analysis running in independent process"
            })
        elif status == "completed":
            status_data.update({
                "message": "Analysis completed successfully"
            })
        elif status == "error":
            status_data.update({
                "message": f"Analysis failed: {error}" if error else "Analysis failed with unknown error"
            })
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
            
    except Exception as e:
        print(f"Failed to update status: {e}")


def set_analysis_starting(**kwargs):
    """Mark analysis as starting with optional metadata."""
    update_analysis_status("starting", **kwargs)


def set_analysis_running(**kwargs):
    """Mark analysis as running with optional metadata."""
    update_analysis_status("running", **kwargs)


def set_analysis_completed(**kwargs):
    """Mark analysis as completed with optional metadata."""
    update_analysis_status("completed", **kwargs)


def set_analysis_error(error, **kwargs):
    """Mark analysis as failed with error details."""
    update_analysis_status("error", error=error, **kwargs)


def get_analysis_status():
    """Get current analysis status from file."""
    try:
        status_file = "logs/analysis_status.json"
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
        return {'status': 'none'}
    except Exception:
        return {'status': 'none'}


def clear_analysis_status():
    """Clear analysis status file."""
    try:
        status_file = "logs/analysis_status.json"
        if os.path.exists(status_file):
            os.remove(status_file)
    except Exception:
        pass
