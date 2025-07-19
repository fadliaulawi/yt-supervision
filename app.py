#!/usr/bin/env python3
"""
Vehicle Detection Dashboard - Main Application
Streamlit web application for interactive vehicle detection and analysis.

This is the main entry point for the dashboard. Launch with:
    streamlit run app.py

Features:
- Real-time YouTube video/stream analysis
- Video file upload and analysis  
- Webcam live detection
- Interactive charts and metrics
- Model selection and configuration
"""

from modules.dashboard_core import StreamlitDashboard

def main():
    """Main function to run the Streamlit dashboard."""
    # Initialize and run dashboard
    dashboard = StreamlitDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
