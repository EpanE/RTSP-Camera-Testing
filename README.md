# RTSP Camera Testing

This repository is a personal sandbox for testing and exploring an RTSP camera in my office. It contains small projects and experiments built on top of the RTSP stream, such as gesture controls, overlays, and surveillance-style utilities.

## What’s inside

- Python scripts that connect to the RTSP camera and process the live feed.
- Prototype projects that experiment with computer vision interactions.
- Utility scripts for quick tests and validations.

## Getting started

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   or 

   Run the venv_create.bat (if you have python version 3.11)

2. Update any script with your RTSP URL, then run the script you want to test:

   ```bash
   python open_rtsp_cam.py
   ```

## Notes

- This repo is intentionally exploratory and may include quick experiments or one-off tests.
- Some scripts may require additional hardware, like a depth camera or gesture sensor.
