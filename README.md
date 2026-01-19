# RTSP Camera Testing

This repository is a personal sandbox for experimenting with RTSP camera feeds. It includes quick scripts as well as a more complete surveillance prototype with AI-based person detection, alerting, and zone editing.

## What’s inside

- Standalone scripts for quick RTSP tests, gesture/overlay prototypes, and validations.
- `proj1_rtsp_surveillance/`: a multi-threaded surveillance prototype with:
  - RTSP capture (with webcam fallback).
  - YOLO-based person detection and intrusion zones.
  - Alert logging and snapshot capture.
  - Optional face blur for privacy.

## Getting started

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   Or run `venv_create.bat` on Windows (Python 3.11 recommended).

2. Update any script with your RTSP URL or device details, then run it:

   ```bash
   python open_rtsp_cam.py
   ```

3. To run the surveillance prototype:

   ```bash
   python proj1_rtsp_surveillance/core/main.py
   ```

   For the CustomTkinter GUI variant:

   ```bash
   python proj1_rtsp_surveillance/core/gui_main.py
   ```

## Notes

- This repo is intentionally exploratory and may include quick experiments or one-off tests.
- Some scripts may require additional hardware or optional dependencies (e.g., CustomTkinter, GPU drivers).
