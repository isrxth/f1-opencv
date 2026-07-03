# f1-opencv

F1 lane-tracking prototype built with OpenCV.

The script reads `sample.mp4`, crops the frame, extracts features inside a road ROI, tracks them with Lucas-Kanade optical flow, and estimates a center line from the tracked points.

## Preview



https://github.com/user-attachments/assets/a3bf0889-733d-46e6-8d97-ae44a483a00e



## Features

- Optical-flow tracking with OpenCV
- Region-of-interest masking for road features
- Per-slice lane center estimation
- Smoothed centerline fitting for downstream steering logic

## Requirements

- Python 3.14 or compatible
- `opencv-python`
- `numpy`

The repository includes a local virtual environment folder named `f1cv/` for development, but you can use any environment you prefer.

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install opencv-python numpy
```

## Run

```bash
python main.py
```

## Controls

- Press `d` to close the OpenCV window and stop the script.

## Notes

- The current implementation is tuned for `sample.mp4` and crops the frame with hard-coded values.
- `output.mp4` is a showcase video for the current result of the pipeline.
- The script computes a steering angle internally, but it does not currently render or output it.
- If you change the input video, you may need to adjust the crop and ROI polygon in `main.py`.
