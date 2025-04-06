
# Wrong Direction Model Detector

## Overview

The wrong direction model detector is designed to identify vehicles moving in the wrong direction within specified areas of a video feed. It uses a YOLO model for object detection and tracking, and monitors predefined regions to detect and log wrong-way movements.

## Key Features

1. **Real-time Detection**: Monitors video feeds in real-time to detect vehicles moving in the wrong direction.
2. **Region-based Monitoring**: Defines specific areas to track vehicle movements.
3. **Image Capture**: Saves images of vehicles detected moving in the wrong direction.
4. **Visual Annotations**: Annotates the video feed with bounding boxes and labels for wrong-way vehicles.
5. **Persistent Tracking**: Tracks vehicles across frames to maintain consistent identification.

## Implementation Details

- **YOLO Model**: Utilizes YOLOv8 for object detection and tracking.
- **Monitored Areas**: Defines polygonal areas (AREA1, AREA2) to monitor vehicle movements.
- **Status Tracking**: Maintains the status of each vehicle to determine wrong-way movements.
- **Image Saving**: Captures and saves images of vehicles detected moving in the wrong direction.
- **Visualization**: Uses OpenCV and cvzone for drawing annotations and polygons on the video feed.

## Code Structure

- **Main Script (`main.py`)**: Handles video processing, YOLO model integration, and real-time detection.
- **Wrong Direction Detection (`wrong_direction.py`)**: Contains the logic for detecting wrong-way movements and updating vehicle status.

## Usage

- **Initialization**: Load the YOLO model and configure the video source.
- **Processing**: Run the detection loop to process video frames and detect wrong-way movements.
- **Output**: Displays annotated video feed and saves images of wrong-way vehicles.
