
# Accident Detection Code Features

## Overview

The accident detection code is designed to monitor video feeds for accidents involving vehicles, bikes, and pedestrians. It uses a deep learning model to detect and classify accidents in real-time, triggering alerts and logging incidents for further analysis.

## Key Features

1. **Real-time Accident Detection**: Utilizes a YOLO model to detect accidents in video frames.
2. **Alert System**: Triggers visual and audible alerts upon detecting an accident.
3. **Snapshot Saving**: Captures and saves snapshots of detected accidents for review.
4. **Cooldown Mechanism**: Prevents repeated alerts for the same incident within a short timeframe.
5. **Logging**: Records details of detected accidents, including class and confidence level.

## Implementation Details

- **Model Integration**: Loads a pre-trained YOLO model for object detection.
- **Frame Processing**: Skips frames to optimize performance and reduce redundant processing.
- **Alert Management**: Uses a cooldown period to manage alert frequency.
- **Snapshot Management**: Saves a limited number of snapshots post-alert to conserve storage.
- **Logging**: Maintains a log file with timestamps and details of detected accidents.

## Code Structure

- **AccidentDetector Class**: Main class handling accident detection, alert management, and logging.
- **Methods**: Includes methods for processing video frames, handling alerts, saving snapshots, and logging detections.

## Usage

- **Initialization**: Create an instance of `AccidentDetector` with the model path and configuration settings.
- **Processing**: Call the `process_video` method with the video file path to start detection.
- **Output**: Generates alerts, saves snapshots, and logs accident details.
