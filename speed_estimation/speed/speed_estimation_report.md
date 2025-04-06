
# Speed Estimation Code Features

## Overview

The speed estimation code is designed to track and count objects, such as vehicles, in real-time video feeds. It utilizes computer vision and deep learning techniques to detect and classify objects, estimate their speed, and log this information for analysis.

## Key Features

1. **Object Detection and Tracking**: Uses a deep learning model to detect and track objects in video frames.
2. **Speed Estimation**: Calculates the speed of tracked objects in km/h based on their movement between frames.
3. **Region-based Counting**: Counts objects entering or exiting defined regions in the video frame.
4. **Data Logging**: Saves tracking data, including speed and direction, to a CSV file for further analysis.
5. **Real-time Visualization**: Displays counts and speed information on the video feed using annotations.

## Implementation Details

- **CSV Data Storage**: Logs data with headers such as Track ID, Label, Action, Speed, Class, Date, and Time.
- **Counting Mechanism**: Supports both linear and polygonal region-based counting.
- **Visualization**: Uses OpenCV for drawing bounding boxes and annotations on the video feed.
- **Configurable Display**: Allows toggling the display of "IN" and "OUT" counts.

## Code Structure

- **ObjectCounter Class**: Main class handling object detection, tracking, speed estimation, and data logging.
- **Methods**: Includes methods for initializing regions, counting objects, saving data to CSV, and displaying counts.

## Usage

- **Initialization**: Create an instance of `ObjectCounter` with desired configurations.
- **Processing**: Call the `count` method with video frames to process and annotate the feed.
- **Output**: Generates annotated video frames with object counts and speed information.
