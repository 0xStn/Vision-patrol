# Vision Patrol - AI-Powered Traffic Monitoring System

## Acknowledgments 🙏

This graduation project represents the culmination of our collaborative efforts in developing **Vision Patrol**, an advanced AI-powered traffic monitoring system. We would like to extend our heartfelt gratitude to our team members who made this project possible:

- **Arwa Osama** - For her invaluable contributions in the Speed violation detector and Wrong direction models 
- **John Mamdouh** - For his invaluable contributions in Traffic detection and helmet violation models 

Together, we have created a comprehensive solution that bridges the gap between artificial intelligence and real-world traffic management applications.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [AI Models & Computer Vision](#ai-models--computer-vision)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

**Vision Patrol** is a **Graduation Project** focusing on the **AI and Computer Vision** aspects of traffic monitoring. It's a comprehensive real-time traffic violation detection system that leverages advanced computer vision techniques and artificial intelligence to monitor traffic conditions, detect various types of violations, and provide real-time alerts through a robust API system.

The system processes video feeds from multiple sources (IP cameras, local video files, YouTube streams) and uses state-of-the-art AI models to detect traffic violations, accidents, and safety issues with high accuracy and minimal false positives.

## ✨ Features

### 🚗 **Advanced Traffic Violation Detection**
- **Red Light Violations**: Intelligent detection of vehicles running red lights at intersections
- **Speed Limit Enforcement**: Real-time speed calculation and violation alerts
- **Wrong Direction Detection**: Identifies vehicles traveling against traffic flow
- **Parking Violations**: Detects unauthorized parking in restricted zones
- **Helmet Safety Monitoring**: Ensures motorcycle riders comply with helmet regulations
- **Comprehensive Accident Detection**: AI-powered detection of 7 different accident types

### 🎥 **Multi-Source Video Processing**
- **Flexible Input Sources**: Support for IP cameras, local video files, and YouTube streams
- **Multiprocessing Architecture**: Parallel processing of multiple video feeds for optimal performance
- **Real-time Analysis**: Live video processing with configurable frame rates and detection intervals
- **Smart Area Management**: Customizable detection zones for different violation types

### 🌐 **RESTful API & Web Interface**
- **FastAPI Backend**: Modern, high-performance API with automatic documentation
- **WebSocket Support**: Real-time communication for live updates and alerts
- **Interactive Web UI**: Browser-based monitoring and configuration interface
- **Area Configuration Tools**: Visual tools for defining detection zones

### 📊 **Intelligent Data Management**
- **Violation Logging**: Comprehensive records with timestamps, evidence, and metadata
- **Automatic Evidence Collection**: Snapshot capture for violation documentation
- **License Plate Recognition**: OCR-based vehicle identification and tracking
- **Smart Alert System**: Configurable cooldown periods and threshold-based alerting

## 🏗️ System Architecture

### **Computer Vision Pipeline**
```
Video Input → Frame Processing → Object Detection → Violation Analysis → Alert Generation
     ↓              ↓                    ↓               ↓              ↓
  Multi-source → Preprocessing → YOLOv8 Models → Rule Engine → API/WebSocket
```

### **Core Components**
- **Video Processor**: Multiprocessing video analysis engine
- **AI Detection Models**: Specialized models for each violation type
- **Area Manager**: Configurable detection zone management
- **Violation Manager**: Alert generation and evidence collection
- **API Server**: RESTful endpoints and WebSocket communication

## 🤖 AI Models & Computer Vision

### **Primary Detection Models**
- **YOLOv8**: State-of-the-art object detection for vehicles, pedestrians, and traffic elements
- **Custom Trained Models**: Specialized neural networks for each violation type
- **Accident Detection Model**: Multi-class classifier for accident type identification
- **Speed Estimation**: Computer vision-based speed calculation using object tracking

### **Computer Vision Techniques**
- **Object Tracking**: Multi-object tracking for speed calculation and behavior analysis
- **Optical Character Recognition**: EasyOCR for license plate text extraction
- **Motion Analysis**: Frame differencing and optical flow for movement detection
- **Geometric Transformations**: Perspective correction for accurate measurements

### **Supported Violation Types**
1. **Traffic Light Violations** (Red light running)
2. **Speed Violations** (Exceeding speed limits)
3. **Direction Violations** (Wrong-way driving)
4. **Parking Violations** (Illegal parking)
5. **Safety Violations** (Helmet non-compliance)
6. **Accident Detection** (7 types: car-car, car-bike, bike-person, etc.)

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU (recommended for real-time processing)
- Webcam or IP camera access
- Minimum 8GB RAM (16GB recommended)

### Setup Instructions

1. **Clone the Repository**
```bash
git clone <repository-url>
cd API_Updates_20
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download AI Models**
```bash
# YOLOv8 model will be downloaded automatically on first run
# Custom models are included in the Models/ directory
```

4. **Configure Video Sources**
Edit `Main_mp.py` to configure your video sources:
```python
video_sources = [
    {
        "id": "camera1", 
        "source": "your_video_source.mp4",  # or IP camera URL
        "use_stream": False,
        "location": "Main Street Intersection",
        "coordinates": {"lat": 40.7128, "lng": -74.0060},
        "speed_limit": 50
    }
]
```

## 📖 Usage

### **Option 1: Direct Video Processing**
```bash
python Main_mp.py
```

### **Option 2: API Server Mode**
```bash
python api_server.py
```
Then open your browser to `http://localhost:8000` for the web interface.

### **Basic Configuration**
1. **Set Detection Areas**: Use the web interface to define detection zones
2. **Configure Thresholds**: Adjust detection sensitivity for each violation type
3. **Set Alert Parameters**: Configure cooldown periods and notification settings

## 📡 API Documentation

### **Key Endpoints**

#### **Video Management**
- `POST /api/videos/start` - Start video processing
- `POST /api/videos/stop` - Stop video processing
- `GET /api/videos/status` - Get processing status

#### **Violation Detection**
- `GET /api/violations` - Retrieve violation records
- `POST /api/areas` - Configure detection areas
- `GET /api/areas/{area_id}` - Get area configuration

#### **Real-time Communication**
- `WebSocket /ws/alerts` - Real-time violation alerts
- `WebSocket /ws/status` - System status updates

### **WebSocket Events**
```javascript
// Connect to real-time alerts
const ws = new WebSocket('ws://localhost:8000/ws/alerts');
ws.onmessage = function(event) {
    const alert = JSON.parse(event.data);
    console.log('New violation:', alert);
};
```

## 📁 Project Structure

```
├── 📁 Models/                    # AI detection models
│   ├── 📁 Accident_det/         # Accident detection model
│   ├── 📁 Speed Model/          # Speed violation detection
│   ├── 📁 Traffic_violation/    # Traffic light violations
│   ├── 📁 Parking Model/        # Parking violation detection
│   ├── 📁 Bike_Violations/      # Helmet safety detection
│   └── 📁 Wrong_dir/            # Wrong direction detection
├── 📁 Processing Models/         # Core processing modules
│   ├── VideoProcessorMP.py      # Main video processor
│   ├── VideoReader.py           # Video input handler
│   └── areas.py                 # Area management
├── 📁 Violation_Proc/           # Violation management
│   ├── violation_manager.py     # Violation logging
│   └── accident_alert_manager.py # Accident alerts
├── 📁 static/                   # Web interface
│   ├── index.html               # Main interface
│   ├── config.html              # Configuration page
│   └── 📁 js/                   # JavaScript modules
├── 📁 area_configs/             # Detection area configurations
├── 📁 snapshots/                # Evidence snapshots
├── 📁 Testing Vids/             # Sample test videos
├── api_server.py                # FastAPI server
├── Main_mp.py                   # Main processing script
└── requirements.txt             # Dependencies
```

## ⚙️ Configuration

### **Detection Parameters**
Configure detection sensitivity in `config.conf`:
```ini
[DETECTION]
confidence_threshold = 0.7
speed_limit_tolerance = 10
accident_detection_enabled = true
helmet_detection_enabled = true
```

### **Area Configuration**
Define detection areas using the web interface or programmatically:
```python
area_manager.add_area(
    area_type=AreaType.TRAFFIC_LIGHT,
    coordinates=[(100, 200), (300, 400)],
    name="Main Intersection"
)
```

## 🧪 Testing

### **Sample Test Videos**
Test videos are provided in the `Testing Vids/` directory:
- Traffic violations
- Accident scenarios
- Speed limit violations
- Parking violations

### **Running Tests**
```bash
# Test with sample video
python Main_mp.py

# API endpoint testing
python -m pytest tests/ -v
```

## 🤝 Contributing

This is a graduation project, but we welcome feedback and suggestions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

This project is developed as part of our graduation requirements. All rights reserved by the development team.

---

## 🎓 Project Information

**Project Type**: Graduation Project - AI & Computer Vision  
**Project Name**: Vision Patrol  
**Focus Area**: Traffic Monitoring and Violation Detection  
**Technology Stack**: Python, OpenCV, PyTorch, FastAPI, YOLOv8  
**Team**: Computer Vision and AI Specialists  

---

**Developed with ❤️ by the Vision Patrol Team**
