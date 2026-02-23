# Vision Patrol — Processing Pipeline Report

**Date:** February 23, 2026  
**Project:** Vision Patrol — AI-Powered Traffic Monitoring System  
**Report Scope:** Video input, frame processing, area management, and orchestration architecture  

---

## 1. Architecture Overview

Vision Patrol follows a **modular monolith with multiprocessing** architecture. A single Python application is organized into distinct modules (video input, detection models, area management, violation management), all running in one deployment unit. Parallel processing is achieved through Python's `multiprocessing.Process` — each video source gets its own OS process with an independent YOLO model and detector instances.

### High-Level Data Flow

```
┌─────────────────┐    ┌──────────────┐    ┌───────────────────┐
│  Video Sources   │───▶│  VideoReader  │───▶│ VideoProcessorMP  │
│  (File/RTSP/YT)  │    │  (cv2/CamGear)│    │  (multiprocessing) │
└─────────────────┘    └──────────────┘    └────────┬──────────┘
                                                     │
                              ┌───────────────────────┼───────────────────────┐
                              ▼                       ▼                       ▼
                    ┌──────────────┐      ┌──────────────────┐    ┌──────────────┐
                    │  YOLOv8 Track │      │   AreaManager    │    │   Detectors   │
                    │  (per frame)  │─────▶│ (zone/line check) │───▶│ (6 modules)   │
                    └──────────────┘      └──────────────────┘    └──────┬───────┘
                                                                         │
                              ┌──────────────────────────────────────────┘
                              ▼
                    ┌───────────────────┐    ┌──────────────────┐
                    │ ViolationManager  │───▶│  License Plate   │
                    │ (record + route)  │    │  Recognition     │
                    └────────┬──────────┘    └──────────────────┘
                             │
                    ┌────────┴──────────┐
                    ▼                   ▼
          ┌──────────────┐    ┌──────────────────┐
          │  JSON/CSV    │    │  FastAPI + WS     │
          │  Persistence │    │  Real-time Push   │
          └──────────────┘    └──────────────────┘
```

---

## 2. VideoReader — Multi-Source Input

**File:** `Processing Models/VideoReader.py` (361 lines)  
**Class:** `VideoReader`

### Purpose

Provides a unified interface for reading video frames from three source types: local files, RTSP/IP camera streams, and YouTube URLs. All downstream components receive frames the same way regardless of source.

### Supported Sources

| Source Type | Detection Method                         | Backend                   |
| ----------- | ---------------------------------------- | ------------------------- |
| Local file  | Path exists on disk                      | OpenCV `cv2.VideoCapture` |
| RTSP stream | URL starts with `rtsp://`                | OpenCV `cv2.VideoCapture` |
| YouTube     | URL contains `youtube.com` or `youtu.be` | CamGear / pytube / yt-dlp |

### YouTube Fallback Chain

YouTube sources use a 4-step fallback mechanism for maximum compatibility:

```
Step 1: CamGear with yt-dlp backend
  → Fail? →
Step 2: Direct CamGear (default backend)
  → Fail? →
Step 3: pytube stream URL → OpenCV
  → Fail? →
Step 4: Direct OpenCV on URL
```

### Thread Safety

- All `read()`, `reset()`, and `release()` methods are protected with `threading.Lock`
- Buffer size set to 1 frame for minimum latency (avoids frame queue buildup)
- Implements context manager protocol (`__enter__` / `__exit__`) for safe resource cleanup

### Key Methods

| Method             | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `read()`           | Returns `(success, frame)` tuple — thread-safe            |
| `reset()`          | Re-initializes video capture from beginning (for looping) |
| `release()`        | Closes all resources with lock protection                 |
| `is_youtube_url()` | Static URL pattern check                                  |

---

## 3. VideoProcessorMP — Central Orchestrator

**File:** `Processing Models/VideoProcessorMP.py` (676 lines)  
**Class:** `VideoProcessorMP(multiprocessing.Process)`

### Role

This is the **heart of the system**. Each instance runs as an independent OS process, handling one video source end-to-end: reading frames, running YOLO detection, dispatching to all 6 specialized detectors, and managing the display window.

### Initialization Flow

```
VideoProcessorMP.__init__(source_config, stop_event)
  └── run()  (called when process starts)
       ├── Load YOLOv8s model (yolov8s.pt)
       ├── Create VideoReader(source)
       ├── Create AreaManager(stream_id)
       ├── Create ViolationManager(stream_id, location, coordinates)
       ├── Create AccidentAlertManager(stream_id, location, coordinates)
       ├── Create AccidentDetector(stream_id, accident_alert_manager)
       ├── Create SpeedDetector(stream_id, violation_manager)
       ├── Create ParkingDetector(stream_id, violation_manager)
       ├── Create WrongDirectionDetector(stream_id, violation_manager)
       ├── Create TrafficViolationDetector(stream_id, violation_manager)
       ├── Create HelmetViolationDetector(stream_id, violation_manager)
       └── Enter main frame loop
```

### Per-Frame Processing Pipeline

Every frame passes through this fixed-order detection sequence:

```
1. Read frame from VideoReader
2. YOLO model.track(frame, persist=True)
   → Filter for vehicle/person classes [1,2,3,4,5,7]
   → Update tracked objects dict (center points, boxes, confidence)
   → Remove stale tracks (>30 frames unseen)
3. AccidentDetector.detect(frame, tracked_objects)
4. SpeedDetector.calculate_speed(frame, tracked_objects, fps)
5. ParkingDetector.update_vehicles(frame, tracked_objects, area_manager)
6. WrongDirectionDetector.detect(frame, tracked_objects, area_manager)
7. TrafficViolationDetector.detect(frame, tracked_objects, area_manager)
8. HelmetViolationDetector.detect(frame, tracked_objects, area_manager)
9. AreaManager.draw_areas(frame)
10. Display frame (cv2.imshow with scaling)
```

**Conditional execution:** Each detector only runs if its corresponding area type has defined zones in the `AreaManager`. This saves compute when not all violations are being monitored.

### Tracked Object Management

Each tracked vehicle/person maintains state across frames:

```python
tracked_objects[vehicle_id] = {
    'center_points': deque(maxlen=10),  # Last 10 center positions
    'crossed_lines': set(),              # Line IDs crossed
    'box': (x1, y1, x2, y2),           # Current bounding box
    'class_id': int,                     # COCO class
    'confidence': float,                 # Detection confidence
    'last_seen': int                     # Frame counter
}
```

Objects are removed after **30 frames** without detection (track timeout).

### Module Loading Pattern

Because some model directories contain spaces (e.g., "Speed Model", "Parking Model"), the project uses `importlib.util.spec_from_file_location()` to load modules by absolute file path instead of standard Python imports:

```python
speed_module_path = os.path.join(speed_dir, "speed_detector.py")
spec = importlib.util.spec_from_file_location("speed_detector", speed_module_path)
speed_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed_module)
SpeedDetector = speed_module.SpeedDetector
```

### Display & Controls

| Key         | Action                                      |
| ----------- | ------------------------------------------- |
| `q` / `ESC` | Exit                                        |
| `a`         | Toggle accident detection                   |
| `t`         | Toggle traffic violation detection          |
| `h`         | Toggle helmet detection                     |
| `r`         | Toggle accident alert response              |
| `l`         | Toggle video looping                        |
| `1`–`7`     | Select area type for interactive definition |
| `c`         | Clear current area being drawn              |
| `d`         | Clear all areas                             |
| `+` / `-`   | Adjust display scale (default 0.65)         |

### Multiprocessing Considerations

- Each process has its **own YOLO model instance** — no shared GPU memory
- `multiprocessing.Event` is used for coordinated shutdown across all processes
- Graceful shutdown: stop signal → 5-second timeout → force terminate → `cv2.destroyAllWindows()`
- Non-stream videos auto-restart with `VideoReader.reset()` when video ends

---

## 4. AreaManager — Zone & Line Management

**File:** `Processing Models/areas.py`  
**Classes:** `AreaType` (Enum), `AreaManager`

### 7 Area Types

| Type              | Shape               | Use Case                                     |
| ----------------- | ------------------- | -------------------------------------------- |
| `DETECTION`       | Polygon (≥3 points) | General detection zone (helmet checking ROI) |
| `SPEED`           | Line (2 points)     | Speed measurement trigger line               |
| `WRONG_DIRECTION` | Line (2 points)     | Directional gate lines (paired)              |
| `PARKING`         | Polygon (≥3 points) | No-parking zone boundary                     |
| `TRAFFIC_LINE`    | Line (2 points)     | Red-light violation crossing line            |
| `TRAFFIC_SIGN`    | Polygon (≥3 points) | Traffic light ROI for HSV detection          |
| `CUSTOM`          | Polygon/Line        | Reserved for future use                      |

### Core Algorithms

**Point-in-Polygon Test:**  
Uses OpenCV's `cv2.pointPolygonTest()` which implements the ray-casting algorithm. Returns positive values for points inside the polygon.

**Box-in-Area Check:**  
Checks whether any of a bounding box's four corners are inside the polygon, OR if any bounding box edge intersects a polygon edge, OR if the polygon is fully contained within the bounding box.

**Line Crossing Detection:**  
Uses the **cross product method** to determine if a movement vector (from position $p_1$ to $p_2$) crosses a defined line segment $(l_1, l_2)$:

$$\text{cross}(v_1, v_2) = v_{1x} \cdot v_{2y} - v_{1y} \cdot v_{2x}$$

The sign of the cross product indicates which side of the line the vehicle has moved to, giving **directional crossing** information.

### Persistence

Area configurations are persisted as JSON files:

```
area_configs/areas_{stream_id}.json
```

Each file contains an array of area definitions with type, name, and coordinate points.

### Interactive Definition

Areas can be defined interactively via OpenCV mouse callbacks:
- **Left-click:** Add a point
- **Right-click:** Complete the current area
- **Keyboard 1–7:** Select area type

Or via the web-based Fabric.js canvas in `config.html` (see API section).

### API Integration

The `AreaManager` exposes methods for the API layer:
- `export_areas()` — Returns all areas as JSON-serializable dicts
- `import_areas(data)` — Loads areas from API-provided configuration
- `clear_all_areas()` — Removes all defined areas

---

## 5. Entry Points

### Main_mp.py (Standalone CLI)

A minimal entry point that creates one `VideoProcessorMP` process per video source and manages coordinated shutdown:

```
Main_mp.py
  ├── Create multiprocessing.Event (shared stop signal)
  ├── For each source in video_sources list:
  │     └── Create & start VideoProcessorMP process
  ├── Main loop: poll cv2.waitKey for ESC/Q
  └── Shutdown: stop_event.set() → join(timeout=5) → terminate()
```

**Limitation:** `video_sources` list is hardcoded — no CLI arguments or config file reading.

### api_server.py (FastAPI Server)

The production entry point that wraps `VideoProcessorMP` in an API-controlled lifecycle:

- Uses **threading** (not multiprocessing) for API-managed processors via `APIVideoProcessor`
- 30+ REST endpoints for sources, areas, violations, manual reviews, accidents, media
- WebSocket channels for real-time violation/accident/system status broadcasting
- Static file serving for the web dashboard
- Video frame streaming via WebSocket (base64 JPEG at ~5 FPS)

---

## 6. Multiprocessing vs. Threading

The system uses **two different execution models** depending on the entry point:

| Entry Point     | Execution Model                              | Reason                                 |
| --------------- | -------------------------------------------- | -------------------------------------- |
| `Main_mp.py`    | `multiprocessing.Process`                    | True parallel processing, bypasses GIL |
| `api_server.py` | `threading.Thread` (via `APIVideoProcessor`) | Easier API integration, shared memory  |

### Trade-offs

| Aspect          | Multiprocessing (Main_mp)                 | Threading (API Server)                |
| --------------- | ----------------------------------------- | ------------------------------------- |
| GIL impact      | None (separate processes)                 | Partial blocking                      |
| Memory          | Each process loads own model (~50MB+ GPU) | Shared model memory possible          |
| Communication   | IPC required (Events)                     | Direct object access                  |
| Crash isolation | Process crash doesn't affect others       | Thread crash can corrupt shared state |
| Startup cost    | Higher (fork + model load)                | Lower                                 |

---

## 7. Summary Assessment

### Strengths

1. **Clean separation** — Each video source runs independently with its own detection pipeline
2. **Flexible input** — Supports files, RTSP, and YouTube with automatic fallback
3. **Conditional execution** — Detectors only run when their areas are configured
4. **Interactive area definition** — Both OpenCV mouse callback and web-based Fabric.js canvas
5. **Graceful shutdown** — Coordinated multi-process termination with timeouts

### Areas for Improvement

1. **No frame queuing** — Frames are processed synchronously; frame drops possible under load
2. **No batch inference** — Each process runs independent YOLO inference; no GPU batching across streams
3. **No frame skip strategy** — Every frame is processed regardless of system load
4. **Config file unused** — `config.conf` exists with proper sections but is never actually parsed
5. **Spaces in directory names** — Requires `importlib.util` workaround; could be solved by renaming directories
6. **No GPU memory management** — Multiple processes competing for GPU resources without coordination
