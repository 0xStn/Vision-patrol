# Vision Patrol — Backend Developer Report

## Implementation Guide for System Architecture, API, and Services

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Current Architecture Analysis](#2-current-architecture-analysis)
3. [Complete API Reference](#3-complete-api-reference)
4. [Data Models & Schemas](#4-data-models--schemas)
5. [WebSocket Real-Time Layer](#5-websocket-real-time-layer)
6. [Persistence Layer — Current vs Target](#6-persistence-layer--current-vs-target)
7. [Configuration System Overhaul](#7-configuration-system-overhaul)
8. [Authentication & Authorization Implementation](#8-authentication--authorization-implementation)
9. [Target Production Architecture](#9-target-production-architecture)
10. [Service Decomposition Plan](#10-service-decomposition-plan)
11. [Database Migration Guide](#11-database-migration-guide)
12. [Notification & Alerting Service](#12-notification--alerting-service)
13. [Containerization & Deployment](#13-containerization--deployment)
14. [Error Handling & Logging Strategy](#14-error-handling--logging-strategy)
15. [Testing Strategy](#15-testing-strategy)
16. [Implementation Roadmap](#16-implementation-roadmap)

---

## 1. System Overview

Vision Patrol is a real-time AI-powered traffic violation detection system. It processes video feeds (files, RTSP streams, YouTube) through multiple YOLOv8-based detection models, identifies violations, captures evidence, and serves results through a FastAPI REST+WebSocket API.

### Core Technical Stack

| Component        | Technology                   | Version   |
| ---------------- | ---------------------------- | --------- |
| Web Framework    | FastAPI                      | 0.103.1   |
| ASGI Server      | Uvicorn                      | 0.23.2    |
| Data Validation  | Pydantic v2                  | (bundled) |
| Object Detection | Ultralytics YOLOv8           | 8.0.196   |
| Video I/O        | OpenCV + vidgear (CamGear)   | 4.x       |
| OCR              | EasyOCR                      | 1.7.1     |
| Language         | Python                       | 3.11+     |
| Frontend         | Vanilla HTML/CSS/JS (static) | —         |

### Current Entry Points

```
api_server.py          →  FastAPI server (production entry point, threading model)
Main_mp.py             →  Multiprocessing entry point (development/desktop mode)
```

---

## 2. Current Architecture Analysis

### 2.1 Architecture Diagram — Current State

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT ARCHITECTURE                              │
│                          (Monolithic, Single Process)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────┐                         │
│   │            FastAPI Application                 │                         │
│   │         (api_server.py — 1640 lines)           │                         │
│   │                                                │                         │
│   │  ┌──────────────┐  ┌────────────────────────┐  │                         │
│   │  │  REST API     │  │  WebSocket Server      │  │                         │
│   │  │  30+ endpoints│  │  4 channels            │  │                         │
│   │  │              │  │  /ws/{channel}          │  │                         │
│   │  │              │  │  /ws/video/{source_id}  │  │                         │
│   │  └───────┬──────┘  └──────────┬─────────────┘  │                         │
│   │          │                    │                  │                         │
│   │  ┌───────▼────────────────────▼─────────────┐  │                         │
│   │  │         In-Memory State                   │  │                         │
│   │  │                                           │  │                         │
│   │  │  video_processors: Dict[str, Processor]   │  │                         │
│   │  │  manager: ConnectionManager (WS)          │  │                         │
│   │  │  notification_handler: NotificationHandler│  │                         │
│   │  └───────┬───────────────────────────────────┘  │                         │
│   └──────────┼──────────────────────────────────────┘                         │
│              │                                                                │
│   ┌──────────▼──────────────────────────────────────────────────────────────┐ │
│   │              APIVideoProcessor (per source)                             │ │
│   │              Runs in threading.Thread                                    │ │
│   │                                                                          │ │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │ │
│   │  │ VideoReader   │  │ YOLO Model   │  │ AreaManager  │                   │ │
│   │  │ (CamGear/    │  │ (yolov8s.pt) │  │ (JSON areas) │                   │ │
│   │  │  cv2.Video)   │  │              │  │              │                   │ │
│   │  └──────┬───────┘  └──────┬───────┘  └──────────────┘                   │ │
│   │         │                 │                                              │ │
│   │         ▼                 ▼                                              │ │
│   │  ┌──────────────────────────────────────────┐                           │ │
│   │  │        Detection Models (6 total)         │                           │ │
│   │  │                                           │                           │ │
│   │  │  AccidentDetector       (accident.pt)     │                           │ │
│   │  │  HelmetViolationDetector (helmet_ds.pt)   │                           │ │
│   │  │  SpeedDetector          (algorithmic)     │                           │ │
│   │  │  TrafficViolationDetector (HSV masking)   │                           │ │
│   │  │  ParkingDetector        (timer-based)     │                           │ │
│   │  │  WrongDirectionDetector (line-crossing)   │                           │ │
│   │  └──────────────┬───────────────────────────┘                           │ │
│   │                 │                                                        │ │
│   │                 ▼                                                        │ │
│   │  ┌──────────────────────────────────────────┐                           │ │
│   │  │   ViolationManager + AccidentAlertManager │                           │ │
│   │  │   LicensePlateRecognizer (synchronous)    │                           │ │
│   │  └──────────────┬───────────────────────────┘                           │ │
│   └─────────────────┼───────────────────────────────────────────────────────┘ │
│                     │                                                         │
│   ┌─────────────────▼────────────────────────────────────────────────────┐    │
│   │                    File-Based Persistence                             │    │
│   │                                                                       │    │
│   │  Violation_Proc/violations/                                           │    │
│   │  ├── unified_violations.json   ← All violations (append-only)        │    │
│   │  ├── unified_violations.csv    ← Same data, CSV format               │    │
│   │  ├── manual_reviews.json       ← Manual review queue                 │    │
│   │  ├── snapshots/                ← Violation evidence images           │    │
│   │  └── plates/                   ← Cropped license plate images        │    │
│   │                                                                       │    │
│   │  Violation_Proc/accident_alerts/                                      │    │
│   │  ├── all_accident_alerts.json  ← All accident records                │    │
│   │  └── *.jpg                     ← Accident snapshots                  │    │
│   │                                                                       │    │
│   │  area_configs/                                                        │    │
│   │  ├── areas_{source_id}.json    ← Per-camera area definitions         │    │
│   │  └── {source_id}.json          ← Area snapshots                      │    │
│   │                                                                       │    │
│   │  static/                       ← Frontend served by FastAPI          │    │
│   └──────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   ┌───────────────────────────────────────────────────────────────────────┐   │
│   │  config.conf (180 lines)  —  ⚠️ NEVER PARSED, 100% UNUSED           │   │
│   │  Contains: API, DB, YOLO, NOTIFICATION, WEBSOCKET sections           │   │
│   └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Critical Issues in Current Backend

| #   | Issue                               | Impact   | Location                                                       |
| --- | ----------------------------------- | -------- | -------------------------------------------------------------- |
| 1   | No authentication/authorization     | Critical | `api_server.py` — all endpoints                                |
| 2   | File-based JSON persistence         | High     | `violation_manager.py`, `accident_alert_manager.py`            |
| 3   | `config.conf` completely unused     | Medium   | All modules — values hardcoded                                 |
| 4   | `asyncio.run()` inside sync threads | High     | `api_server.py:APIViolationManager`, `APIAccidentAlertManager` |
| 5   | No request pagination               | Medium   | All list endpoints                                             |
| 6   | CORS allows all origins             | High     | `api_server.py` line ~35                                       |
| 7   | Duplicate endpoint definitions      | Low      | `GET/PUT /api/sources/{id}/settings` defined twice             |
| 8   | No input validation beyond Pydantic | Medium   | URL/source validation missing                                  |
| 9   | No structured logging               | Medium   | `print()` statements everywhere                                |
| 10  | Synchronous plate recognition       | High     | `violation_manager.py:_record_violation()` blocks detection    |
| 11  | No rate limiting                    | Medium   | All API endpoints                                              |
| 12  | No health check endpoint            | Low      | Missing `/health` or `/readiness`                              |

### 2.3 Request Flow — Current System

```
Client Request
     │
     ▼
┌─────────────────┐
│   FastAPI ASGI   │
│   (Uvicorn)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────────────────────────┐
│  CORS Middleware │────▶│  Allow ALL origins (*)               │
│                  │     │  Allow ALL methods                   │
│                  │     │  Allow ALL headers                   │
└────────┬────────┘     └─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Route Resolution                                       │
│                                                         │
│  Order matters:  API routes registered BEFORE static    │
│  Static mount happens in startup_event() after routes   │
│  ("/", StaticFiles, html=True)                          │
└────────┬────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────┐
         │  REST Endpoint                       │  WebSocket Endpoint
         ▼                                      ▼
┌─────────────────┐                  ┌─────────────────┐
│  Pydantic v2     │                  │  ConnectionMgr   │
│  Validation      │                  │  Channel-based   │
│  (Body, Path)    │                  │  accept/track    │
└────────┬────────┘                  └────────┬────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐                  ┌─────────────────┐
│  Handler Logic   │                  │  While True:     │
│  (in-process)    │                  │    receive msg   │
│                  │                  │    broadcast     │
│  Read/write JSON │                  │    ping/pong     │
│  files directly  │                  └─────────────────┘
└─────────────────┘
```

---

## 3. Complete API Reference

### 3.1 Endpoint Summary Table

| Method   | Path                                              | Auth | Paginated | Description                                             |
| -------- | ------------------------------------------------- | ---- | --------- | ------------------------------------------------------- |
| `POST`   | `/api/sources`                                    | ❌    | ❌         | Add a new video source                                  |
| `GET`    | `/api/sources`                                    | ❌    | ❌         | List all video sources                                  |
| `DELETE` | `/api/sources/{source_id}`                        | ❌    | —         | Remove and stop a video source                          |
| `POST`   | `/api/sources/{source_id}/start`                  | ❌    | —         | Start processing a video source                         |
| `POST`   | `/api/sources/{source_id}/stop`                   | ❌    | —         | Stop processing a video source                          |
| `GET`    | `/api/sources/{source_id}/areas`                  | ❌    | ❌         | Get area configuration for source                       |
| `PUT`    | `/api/sources/{source_id}/areas`                  | ❌    | —         | Update area configuration                               |
| `POST`   | `/api/sources/{source_id}/areas/import`           | ❌    | —         | Import area configuration from JSON                     |
| `GET`    | `/api/sources/{source_id}/areas/export`           | ❌    | —         | Export area configuration as JSON                       |
| `POST`   | `/api/sources/{source_id}/areas/clear-all`        | ❌    | —         | Clear all areas for a source                            |
| `GET`    | `/api/sources/{source_id}/settings`               | ❌    | —         | Get model settings for source *(duplicated)*            |
| `PUT`    | `/api/sources/{source_id}/settings`               | ❌    | —         | Update model settings *(duplicated)*                    |
| `GET`    | `/api/sources/{source_id}/frame`                  | ❌    | —         | Capture a single video frame                            |
| `GET`    | `/api/violations`                                 | ❌    | ❌         | Get all violations (unfiltered)                         |
| `POST`   | `/api/violations`                                 | ❌    | ❌         | Get violations with filters                             |
| `GET`    | `/api/violations/{violation_id}`                  | ❌    | —         | Get specific violation details                          |
| `GET`    | `/api/violations/statistics`                      | ❌    | —         | Get violation statistics                                |
| `GET`    | `/api/violations/types/{type}`                    | ❌    | ❌         | Get violations by type                                  |
| `POST`   | `/api/manual-reviews`                             | ❌    | —         | Create manual review with violation                     |
| `GET`    | `/api/manual-reviews`                             | ❌    | ❌         | List all manual reviews                                 |
| `GET`    | `/api/manual-reviews/{review_id}`                 | ❌    | —         | Get specific manual review                              |
| `PUT`    | `/api/manual-reviews/{review_id}`                 | ❌    | —         | Update manual review (license plate)                    |
| `DELETE` | `/api/manual-reviews/{review_id}`                 | ❌    | —         | Delete a manual review                                  |
| `POST`   | `/api/accidents`                                  | ❌    | ❌         | Get accidents with filters                              |
| `GET`    | `/api/accidents/statistics`                       | ❌    | —         | Get accident statistics                                 |
| `GET`    | `/api/accidents/{accident_id}`                    | ❌    | —         | Get specific accident details                           |
| `GET`    | `/api/media/violations/{violation_id}`            | ❌    | —         | Get violation snapshot image                            |
| `GET`    | `/api/media/accidents/{accident_id}`              | ❌    | —         | Get accident snapshot image                             |
| `GET`    | `/api/media/plates/{violation_id}`                | ❌    | —         | Get license plate snapshot                              |
| `GET`    | `/api/media/manual-reviews/{review_id}/plate`     | ❌    | —         | Get manual review plate snapshot                        |
| `GET`    | `/api/media/manual-reviews/{review_id}/violation` | ❌    | —         | Get manual review violation snapshot                    |
| `GET`    | `/api/system/status`                              | ❌    | —         | Get system status                                       |
| `WS`     | `/ws/{channel}`                                   | ❌    | —         | Real-time event channel (violations, accidents, system) |
| `WS`     | `/ws/video/{source_id}`                           | ❌    | —         | Live video frame stream                                 |

### 3.2 Detailed Endpoint Specifications

#### 3.2.1 Video Source Management

##### POST /api/sources

Add a new video source (camera, file, or stream).

**Request Body:**
```json
{
  "source_id": "camera_01",              // Optional — auto-generated UUID if omitted
  "url": "rtsp://192.168.1.100:554/stream",
  "name": "Main Intersection Camera",
  "camera_location": "Cairo, Tahrir Square",
  "coordinates": { "lat": 30.0444, "lng": 31.2357 },
  "speed_limit": 60,
  "use_stream": true
}
```

**Pydantic Model: `VideoSource`**
```python
class VideoSource(BaseModel):
    source_id: Optional[str] = None
    url: str
    name: Optional[str] = None
    camera_location: Optional[str] = "Unknown"
    coordinates: Optional[Coordinates] = None
    speed_limit: Optional[int] = 60
    use_stream: Optional[bool] = False
```

**Response (201):**
```json
{
  "source_id": "camera_01",
  "url": "rtsp://192.168.1.100:554/stream",
  "name": "Main Intersection Camera",
  "status": "stopped",
  "camera_location": "Cairo, Tahrir Square",
  "coordinates": { "lat": 30.0444, "lng": 31.2357 }
}
```

**Internal Logic:**
1. If `source_id` is `None`, generates `uuid4()`.
2. Checks for duplicate `source_id` (returns 400).
3. Creates `APIVideoProcessor` instance stored in `video_processors[source_id]`.
4. Does NOT start processing — call `/start` separately.

---

##### GET /api/sources

List all registered video sources with their current status.

**Response (200):**
```json
[
  {
    "source_id": "camera_01",
    "url": "rtsp://...",
    "name": "Main Intersection Camera",
    "status": "running",
    "camera_location": "Cairo, Tahrir Square",
    "coordinates": { "lat": 30.0444, "lng": 31.2357 }
  }
]
```

**Note:** Status is one of `"stopped"`, `"starting"`, `"running"`, `"error"`.

---

##### DELETE /api/sources/{source_id}

Stop processing (if running) and remove the video source.

**Response (200):** `{ "message": "Video source removed: camera_01" }`

**Internal Logic:** Calls `processor.stop()` then removes from `video_processors` dict.

---

##### POST /api/sources/{source_id}/start

Start video processing for the specified source.

**Response (200):** `{ "message": "Video source started: camera_01" }`

**Internal Logic:**
- `APIVideoProcessor.start()` creates a `threading.Thread` target that:
  1. Instantiates `VideoProcessorMP` (NOT as a separate process — runs `run()` directly).
  2. Creates all 6 detection model instances.
  3. Enters frame-by-frame processing loop.

---

##### POST /api/sources/{source_id}/stop

Stop video processing for the specified source.

**Response (200):** `{ "message": "Video source stopped: camera_01" }`

---

#### 3.2.2 Area Configuration

##### GET /api/sources/{source_id}/areas

Returns the current area configuration for detection zones.

**Response (200):**
```json
{
  "source_id": "camera_01",
  "areas": {
    "detection": [
      { "points": [[100,200],[400,200],[400,600],[100,600]], "id": 0 }
    ],
    "speed": [
      { "points": [[0,300],[800,300]], "id": 0, "direction": "down" }
    ],
    "parking": [],
    "wrong_direction": [],
    "traffic_sign": [],
    "traffic_line": []
  }
}
```

##### PUT /api/sources/{source_id}/areas

Update area configuration. Expects the same structure as GET response `areas` field.

##### POST /api/sources/{source_id}/areas/import

Import area configuration from a JSON body.

##### GET /api/sources/{source_id}/areas/export

Export area configuration as downloadable JSON.

##### POST /api/sources/{source_id}/areas/clear-all

Clear all defined areas for the source.

---

#### 3.2.3 Model Settings

##### GET /api/sources/{source_id}/settings

Returns which detection models are enabled/disabled.

**Response (200):**
```json
{
  "models": {
    "accident_detection": true,
    "helmet_detection": true,
    "traffic_violation": true,
    "speed_detection": true,
    "parking_detection": true,
    "wrong_direction": true
  }
}
```

##### PUT /api/sources/{source_id}/settings

Toggle detection models on/off.

**Request Body:**
```json
{
  "models": {
    "helmet_detection": false,
    "speed_detection": true
  }
}
```

**⚠️ BUG:** This endpoint is defined TWICE in `api_server.py` — once as `async def` (lines ~1100-1120) and once as `def` (lines ~1600-1630). The second definition silently overrides the first. Fix: remove the duplicate.

---

#### 3.2.4 Violations

##### GET /api/violations

Returns ALL violations without filtering. Reads `unified_violations.json`.

**Response (200):** Array of violation objects.

##### POST /api/violations

Filter violations by criteria.

**Request Body (`ViolationFilter`):**
```json
{
  "violation_type": "speed",
  "camera_id": "camera_01",
  "date_from": "2024-01-01",
  "date_to": "2024-12-31",
  "has_license_plate": true
}
```

**Pydantic Model:**
```python
class ViolationFilter(BaseModel):
    violation_type: Optional[str] = None
    camera_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    has_license_plate: Optional[bool] = None
```

##### GET /api/violations/{violation_id}

Returns a single violation by ID. ⚠️ Scans entire JSON array (O(n)).

##### GET /api/violations/statistics

Returns aggregated counts by type, source, and date.

**Response (200):**
```json
{
  "total": 142,
  "by_type": { "speed": 80, "helmet": 30, "traffic_light": 20, "parking": 12 },
  "by_source": { "camera_01": 100, "camera_02": 42 },
  "by_date": { "2024-06-15": 25, "2024-06-16": 30 }
}
```

##### GET /api/violations/types/{type}

Filter violations by type (e.g., `speed`, `helmet`, `traffic_light`, `parking`, `wrong_direction`).

---

#### 3.2.5 Manual Reviews

##### POST /api/manual-reviews

Create a manual review entry when automated license plate recognition fails. The violation data is embedded.

**Request Body (`ManualReviewWithViolationRequest`):**
```json
{
  "violation_data": {
    "violation_id": "000001_abc123",
    "violation_type": "speed",
    "camera_id": "camera_01",
    "date": "2024-06-15",
    "time": "14:30:00",
    "snapshot_path": "snapshots/000001_abc123.jpg"
  },
  "reason": "Plate not readable",
  "notes": "Partially obscured by another vehicle"
}
```

**Response (`ManualReviewResponse`):**
```json
{
  "review_id": "mr_abc123",
  "status": "pending",
  "violation_data": { ... },
  "reason": "Plate not readable",
  "notes": "Partially obscured by another vehicle",
  "license_plate": null,
  "reviewer_name": null,
  "created_at": "2024-06-15T14:30:00",
  "updated_at": "2024-06-15T14:30:00"
}
```

##### PUT /api/manual-reviews/{review_id}

Update a manual review — primarily for adding the license plate after human review.

**Request Body (`ManualReviewUpdate`):**
```json
{
  "license_plate": "ABC 1234",
  "notes": "Plate confirmed from zoomed snapshot",
  "reviewer_name": "Officer Ahmed"
}
```

**Internal Logic:**
1. Updates the manual review record.
2. Calls `update_violation_license_plate()` to update the violation in `unified_violations.json`.
3. Calls `update_csv_license_plate()` to update the violation in `unified_violations.csv`.
4. Calls `add_violation_to_database()` to add to main violations (currently writes to JSON).
5. Removes from `manual_reviews.json`.

---

#### 3.2.6 Accidents

##### POST /api/accidents

Filter accidents by criteria.

**Request Body (`AccidentFilter`):**
```json
{
  "camera_id": "camera_01",
  "date_from": "2024-01-01",
  "date_to": "2024-12-31"
}
```

##### GET /api/accidents/statistics

Aggregated accident counts by source and date.

##### GET /api/accidents/{accident_id}

Get specific accident details.

---

#### 3.2.7 Media / Evidence

All media endpoints serve `FileResponse` (binary image files).

| Endpoint                                       | Returns                            |
| ---------------------------------------------- | ---------------------------------- |
| `GET /api/media/violations/{id}`               | Full violation snapshot (JPEG)     |
| `GET /api/media/accidents/{id}`                | Accident closeup snapshot (JPEG)   |
| `GET /api/media/plates/{id}`                   | Cropped license plate image (JPEG) |
| `GET /api/media/manual-reviews/{id}/plate`     | Plate snapshot for review          |
| `GET /api/media/manual-reviews/{id}/violation` | Violation snapshot for review      |

**Path Resolution Logic:** The media endpoints normalize path separators and try both `Violation_Proc/violations/{path}` and `Violation_Proc/{path}` as fallback.

---

#### 3.2.8 System

##### GET /api/system/status

**Response (`SystemStatus`):**
```json
{
  "active_sources": 3,
  "active_connections": {
    "violations": 2,
    "accidents": 1,
    "system": 0
  },
  "system_uptime": 3600.5
}
```

---

#### 3.2.9 Frame Capture

##### GET /api/sources/{source_id}/frame

Captures a single frame from a running video source.

**Response (200):**
```json
{
  "source_id": "camera_01",
  "frame_data": "base64_encoded_jpeg_string...",
  "width": 1920,
  "height": 1080
}
```

---

## 4. Data Models & Schemas

### 4.1 Pydantic Models — Current

```python
# --- Request Models ---

class Coordinates(BaseModel):
    lat: float
    lng: float

class VideoSource(BaseModel):
    source_id: Optional[str] = None
    url: str
    name: Optional[str] = None
    camera_location: Optional[str] = "Unknown"
    coordinates: Optional[Coordinates] = None
    speed_limit: Optional[int] = 60
    use_stream: Optional[bool] = False

class ModelSettings(BaseModel):
    accident_detection: Optional[bool] = True
    helmet_detection: Optional[bool] = True
    traffic_violation: Optional[bool] = True
    speed_detection: Optional[bool] = True
    parking_detection: Optional[bool] = True
    wrong_direction: Optional[bool] = True

class AreaDefinition(BaseModel):
    area_type: str
    points: List[List[int]]
    properties: Optional[Dict[str, Any]] = None

class AreaConfiguration(BaseModel):
    areas: Dict[str, List[AreaDefinition]]

class ViolationFilter(BaseModel):
    violation_type: Optional[str] = None
    camera_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    has_license_plate: Optional[bool] = None

class AccidentFilter(BaseModel):
    camera_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

class ManualReviewWithViolationRequest(BaseModel):
    violation_data: Dict[str, Any]
    reason: str
    notes: Optional[str] = None

class ManualReviewUpdate(BaseModel):
    license_plate: Optional[str] = None
    notes: Optional[str] = None
    reviewer_name: Optional[str] = None

# --- Response Models ---

class VideoSourceResponse(BaseModel):
    source_id: str
    url: str
    name: Optional[str] = None
    status: str
    camera_location: Optional[str] = None
    coordinates: Optional[Coordinates] = None

class ManualReviewResponse(BaseModel):
    review_id: str
    status: str
    violation_data: Dict[str, Any]
    reason: str
    notes: Optional[str] = None
    license_plate: Optional[str] = None
    reviewer_name: Optional[str] = None
    created_at: str
    updated_at: str

class SystemStatus(BaseModel):
    active_sources: int
    active_connections: Dict[str, int]
    system_uptime: float
```

### 4.2 Violation Record Schema (JSON)

Every violation stored in `unified_violations.json` has this shape:

```json
{
  "date": "2024-06-15",
  "time": "14:30:00",
  "license_plate": "ABC 1234",
  "camera_id": "camera_01",
  "camera_location": "Cairo, Tahrir Square",
  "location_coordinates": { "lat": 30.0444, "lng": 31.2357 },
  "violation_type": "speed",
  "violation_id": "000001_abc123",
  "vehicle_id": 42,
  "snapshot_path": "snapshots/000001_abc123.jpg",
  "plate_snapshot_path": "plates/000001_abc123_plate.jpg",
  "extra_info": { "speed": 85.3 }
}
```

### 4.3 Accident Record Schema (JSON)

```json
{
  "accident_id": "acc_abc123",
  "date": "2024-06-15",
  "time": "14:30:00",
  "camera_id": "camera_01",
  "camera_location": "Cairo, Tahrir Square",
  "location_coordinates": { "lat": 30.0444, "lng": 31.2357 },
  "accident_type": "car_car_accident",
  "confidence": 0.92,
  "vehicles_involved": [12, 15],
  "snapshot_paths": {
    "closeup": "closeup_acc_abc123.jpg"
  }
}
```

### 4.4 Manual Review Record Schema (JSON)

```json
{
  "review_id": "mr_abc123",
  "status": "pending",
  "violation_data": { ... },
  "reason": "Plate not readable",
  "notes": "Optional notes",
  "license_plate": null,
  "reviewer_name": null,
  "created_at": "2024-06-15T14:30:00",
  "updated_at": "2024-06-15T14:30:00"
}
```

---

## 5. WebSocket Real-Time Layer

### 5.1 Connection Manager Architecture

```python
class ConnectionManager:
    """Manages WebSocket connections organized by channels."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "violations": [],
            "accidents": [],
            "system": []
        }
```

**Channels:**

| Channel      | Purpose                  | Event Types                          |
| ------------ | ------------------------ | ------------------------------------ |
| `violations` | New violation detected   | `new_violation`, `violation_update`  |
| `accidents`  | Accident alerts          | `new_accident`                       |
| `system`     | System status broadcasts | `status_update`, `connected`, `pong` |

### 5.2 WebSocket Endpoints

#### WS /ws/{channel}

Connect to a notification channel. Valid channels: `violations`, `accidents`, `system`.

**Connection Flow:**
```
Client → ws://host:8000/ws/violations
Server → { "event": "connected", "message": "Connected to violations channel", "timestamp": "..." }
Client → "ping"
Server → { "event": "pong", "timestamp": "..." }
...
Server → { "event": "new_violation", "data": { ... violation record ... }, "timestamp": "..." }
```

#### WS /ws/video/{source_id}

Live video frame streaming. Sends base64-encoded JPEG frames at ~5 FPS.

**Frame Message Format:**
```json
{
  "event": "frame",
  "source_id": "camera_01",
  "frame_data": "base64_jpeg_string...",
  "width": 640,
  "height": 360,
  "timestamp": "2024-06-15T14:30:00"
}
```

### 5.3 Notification Handler — Current Implementation

```
Detection Model detects violation
         │
         ▼
ViolationManager._record_violation()       # Synchronous in detection thread
         │
         ▼
APIViolationManager._update_json_log()     # Overridden in api_server.py
         │
         ▼
asyncio.run(                               # ⚠️ PROBLEMATIC — creates new event loop
    notification_handler.notify_new_violation(violation_record)
)
         │
         ▼
manager.broadcast(channel="violations", message=violation_data)
         │
         ▼
WebSocket clients receive notification
```

**⚠️ Problem:** `asyncio.run()` creates a new event loop inside a sync thread. This works BUT:
- Is not idiomatic
- Can conflict with the main FastAPI event loop in edge cases
- Should use `asyncio.run_coroutine_threadsafe(coro, loop)` instead

---

## 6. Persistence Layer — Current vs Target

### 6.1 Current: File-Based JSON

```
Data Flow:
  violation detected → ViolationManager._record_violation()
       │
       ├── Save snapshot (cv2.imwrite) → snapshots/VID.jpg
       ├── LPR recognition (sync) → plates/VID_plate.jpg
       ├── _update_json_log() → unified_violations.json (read-all → append → write-all)
       └── _update_csv_log() → unified_violations.csv (append row)
```

**Problems:**
- **Read-modify-write on every violation:** Reads entire JSON, appends, rewrites. O(n) per write.
- **No indexing:** Finding a violation by ID scans entire array.
- **No transactions:** Concurrent writes can corrupt JSON.
- **No relationships:** Manual reviews reference violations by embedded data, not foreign keys.
- **File locking:** Uses `threading.Lock()` but no file-level locks (multi-process unsafe).

### 6.2 Target: PostgreSQL (Recommended)

See Section 11 for full migration guide.

---

## 7. Configuration System Overhaul

### 7.1 Current State: Dead Code

The file `config.conf` exists with 180 lines of INI configuration covering 18 sections. **None of it is parsed or used by any module.** Every value is hardcoded directly in source code.

```ini
# Example sections from config.conf (ALL UNUSED):
[GENERAL]
project_name = Vision Patrol
debug_mode = false

[API]
host = 0.0.0.0
port = 8000
cors_origins = *
jwt_secret = your-secret-key-here   # ← Placeholder, never used

[DATABASE]
db_type = sqlite                     # ← Never used, system uses JSON files
db_name = vision_patrol.db

[SPEED_DETECTION]
speed_limit_kph = 50                 # ← Code hardcodes speed_limit=1
pixels_per_meter = 30                # ← Code hardcodes the same value

[NOTIFICATION]
enable_email = false                 # ← No notification system exists
enable_sms = false
enable_webhook = false
```

### 7.2 Implementation: Parse and Use config.conf

**Step 1: Create `config_manager.py`**

```python
"""
config_manager.py — Centralized configuration manager.
Parses config.conf and exposes typed settings throughout the application.
"""
import os
import configparser
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.conf"


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60


@dataclass
class DatabaseConfig:
    db_type: str = "sqlite"           # sqlite | postgresql | mysql
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "vision_patrol"
    db_user: str = "vision_patrol"
    db_password: str = ""
    
    @property
    def connection_url(self) -> str:
        if self.db_type == "sqlite":
            return f"sqlite:///./{self.db_name}.db"
        elif self.db_type == "postgresql":
            return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        elif self.db_type == "mysql":
            return f"mysql+aiomysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        return f"sqlite:///./{self.db_name}.db"


@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.5
    yolo_model_path: str = "yolov8s.pt"
    device: str = "auto"              # auto | cpu | cuda | mps


@dataclass
class AccidentConfig:
    model_path: str = "Models/Accident_det/accident.pt"
    confidence_threshold: float = 0.86
    cooldown_seconds: int = 35
    frame_skip: int = 3
    auto_disable_count: int = 3
    auto_disable_window: int = 80
    auto_disable_duration: int = 600


@dataclass
class SpeedConfig:
    pixels_per_meter: int = 30
    speed_limit_kph: float = 50.0
    smoothing_window: int = 10
    max_speed_change: float = 5.0


@dataclass
class NotificationConfig:
    enable_email: bool = False
    email_host: str = ""
    email_port: int = 587
    email_user: str = ""
    email_password: str = ""
    enable_sms: bool = False
    enable_webhook: bool = False
    webhook_url: str = ""


@dataclass
class AppConfig:
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    accident: AccidentConfig = field(default_factory=AccidentConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    debug_mode: bool = False
    max_workers: int = 4


def load_config(config_path: str = None) -> AppConfig:
    """Load configuration from INI file, falling back to defaults."""
    path = Path(config_path) if config_path else CONFIG_PATH
    config = AppConfig()
    
    if not path.exists():
        print(f"WARNING: Config file not found at {path}, using defaults")
        return config
    
    parser = configparser.ConfigParser()
    parser.read(str(path))
    
    def get(section: str, key: str, fallback=None):
        return parser.get(section, key, fallback=fallback)
    
    def getint(section: str, key: str, fallback=0):
        return parser.getint(section, key, fallback=fallback)
    
    def getfloat(section: str, key: str, fallback=0.0):
        return parser.getfloat(section, key, fallback=fallback)
    
    def getbool(section: str, key: str, fallback=False):
        return parser.getboolean(section, key, fallback=fallback)
    
    # Parse each section
    if parser.has_section("GENERAL"):
        config.debug_mode = getbool("GENERAL", "debug_mode", False)
    
    if parser.has_section("API"):
        config.api.host = get("API", "host", "0.0.0.0")
        config.api.port = getint("API", "port", 8000)
        cors = get("API", "cors_origins", "*")
        config.api.cors_origins = [o.strip() for o in cors.split(",")]
        config.api.jwt_secret = get("API", "jwt_secret", "change-me-in-production")
    
    if parser.has_section("DATABASE"):
        config.database.db_type = get("DATABASE", "db_type", "sqlite")
        config.database.db_host = get("DATABASE", "db_host", "localhost")
        config.database.db_port = getint("DATABASE", "db_port", 5432)
        config.database.db_name = get("DATABASE", "db_name", "vision_patrol")
        config.database.db_user = get("DATABASE", "db_user", "vision_patrol")
        config.database.db_password = get("DATABASE", "db_password", "")
    
    if parser.has_section("SPEED_DETECTION"):
        config.speed.speed_limit_kph = getfloat("SPEED_DETECTION", "speed_limit_kph", 50.0)
        config.speed.pixels_per_meter = getint("SPEED_DETECTION", "pixels_per_meter", 30)
    
    if parser.has_section("ACCIDENT_DETECTION"):
        config.accident.confidence_threshold = getfloat("ACCIDENT_DETECTION", "confidence_threshold", 0.86)
        config.accident.cooldown_seconds = getint("ACCIDENT_DETECTION", "cooldown_seconds", 35)
    
    if parser.has_section("NOTIFICATION"):
        config.notification.enable_email = getbool("NOTIFICATION", "enable_email", False)
        config.notification.enable_webhook = getbool("NOTIFICATION", "enable_webhook", False)
        config.notification.webhook_url = get("NOTIFICATION", "webhook_url", "")
    
    if parser.has_section("MULTIPROCESSING"):
        config.max_workers = getint("MULTIPROCESSING", "max_workers", 4)
    
    return config


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
```

**Step 2: Wire into `api_server.py`**

```python
from config_manager import get_config

config = get_config()

app = FastAPI(title="Vision Patrol API", debug=config.debug_mode)

# Use config for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,  # Instead of ["*"]
    ...
)

# Use config for uvicorn
if __name__ == "__main__":
    uvicorn.run("api_server:app", host=config.api.host, port=config.api.port)
```

**Step 3: Wire into detection models**

```python
from config_manager import get_config

config = get_config()

# In SpeedDetector.__init__():
self.speed_limit = config.speed.speed_limit_kph  # Instead of hardcoded 1.0
self.pixels_per_meter = config.speed.pixels_per_meter

# In AccidentDetector.__init__():
self.conf_threshold = config.accident.confidence_threshold
self.cooldown = config.accident.cooldown_seconds
```

---

## 8. Authentication & Authorization Implementation

### 8.1 Architecture Design

```
┌──────────────────────────────────────────────────────────┐
│                  Authentication Flow                      │
│                                                           │
│  Client                        Server                     │
│  ──────                        ──────                     │
│  POST /api/auth/login          Validate credentials       │
│  { "username", "password" }    Generate JWT access token   │
│   ◄──── 200 { "access_token", "token_type": "bearer" }   │
│                                                           │
│  GET /api/violations           Decode & verify JWT        │
│  Authorization: Bearer <token> Check role permissions     │
│   ◄──── 200 [violations...]    Return filtered data       │
│                                                           │
│  WS /ws/violations?token=xxx   Verify token on connect   │
│   ◄──── Accept / Reject        Channel-level auth         │
└──────────────────────────────────────────────────────────┘
```

### 8.2 Implementation Code

```python
# auth.py — JWT Authentication Module

from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from config_manager import get_config

config = get_config()

SECRET_KEY = config.api.jwt_secret
ALGORITHM = config.api.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = config.api.jwt_expiration_minutes

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class User(BaseModel):
    username: str
    role: str  # "admin", "operator", "viewer"
    disabled: bool = False


# -- For production, replace with database lookup --
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
        "disabled": False,
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "viewer")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    
    user = fake_users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    return User(**{k: v for k, v in user.items() if k != "hashed_password"})


def require_role(*roles: str):
    """Dependency factory for role-based access control."""
    async def role_checker(user: User = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' not authorized. Required: {roles}"
            )
        return user
    return role_checker


async def authenticate_websocket(websocket: WebSocket) -> Optional[User]:
    """Authenticate WebSocket connections via query parameter token."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = fake_users_db.get(username)
        if user:
            return User(**{k: v for k, v in user.items() if k != "hashed_password"})
    except JWTError:
        pass
    await websocket.close(code=1008, reason="Invalid authentication token")
    return None
```

### 8.3 RBAC Role Matrix

| Endpoint Category     | `admin` | `operator` | `viewer`      |
| --------------------- | ------- | ---------- | ------------- |
| Source management     | ✅       | ✅          | ❌             |
| Start/stop processing | ✅       | ✅          | ❌             |
| Area configuration    | ✅       | ✅          | ❌             |
| Model settings        | ✅       | ✅          | ❌             |
| View violations       | ✅       | ✅          | ✅             |
| View accidents        | ✅       | ✅          | ✅             |
| Manual review CRUD    | ✅       | ✅          | ❌             |
| Media/snapshots       | ✅       | ✅          | ✅             |
| System status         | ✅       | ✅          | ✅             |
| User management       | ✅       | ❌          | ❌             |
| WebSocket channels    | ✅       | ✅          | ✅ (read-only) |

### 8.4 Applying Auth to Existing Endpoints

```python
from auth import get_current_user, require_role

# Read-only endpoints — any authenticated user
@app.get("/api/violations")
async def get_all_violations(user: User = Depends(get_current_user)):
    ...

# Operator+ endpoints
@app.post("/api/sources")
async def add_source(source: VideoSource, user: User = Depends(require_role("admin", "operator"))):
    ...

# Admin-only endpoints
@app.delete("/api/sources/{source_id}")
async def remove_source(source_id: str, user: User = Depends(require_role("admin"))):
    ...

# WebSocket with auth
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    user = await authenticate_websocket(websocket)
    if not user:
        return  # Connection already closed
    await manager.connect(websocket, channel)
    ...
```

### 8.5 Required Dependencies

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

---

## 9. Target Production Architecture

### 9.1 Recommended Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION ARCHITECTURE (Target)                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌──────────────────────────────────────────────────────┐                            │
│  │                    NGINX Reverse Proxy                │                            │
│  │   ┌───────────┐  ┌────────────┐  ┌────────────────┐  │                            │
│  │   │ TLS/SSL   │  │ Rate Limit │  │ Static Files   │  │                            │
│  │   │ Termination│  │ (per IP)   │  │ (HTML/CSS/JS)  │  │                            │
│  │   └───────────┘  └────────────┘  └────────────────┘  │                            │
│  └──────────────────────┬───────────────────────────────┘                            │
│                         │                                                            │
│         ┌───────────────┼───────────────┐                                            │
│         │               │               │                                            │
│         ▼               ▼               ▼                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                                  │
│  │  API Service  │ │ API Service  │ │ API Service  │   ← Horizontal Scaling           │
│  │  (Uvicorn)    │ │  (Uvicorn)   │ │  (Uvicorn)   │      via Docker/K8s             │
│  │  Port 8001    │ │  Port 8002   │ │  Port 8003   │                                  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                                  │
│         │                │                │                                            │
│         └────────────────┼────────────────┘                                            │
│                          │                                                             │
│         ┌────────────────┼────────────────┐                                            │
│         │                │                │                                            │
│         ▼                ▼                ▼                                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐                                │
│  │  PostgreSQL   │ │    Redis     │ │  Object Storage  │                                │
│  │              │ │              │ │  (MinIO / S3)    │                                │
│  │  Violations   │ │  WS PubSub  │ │  Snapshots       │                                │
│  │  Accidents    │ │  Cache       │ │  Plates          │                                │
│  │  Reviews      │ │  Sessions   │ │  Accident imgs   │                                │
│  │  Users        │ │              │ │                  │                                │
│  └──────────────┘ └──────────────┘ └──────────────────┘                                │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                       PROCESSING TIER (Separate Containers)                      │  │
│  │                                                                                  │  │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐          │  │
│  │  │ Video Processor    │  │ Video Processor    │  │ Video Processor    │          │  │
│  │  │ Camera Group A     │  │ Camera Group B     │  │ Camera Group C     │          │  │
│  │  │                    │  │                    │  │                    │          │  │
│  │  │ • VideoReader      │  │ • VideoReader      │  │ • VideoReader      │          │  │
│  │  │ • YOLO Detection   │  │ • YOLO Detection   │  │ • YOLO Detection   │          │  │
│  │  │ • 6 Sub-models     │  │ • 6 Sub-models     │  │ • 6 Sub-models     │          │  │
│  │  │ • LPR Pipeline     │  │ • LPR Pipeline     │  │ • LPR Pipeline     │          │  │
│  │  └────────┬───────────┘  └────────┬───────────┘  └────────┬───────────┘          │  │
│  │           │                       │                       │                       │  │
│  │           └───────────┬───────────┘───────────────────────┘                       │  │
│  │                       │                                                           │  │
│  │                       ▼                                                           │  │
│  │            ┌──────────────────────┐                                               │  │
│  │            │   Message Queue      │                                               │  │
│  │            │   (Redis Streams /   │                                               │  │
│  │            │    RabbitMQ)         │                                               │  │
│  │            │                      │                                               │  │
│  │            │  Channels:           │                                               │  │
│  │            │  • violation_events  │                                               │  │
│  │            │  • accident_events   │                                               │  │
│  │            │  • frame_events      │                                               │  │
│  │            └──────────────────────┘                                               │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                       ASYNC WORKERS (Background Services)                        │  │
│  │                                                                                  │  │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐          │  │
│  │  │ LPR Worker         │  │ Notification Worker│  │ Cleanup Worker     │          │  │
│  │  │ (async plate recog)│  │ (email/SMS/webhook)│  │ (old data pruning) │          │  │
│  │  └────────────────────┘  └────────────────────┘  └────────────────────┘          │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Key Architecture Decisions

| Decision                          | Rationale                                                         |
| --------------------------------- | ----------------------------------------------------------------- |
| Separate API from Processing      | API can scale independently; GPU workers need different resources |
| PostgreSQL over MongoDB           | Relational violations data; strong filtering; ACID transactions   |
| Redis for WebSocket PubSub        | Multi-instance WS broadcast; sub-millisecond latency              |
| Message Queue for detection → API | Decouples processing from API; enables async LPR                  |
| Object Storage for media          | Scales independently; CDN-compatible; backup-friendly             |
| NGINX as reverse proxy            | TLS termination; rate limiting; static file serving               |
| JWT stateless auth                | Simple, scalable; no session storage needed                       |

---

## 10. Service Decomposition Plan

### 10.1 Service Boundary Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     SERVICE BOUNDARIES                            │
│                                                                   │
│  ┌─────────────────────────┐    ┌──────────────────────────────┐ │
│  │   API Gateway Service    │    │   Video Processing Service   │ │
│  │   (api_server.py)        │    │   (VideoProcessorMP.py)      │ │
│  │                          │    │                               │ │
│  │  Responsibilities:       │    │  Responsibilities:            │ │
│  │  • REST endpoints        │    │  • Video stream ingestion     │ │
│  │  • WebSocket channels    │    │  • YOLO object detection      │ │
│  │  • Authentication        │    │  • Detection model execution  │ │
│  │  • Request validation    │    │  • Evidence capture           │ │
│  │  • Response formatting   │    │  • Frame extraction           │ │
│  │  • Static file serving   │    │                               │ │
│  │                          │    │  Communicates via:             │ │
│  │  Owns:                   │    │  • Message Queue (events)     │ │
│  │  • violations table      │    │  • Object Storage (images)    │ │
│  │  • accidents table       │    │  • Shared PostgreSQL (writes) │ │
│  │  • manual_reviews table  │    │                               │ │
│  │  • users table           │    │  Owns:                        │ │
│  └─────────────────────────┘    │  • area_configs               │ │
│                                  │  • model settings             │ │
│  ┌─────────────────────────┐    └──────────────────────────────┘ │
│  │   LPR Service            │                                     │
│  │   (License Recognition)   │    ┌──────────────────────────────┐ │
│  │                          │    │   Notification Service        │ │
│  │  Responsibilities:       │    │                               │ │
│  │  • Plate detection       │    │  Responsibilities:            │ │
│  │  • OCR processing        │    │  • Email alerts               │ │
│  │  • Plate image storage   │    │  • SMS notifications          │ │
│  │                          │    │  • Webhook delivery            │ │
│  │  Communicates via:       │    │  • Push notifications          │ │
│  │  • Message Queue (async) │    │                               │ │
│  │  • PostgreSQL (updates)  │    │  Communicates via:             │ │
│  └─────────────────────────┘    │  • Message Queue (subscribes)  │ │
│                                  └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 Migration Phases

**Phase 1 — Fondation (Current Sprint)**
- Parse `config.conf` via `ConfigManager`
- Add JWT authentication
- Add request pagination
- Fix duplicate endpoints
- Fix `asyncio.run()` → `run_coroutine_threadsafe()`

**Phase 2 — Database Migration**
- Add SQLAlchemy models
- Migrate JSON → PostgreSQL
- Add Alembic for migrations
- Keep JSON as fallback during transition

**Phase 3 — Service Extraction**
- Extract LPR into async worker
- Add Redis PubSub for WebSocket
- Add message queue for detection events
- Split processing into separate container

**Phase 4 — Production Hardening**
- Add NGINX reverse proxy
- Docker Compose for all services
- Structured logging (JSON)
- Monitoring & health checks
- Rate limiting

---

## 11. Database Migration Guide

### 11.1 SQLAlchemy Models

```python
# models/database.py — SQLAlchemy ORM Models

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class ViolationType(enum.Enum):
    SPEED = "speed"
    HELMET = "helmet"
    TRAFFIC_LIGHT = "traffic_light"
    PARKING = "parking"
    WRONG_DIRECTION = "wrong_direction"


class ReviewStatus(enum.Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class UserRole(enum.Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class VideoSource(Base):
    __tablename__ = "video_sources"
    
    id = Column(String(36), primary_key=True)  # UUID
    url = Column(String(500), nullable=False)
    name = Column(String(200))
    camera_location = Column(String(300), default="Unknown")
    latitude = Column(Float, default=0.0)
    longitude = Column(Float, default=0.0)
    speed_limit = Column(Integer, default=60)
    use_stream = Column(Boolean, default=False)
    status = Column(String(20), default="stopped")
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    violations = relationship("Violation", back_populates="source")
    accidents = relationship("Accident", back_populates="source")
    area_configs = relationship("AreaConfig", back_populates="source")


class Violation(Base):
    __tablename__ = "violations"
    
    id = Column(String(50), primary_key=True)  # violation_id
    violation_type = Column(Enum(ViolationType), nullable=False, index=True)
    camera_id = Column(String(36), ForeignKey("video_sources.id"), nullable=False, index=True)
    vehicle_id = Column(Integer)
    license_plate = Column(String(20), index=True)
    snapshot_path = Column(String(500))
    plate_snapshot_path = Column(String(500))
    extra_info = Column(JSON, default={})
    date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    source = relationship("VideoSource", back_populates="violations")
    manual_review = relationship("ManualReview", back_populates="violation", uselist=False)


class Accident(Base):
    __tablename__ = "accidents"
    
    id = Column(String(50), primary_key=True)  # accident_id
    accident_type = Column(String(50))
    camera_id = Column(String(36), ForeignKey("video_sources.id"), nullable=False, index=True)
    confidence = Column(Float)
    vehicles_involved = Column(JSON, default=[])
    snapshot_paths = Column(JSON, default={})
    date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    source = relationship("VideoSource", back_populates="accidents")


class ManualReview(Base):
    __tablename__ = "manual_reviews"
    
    id = Column(String(50), primary_key=True)  # review_id
    violation_id = Column(String(50), ForeignKey("violations.id"), nullable=False)
    status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING, index=True)
    reason = Column(String(500))
    notes = Column(Text)
    license_plate = Column(String(20))
    reviewer_name = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    violation = relationship("Violation", back_populates="manual_review")


class AreaConfig(Base):
    __tablename__ = "area_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(36), ForeignKey("video_sources.id"), nullable=False)
    area_type = Column(String(50), nullable=False)
    points = Column(JSON, nullable=False)
    properties = Column(JSON, default={})
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    source = relationship("VideoSource", back_populates="area_configs")
```

### 11.2 Database Connection Setup

```python
# models/db_session.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config_manager import get_config
from models.database import Base

config = get_config()

engine = create_async_engine(
    config.database.connection_url,
    echo=config.debug_mode,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI dependency for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

### 11.3 Alembic Migration Setup

```bash
# Initialize Alembic
pip install alembic
alembic init alembic

# In alembic/env.py, set:
#   target_metadata = Base.metadata
#   sqlalchemy.url from config_manager

# Generate initial migration
alembic revision --autogenerate -m "initial_schema"

# Apply migration
alembic upgrade head
```

### 11.4 Repository Pattern Implementation

```python
# repositories/violation_repo.py

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.database import Violation, ViolationType
from typing import Optional, List
from datetime import datetime


class ViolationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, violation_data: dict) -> Violation:
        violation = Violation(**violation_data)
        self.session.add(violation)
        await self.session.flush()
        return violation
    
    async def get_by_id(self, violation_id: str) -> Optional[Violation]:
        result = await self.session.execute(
            select(Violation).where(Violation.id == violation_id)
        )
        return result.scalar_one_or_none()
    
    async def list_filtered(
        self,
        violation_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        has_plate: Optional[bool] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[List[Violation], int]:
        query = select(Violation)
        count_query = select(func.count(Violation.id))
        
        if violation_type:
            query = query.where(Violation.violation_type == ViolationType(violation_type))
            count_query = count_query.where(Violation.violation_type == ViolationType(violation_type))
        if camera_id:
            query = query.where(Violation.camera_id == camera_id)
            count_query = count_query.where(Violation.camera_id == camera_id)
        if date_from:
            query = query.where(Violation.date >= date_from)
            count_query = count_query.where(Violation.date >= date_from)
        if date_to:
            query = query.where(Violation.date <= date_to)
            count_query = count_query.where(Violation.date <= date_to)
        if has_plate is True:
            query = query.where(Violation.license_plate.isnot(None))
            count_query = count_query.where(Violation.license_plate.isnot(None))
        elif has_plate is False:
            query = query.where(Violation.license_plate.is_(None))
            count_query = count_query.where(Violation.license_plate.is_(None))
        
        # Pagination
        total = (await self.session.execute(count_query)).scalar()
        query = query.order_by(Violation.date.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await self.session.execute(query)
        return result.scalars().all(), total
    
    async def get_statistics(self) -> dict:
        # Count by type
        type_counts = await self.session.execute(
            select(Violation.violation_type, func.count(Violation.id))
            .group_by(Violation.violation_type)
        )
        
        return {
            "total": sum(count for _, count in type_counts.all()),
            "by_type": {str(vtype.value): count for vtype, count in type_counts.all()},
        }
```

### 11.5 Data Migration Script

```python
# scripts/migrate_json_to_db.py
"""One-time migration of JSON flat files → PostgreSQL."""

import json
import asyncio
from datetime import datetime
from models.db_session import AsyncSessionLocal, init_db
from models.database import Violation, Accident, ManualReview, ViolationType

VIOLATIONS_JSON = "Violation_Proc/violations/unified_violations.json"
ACCIDENTS_JSON = "Violation_Proc/accident_alerts/all_accident_alerts.json"
REVIEWS_JSON = "Violation_Proc/violations/manual_reviews.json"


async def migrate():
    await init_db()
    
    async with AsyncSessionLocal() as session:
        # Migrate violations
        with open(VIOLATIONS_JSON) as f:
            violations = json.load(f)
        
        for v in violations:
            violation = Violation(
                id=v["violation_id"],
                violation_type=ViolationType(v["violation_type"]),
                camera_id=v["camera_id"],
                vehicle_id=v.get("vehicle_id"),
                license_plate=v.get("license_plate") or None,
                snapshot_path=v.get("snapshot_path"),
                plate_snapshot_path=v.get("plate_snapshot_path"),
                extra_info=v.get("extra_info", {}),
                date=datetime.strptime(f"{v['date']} {v['time']}", "%Y-%m-%d %H:%M:%S"),
            )
            session.add(violation)
        
        # Migrate accidents
        with open(ACCIDENTS_JSON) as f:
            accidents = json.load(f)
        
        for a in accidents:
            accident = Accident(
                id=a["accident_id"],
                accident_type=a.get("accident_type"),
                camera_id=a["camera_id"],
                confidence=a.get("confidence"),
                vehicles_involved=a.get("vehicles_involved", []),
                snapshot_paths=a.get("snapshot_paths", {}),
                date=datetime.strptime(f"{a['date']} {a['time']}", "%Y-%m-%d %H:%M:%S"),
            )
            session.add(accident)
        
        await session.commit()
        print(f"Migrated {len(violations)} violations and {len(accidents)} accidents.")


if __name__ == "__main__":
    asyncio.run(migrate())
```

---

## 12. Notification & Alerting Service

### 12.1 Architecture

```
Detection Event (violation/accident)
         │
         ▼
┌─────────────────┐
│  Message Queue   │     Violation/Accident event published
│  (Redis Stream)  │     as JSON message
└────────┬────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│  WebSocket       │                   │  Notification    │
│  Broadcaster     │                   │  Worker          │
│                  │                   │                  │
│  Reads from      │                   │  Reads from      │
│  Redis PubSub    │                   │  Redis Stream    │
│  Broadcasts to   │                   │  Sends:          │
│  WS clients      │                   │  • Email (SMTP)  │
└─────────────────┘                   │  • SMS (Twilio)  │
                                       │  • Webhook (HTTP)│
                                       └─────────────────┘
```

### 12.2 Implementation

```python
# services/notification_service.py

import httpx
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config_manager import get_config

config = get_config()


class NotificationService:
    """Sends notifications for high-priority events."""
    
    async def send_email(self, subject: str, body: str, recipients: list[str]):
        if not config.notification.enable_email:
            return
        
        msg = MIMEMultipart()
        msg["From"] = config.notification.email_user
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        
        with smtplib.SMTP(config.notification.email_host, config.notification.email_port) as server:
            server.starttls()
            server.login(config.notification.email_user, config.notification.email_password)
            server.send_message(msg)
    
    async def send_webhook(self, event_type: str, payload: dict):
        if not config.notification.enable_webhook:
            return
        
        async with httpx.AsyncClient() as client:
            await client.post(
                config.notification.webhook_url,
                json={"event": event_type, "data": payload},
                timeout=10.0,
            )
    
    async def notify_accident(self, accident_data: dict):
        """Send high-priority notifications for accidents."""
        subject = f"⚠️ ACCIDENT DETECTED — Camera {accident_data.get('camera_id')}"
        body = f"""
        <h2>Accident Alert</h2>
        <p><strong>Camera:</strong> {accident_data.get('camera_id')}</p>
        <p><strong>Location:</strong> {accident_data.get('camera_location')}</p>
        <p><strong>Type:</strong> {accident_data.get('accident_type')}</p>
        <p><strong>Time:</strong> {accident_data.get('date')} {accident_data.get('time')}</p>
        <p><strong>Confidence:</strong> {accident_data.get('confidence', 0):.1%}</p>
        """
        
        await self.send_email(subject, body, ["admin@example.com"])
        await self.send_webhook("accident", accident_data)
```

---

## 13. Containerization & Deployment

### 13.1 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    python-jose[cryptography] \
    passlib[bcrypt] \
    sqlalchemy[asyncio] \
    asyncpg \
    alembic \
    redis \
    httpx

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p Violation_Proc/violations/snapshots \
    Violation_Proc/violations/plates \
    Violation_Proc/accident_alerts \
    area_configs \
    static

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/system/status || exit 1

# Run with uvicorn (production settings)
CMD ["uvicorn", "api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "65"]
```

### 13.2 Docker Compose — Full Stack

```yaml
# docker-compose.yml
version: "3.9"

services:
  # --- API + Processing Server ---
  api:
    build: .
    container_name: vision-patrol-api
    ports:
      - "8000:8000"
    volumes:
      - ./config.conf:/app/config.conf:ro
      - violation_data:/app/Violation_Proc
      - area_configs:/app/area_configs
      - ./Models:/app/Models:ro          # Model weights (read-only)
    environment:
      - DATABASE_URL=postgresql+asyncpg://vp_user:vp_pass@postgres:5432/vision_patrol
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # --- PostgreSQL Database ---
  postgres:
    image: postgres:16-alpine
    container_name: vision-patrol-db
    environment:
      POSTGRES_DB: vision_patrol
      POSTGRES_USER: vp_user
      POSTGRES_PASSWORD: vp_pass
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U vp_user -d vision_patrol"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # --- Redis (Cache + PubSub) ---
  redis:
    image: redis:7-alpine
    container_name: vision-patrol-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # --- NGINX Reverse Proxy ---
  nginx:
    image: nginx:alpine
    container_name: vision-patrol-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./static:/usr/share/nginx/html:ro
    depends_on:
      - api
    restart: unless-stopped

volumes:
  pgdata:
  violation_data:
  area_configs:
```

### 13.3 NGINX Configuration

```nginx
# nginx.conf
upstream api_backend {
    server api:8000;
}

server {
    listen 80;
    server_name _;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/s;

    # Static files (served directly by NGINX)
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api/ {
        limit_req zone=api_limit burst=50 nodelay;
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket proxy
    location /ws/ {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }

    # Media files (violation snapshots)
    location /api/media/ {
        proxy_pass http://api_backend;
        proxy_cache_valid 200 1h;
        add_header Cache-Control "public, max-age=3600";
    }
}
```

---

## 14. Error Handling & Logging Strategy

### 14.1 Structured Logging Implementation

```python
# logging_config.py

import logging
import json
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry)


def setup_logging(debug: bool = False):
    """Configure structured logging for the application."""
    level = logging.DEBUG if debug else logging.INFO
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    
    # Suppress noisy libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return root_logger
```

### 14.2 Global Error Handler

```python
# In api_server.py:

from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": str(uuid.uuid4())},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
```

### 14.3 Pagination Pattern

```python
# schemas/pagination.py

from pydantic import BaseModel, Field
from typing import Generic, TypeVar, List

T = TypeVar("T")


class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    @classmethod
    def create(cls, items: List[T], total: int, page: int, page_size: int):
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=(total + page_size - 1) // page_size,
        )
```

**Usage:**
```python
@app.post("/api/violations", response_model=PaginatedResponse[dict])
async def get_filtered_violations(
    filter_params: ViolationFilter = Body(default=None),
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_db),
):
    repo = ViolationRepository(db)
    items, total = await repo.list_filtered(
        violation_type=filter_params.violation_type if filter_params else None,
        page=pagination.page,
        page_size=pagination.page_size,
    )
    return PaginatedResponse.create(items, total, pagination.page, pagination.page_size)
```

---

## 15. Testing Strategy

### 15.1 Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_api/
│   ├── test_sources.py      # Source CRUD endpoints
│   ├── test_violations.py   # Violation endpoints
│   ├── test_accidents.py    # Accident endpoints
│   ├── test_reviews.py      # Manual review endpoints
│   ├── test_media.py        # Media/snapshot endpoints
│   └── test_auth.py         # Authentication tests
├── test_models/
│   ├── test_speed.py        # Speed detection logic
│   ├── test_parking.py      # Parking detection logic
│   └── test_wrong_dir.py    # Direction detection logic
├── test_services/
│   ├── test_violation_mgr.py
│   └── test_notification.py
└── test_integration/
    └── test_full_pipeline.py
```

### 15.2 Fixtures

```python
# tests/conftest.py

import pytest
from httpx import AsyncClient, ASGITransport
from api_server import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_source():
    return {
        "url": "test_video.mp4",
        "name": "Test Camera",
        "camera_location": "Test Location",
        "coordinates": {"lat": 30.0, "lng": 31.0},
        "speed_limit": 60,
    }


@pytest.fixture
def sample_violation():
    return {
        "date": "2024-06-15",
        "time": "14:30:00",
        "license_plate": "ABC 1234",
        "camera_id": "test_cam",
        "camera_location": "Test",
        "violation_type": "speed",
        "violation_id": "test_001",
        "vehicle_id": 1,
        "snapshot_path": "",
        "extra_info": {"speed": 85.0},
    }
```

### 15.3 Example Tests

```python
# tests/test_api/test_sources.py

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_add_source(client: AsyncClient, sample_source):
    response = await client.post("/api/sources", json=sample_source)
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == sample_source["url"]
    assert "source_id" in data


@pytest.mark.asyncio
async def test_list_sources(client: AsyncClient):
    response = await client.get("/api/sources")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_duplicate_source_id(client: AsyncClient, sample_source):
    sample_source["source_id"] = "duplicate_test"
    await client.post("/api/sources", json=sample_source)
    response = await client.post("/api/sources", json=sample_source)
    assert response.status_code == 400
```

---

## 16. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task                                   | Priority | Effort  | Dependencies   |
| -------------------------------------- | -------- | ------- | -------------- |
| Create `config_manager.py`             | P0       | 4 hrs   | None           |
| Wire config into `api_server.py`       | P0       | 2 hrs   | config_manager |
| Wire config into detection models      | P0       | 3 hrs   | config_manager |
| Remove duplicate settings endpoints    | P0       | 0.5 hrs | None           |
| Fix `asyncio.run()` → threadsafe calls | P0       | 2 hrs   | None           |
| Add JWT auth module `auth.py`          | P0       | 4 hrs   | config_manager |
| Apply auth to all endpoints            | P0       | 3 hrs   | auth.py        |
| Add pagination to list endpoints       | P1       | 3 hrs   | None           |
| Restrict CORS origins from config      | P1       | 0.5 hrs | config_manager |

### Phase 2: Database (Week 3-4)

| Task                            | Priority | Effort | Dependencies |
| ------------------------------- | -------- | ------ | ------------ |
| Define SQLAlchemy models        | P0       | 4 hrs  | None         |
| Setup Alembic migrations        | P0       | 2 hrs  | models       |
| Create repository layer         | P0       | 6 hrs  | models       |
| Write data migration script     | P0       | 3 hrs  | repos        |
| Refactor endpoints to use repos | P0       | 6 hrs  | repos        |
| Validate with integration tests | P1       | 4 hrs  | repos        |

### Phase 3: Production Readiness (Week 5-6)

| Task                           | Priority | Effort | Dependencies |
| ------------------------------ | -------- | ------ | ------------ |
| Add structured JSON logging    | P1       | 3 hrs  | None         |
| Create Dockerfile              | P1       | 2 hrs  | None         |
| Create docker-compose.yml      | P1       | 3 hrs  | Dockerfile   |
| Add NGINX reverse proxy config | P1       | 2 hrs  | Docker       |
| Add global error handlers      | P1       | 2 hrs  | None         |
| Add health check endpoint      | P2       | 1 hr   | None         |
| Setup Redis for WS PubSub      | P2       | 4 hrs  | Docker       |

### Phase 4: Service Extraction (Week 7-8)

| Task                                     | Priority | Effort | Dependencies |
| ---------------------------------------- | -------- | ------ | ------------ |
| Extract LPR into async queue worker      | P1       | 6 hrs  | Redis/MQ     |
| Add notification service                 | P2       | 4 hrs  | Redis/MQ     |
| Split processing into separate container | P2       | 6 hrs  | Docker       |
| Add monitoring/metrics endpoint          | P2       | 3 hrs  | None         |
| Write comprehensive test suite           | P1       | 8 hrs  | All above    |

---

## Appendix A: Quick Reference — File Map

| File                                                             | Lines | Purpose                                              |
| ---------------------------------------------------------------- | ----- | ---------------------------------------------------- |
| `api_server.py`                                                  | 1640  | FastAPI REST + WebSocket server                      |
| `config.conf`                                                    | 180   | Configuration file (UNUSED — needs parsing)          |
| `Main_mp.py`                                                     | ~200  | Multiprocessing entry point (dev mode)               |
| `Processing Models/VideoProcessorMP.py`                          | 676   | Core video processing loop + detection orchestration |
| `Processing Models/VideoReader.py`                               | ~150  | Video source abstraction (file/stream/YT)            |
| `Processing Models/areas.py`                                     | ~400  | Detection zone management (AreaManager)              |
| `Violation_Proc/violation_manager.py`                            | 1051  | Centralized violation recording + LPR                |
| `Violation_Proc/accident_alert_manager.py`                       | ~400  | Accident event management                            |
| `Violation_Proc/License_recognition/license_plate_detector.py`   | 237   | YOLO plate detection + EasyOCR                       |
| `Violation_Proc/License_recognition/license_plate_recognizer.py` | ~200  | LPR orchestration                                    |
| `Models/Accident_det/Accident_Detector.py`                       | 416   | YOLOv8 accident detection (7 classes)                |
| `Models/Bike_Violations/helmet_detector.py`                      | 395   | YOLOv8 helmet detection (2 classes)                  |
| `Models/Speed Model/speed_detector.py`                           | 396   | Pixel-based speed estimation                         |
| `Models/Traffic_violation/traffic_violation_detector.py`         | 208   | Red-light violation detection                        |
| `Models/Traffic_violation/masking.py`                            | ~100  | HSV traffic light color detection                    |
| `Models/Parking Model/parking_Detctor.py`                        | ~150  | Timer-based parking violation                        |
| `Models/Wrong_dir/wrong_det.py`                                  | ~300  | Direction detection via line-pair gates              |

---

## Appendix B: Environment Variables (Recommended)

For production deployments, override `config.conf` values with environment variables:

```bash
# .env
VP_API_HOST=0.0.0.0
VP_API_PORT=8000
VP_JWT_SECRET=your-production-secret-key-minimum-32-chars
VP_DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/vision_patrol
VP_REDIS_URL=redis://redis:6379/0
VP_CORS_ORIGINS=https://dashboard.example.com,https://admin.example.com
VP_DEBUG=false
VP_MAX_WORKERS=4
```

---

*Report generated for the Vision Patrol graduation project. This document provides a complete backend developer guide with implementation-ready code, architecture diagrams, and a phased migration roadmap.*
