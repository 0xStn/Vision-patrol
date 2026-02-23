# Vision Patrol — Violation Processing Report

**Date:** February 23, 2026  
**Project:** Vision Patrol — AI-Powered Traffic Monitoring System  
**Report Scope:** Violation lifecycle, evidence pipeline, license plate recognition, data persistence, and real-time notification  

---

## 1. Overview

The violation processing subsystem is the **evidence backbone** of Vision Patrol. It receives raw detection events from the 6 detector modules and transforms them into auditable violation records with photographic evidence, license plate identification, and real-time notifications.

### Violation Pipeline Summary

```
Detector triggers violation
  → ViolationManager.record_{type}_violation()
    → Save vehicle snapshot (45% margin crop)
      → Run license plate recognition (YOLO + EasyOCR)
        → Confidence ≥ 0.3? → unified_violations.json ✓
        → Confidence < 0.3? → manual_reviews.json (pending) ⏳
          → WebSocket broadcast to connected clients
```

---

## 2. ViolationManager — Central Authority

**File:** `Violation_Proc/violation_manager.py` (1,051 lines)  
**Class:** `ViolationManager`

### Responsibilities

The `ViolationManager` is the **single source of truth** for all non-accident violations. It handles:

1. Recording violations from 5 different detector types
2. Capturing and storing evidence snapshots
3. Orchestrating license plate recognition
4. Routing violations based on plate confidence
5. Managing the manual review workflow
6. Persisting data in dual format (JSON + CSV)
7. Thread-safe concurrent access

### Supported Violation Types

| Type                | Recording Method             | Trigger Source             |
| ------------------- | ---------------------------- | -------------------------- |
| Parking             | `record_parking_violation()` | `ParkingDetector`          |
| Speed               | `record_speed_violation()`   | `SpeedDetector`            |
| Wrong Direction     | `record_wrong_direction()`   | `WrongDirectionDetector`   |
| Traffic Light (Red) | `record_traffic_violation()` | `TrafficViolationDetector` |
| Helmet              | `record_helmet_violation()`  | `HelmetViolationDetector`  |

### Violation ID Format

Each violation receives a globally unique ID:

```
{counter:06d}_{uuid_hex[:6]}
```

**Example:** `000001_942df1`, `000042_a3bc7e`

The counter increments per `ViolationManager` instance. Combined with the UUID fragment, this ensures uniqueness even across restarts.

### Thread Safety

All recording and persistence operations are protected by `threading.Lock`:

```python
self.lock = threading.Lock()

def record_speed_violation(self, ...):
    with self.lock:
        # ... all operations atomically
```

This is critical because the API server uses threading (not multiprocessing), and multiple video processors may record violations concurrently.

---

## 3. Evidence Snapshot Pipeline

### Vehicle Snapshot

When a violation is recorded, the system captures a **cropped vehicle image** with surrounding context:

```
Original frame
  → Locate vehicle bounding box (x1, y1, x2, y2)
    → Expand by 45% margin on all sides
      → Clip to frame boundaries
        → Overlay violation type text + timestamp
          → Save as JPEG to violations/snapshots/
```

The 45% margin ensures the vehicle is visible in context — not just a tight crop, but including road markings, nearby vehicles, and surroundings that provide evidential context.

### Snapshot Naming

```
{violation_type}_{violation_id}_{stream_id}.jpg
```

**Example:** `speed_000015_camera1.jpg`

---

## 4. License Plate Recognition

### Architecture

```
┌────────────────────────┐
│  LicensePlateRecognizer │  (Queue-based wrapper)
│  process_violation_sync()│
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│  LicensePlateDetector   │  (Detection + OCR engine)
├─────────────────────────┤
│  1. YOLO Plate Detection│  (Licenses.pt, conf ≥ 0.6)
│  2. ROI Crop + Margin   │  (10% expansion)
│  3. Image Preprocessing │  (PIL: contrast 2.0×, sharpness 2.0×)
│  4. EasyOCR Recognition │  (English text extraction)
└────────────────────────┘
```

### Processing Steps

| Step                  | Tool                   | Detail                                                           |
| --------------------- | ---------------------- | ---------------------------------------------------------------- |
| Plate localization    | YOLOv8 (`Licenses.pt`) | Detects plate bounding box within vehicle crop, confidence ≥ 0.6 |
| ROI expansion         | OpenCV                 | 10% margin around detected plate for better OCR                  |
| Resize                | PIL                    | Upscale if plate crop is too small for reliable OCR              |
| Contrast enhancement  | PIL.ImageEnhance       | 2.0× contrast boost for visibility                               |
| Sharpness enhancement | PIL.ImageEnhance       | 2.0× sharpness boost for character clarity                       |
| OCR                   | EasyOCR (English)      | Extracts text characters from enhanced plate image               |

### EasyOCR Initialization

EasyOCR is initialized in a **background thread** to avoid blocking application startup (model loading takes several seconds):

```python
# Background thread loads OCR model
self.ocr_thread = threading.Thread(target=self._init_ocr)
self.ocr_thread.daemon = True
self.ocr_thread.start()
```

A readiness flag prevents OCR calls before initialization completes.

### Plate Image Persistence

The plate crop is **always saved** regardless of OCR success:

```
violations/plates/{violation_id}_plate.jpg
```

This ensures manual reviewers can always see the plate image even when OCR fails.

---

## 5. Routing Decision

After license plate recognition, the violation is routed based on OCR confidence:

```
                        ┌─────────────────────┐
                        │ Plate text exists?   │
                        │ Confidence ≥ 0.3?    │
                        └────┬──────────┬──────┘
                             │          │
                          YES ▼          ▼ NO
                   ┌──────────────┐  ┌───────────────┐
                   │  Confirmed   │  │ Manual Review  │
                   │  Violations  │  │   (Pending)    │
                   ├──────────────┤  ├───────────────┤
                   │ JSON + CSV   │  │ JSON only      │
                   │ Full record  │  │ Status: pending│
                   └──────────────┘  └───────────────┘
```

### Confirmed Violation Record Structure

```json
{
    "violation_id": "000015_a3bc7e",
    "type": "speed",
    "timestamp": "2026-02-23T14:30:15",
    "date": "2026-02-23",
    "time": "14:30:15",
    "stream_id": "camera1",
    "camera_location": "Main Street",
    "coordinates": {"lat": 30.0444, "lng": 31.2357},
    "vehicle_id": 42,
    "license_plate": "2835BSY",
    "plate_confidence": 0.85,
    "speed": 65.3,
    "speed_limit": 50.0,
    "snapshot_path": "snapshots/speed_000015_camera1.jpg",
    "plate_image_path": "plates/000015_a3bc7e_plate.jpg"
}
```

### Manual Review Record Structure

```json
{
    "review_id": "MR_000003_b2f4c1",
    "violation_id": "000016_d4e5f6",
    "type": "parking",
    "status": "pending",
    "timestamp": "2026-02-23T14:35:22",
    "stream_id": "camera1",
    "camera_location": "Main Street",
    "detected_plate": "",
    "plate_confidence": 0.0,
    "corrected_plate": null,
    "reviewer_notes": null,
    "snapshot_path": "snapshots/parking_000016_camera1.jpg",
    "plate_image_path": "plates/000016_d4e5f6_plate.jpg"
}
```

---

## 6. Manual Review Workflow

When OCR fails or confidence is too low, violations enter the **manual review queue**:

```
1. Violation recorded → OCR fails → manual_reviews.json (status: "pending")
2. Reviewer accesses /api/manual-reviews → sees pending items
3. Reviewer views snapshot + plate image via /api/media/
4. Reviewer submits corrected plate text
5. System moves violation from manual_reviews → unified_violations with corrected plate
6. Manual review status updated to "resolved"
```

### API Endpoints for Manual Review

| Method | Endpoint                         | Action                              |
| ------ | -------------------------------- | ----------------------------------- |
| GET    | `/api/manual-reviews`            | List all pending reviews            |
| GET    | `/api/manual-reviews/{id}`       | Get specific review details         |
| PUT    | `/api/manual-reviews/{id}`       | Update with corrected plate / notes |
| DELETE | `/api/manual-reviews/{id}`       | Remove review                       |
| GET    | `/api/media/manual-reviews/{id}` | View snapshot image                 |

---

## 7. AccidentAlertManager — Separate Alert Pipeline

**File:** `Violation_Proc/accident_alert_manager.py`  
**Class:** `AccidentAlertManager`

Accidents are handled **separately** from standard violations because:
- They require different evidence (close-up snapshots, not plate recognition)
- They have different urgency (immediate alert vs. violation logging)
- Privacy considerations (vehicle involvement details excluded from JSON)

### Accident Alert Record

```json
{
    "alert_id": "accident_camera1_1708704615",
    "timestamp": "2026-02-23T15:30:15",
    "stream_id": "camera1",
    "camera_location": "Main Street",
    "coordinates": {"lat": 30.0444, "lng": 31.2357},
    "snapshot_path": "snapshots/accident_camera1_1708704615.jpg",
    "severity": "high"
}
```

**Note:** `vehicles_involved` and `accident_class` are deliberately excluded from JSON persistence for privacy. They exist only in the CSV log.

### Directory Structure

```
Violation_Proc/accident_alerts/
├── all_accident_alerts.json     # Alert records (privacy-filtered)
├── all_accident_alerts.csv      # Full records including vehicle details
└── snapshots/                   # Close-up accident photos
```

---

## 8. Data Persistence Strategy

### Dual-Format Approach

All data is persisted in **both JSON and CSV**:

| Format | Purpose                          | Benefits                                             |
| ------ | -------------------------------- | ---------------------------------------------------- |
| JSON   | API serving, application state   | Structured, queryable, easily served via REST        |
| CSV    | Data export, analysis, reporting | Spreadsheet-compatible, lightweight, append-friendly |

### Atomic File Writes

To prevent data corruption from crashes or concurrent writes, the system uses the **temp file + atomic replace** pattern:

```python
# Write to temporary file first
temp_path = str(json_path) + '.tmp'
with open(temp_path, 'w') as f:
    json.dump(data, f, indent=2)
# Atomic replace (OS-level)
os.replace(temp_path, str(json_path))
```

`os.replace()` is atomic on most filesystems — the file is either the old version or the new version, never a partial write.

### Directory Structure

```
Violation_Proc/
├── violations/
│   ├── unified_violations.json    # All confirmed violations
│   ├── unified_violations.csv     # CSV mirror
│   ├── manual_reviews.json        # Pending manual reviews
│   ├── snapshots/                 # All violation type snapshots
│   │   ├── speed_000015_camera1.jpg
│   │   ├── parking_000016_camera1.jpg
│   │   └── ...
│   └── plates/                    # License plate crops
│       ├── 000015_a3bc7e_plate.jpg
│       └── ...
└── accident_alerts/
    ├── all_accident_alerts.json
    ├── all_accident_alerts.csv
    └── snapshots/
        └── accident_camera1_1708704615.jpg
```

---

## 9. Real-Time Notification (WebSocket)

### How It Works

In the API server mode (`api_server.py`), the `ViolationManager` is subclassed as `APIViolationManager`:

```python
class APIViolationManager(ViolationManager):
    def _update_json_log(self, violation_record):
        super()._update_json_log(violation_record)
        # Hook: broadcast violation via WebSocket
        asyncio.run(notification_handler.broadcast_violation(violation_record))
```

### Three WebSocket Channels

| Channel      | Path             | Events                           |
| ------------ | ---------------- | -------------------------------- |
| `violations` | `/ws/violations` | New violation recorded           |
| `accidents`  | `/ws/accidents`  | Accident alert triggered         |
| `system`     | `/ws/system`     | System status (every 10 seconds) |

### Connection Manager

The `ConnectionManager` class handles WebSocket lifecycle:
- Tracks active connections per channel
- Broadcasts JSON messages to all channel subscribers
- Auto-reconnect on client side with 5-second delay

---

## 10. Observed Test Data Analysis

### Violations Data (`unified_violations.json`)

The test data contains speed violations with:
- License plates: `"2835BSY"`, `"9079GCH"`, `"5553DN"`
- Speeds: **1.4–2.8 km/h** — these are extremely low, confirming the speed limit was set to 1 km/h for testing purposes
- All from the same stream with consistent GPS coordinates

### Manual Reviews Data (`manual_reviews.json`)

Contains pending reviews where plate detection returned:
- `plate_confidence: 0.00` — complete OCR failure
- Mix of speed and parking violation types
- Shows the routing logic correctly sends low-confidence results to manual review

### Accident Alerts (`all_accident_alerts.json`)

Empty array (`[]`) — no accidents were detected during testing, which is expected since accident detection is disabled by default.

---

## 11. Prepared but Unused: Django Integration

The codebase contains **commented-out Django ORM integration** in both `violation_manager.py` and `accident_alert_manager.py`:

```python
# Prepared Django model integration (commented out)
# from django.apps import apps
# ViolationRecord = apps.get_model('violations', 'ViolationRecord')
# record = ViolationRecord.objects.create(...)
```

This indicates the team planned for database integration but prioritized the file-based approach for the graduation project scope.

---

## 12. Summary Assessment

### Strengths

1. **Complete evidence chain** — Every violation produces vehicle snapshot + plate crop + structured record
2. **Smart routing** — Automatic separation of confirmed vs. uncertain violations
3. **Manual review workflow** — Handles OCR failures gracefully with human-in-the-loop correction
4. **Thread-safe operations** — All recording and persistence protected by locks
5. **Atomic file writes** — Prevents data corruption from crashes
6. **Dual format persistence** — JSON for API, CSV for analysis
7. **Separate accident pipeline** — Appropriate privacy filtering and different evidence requirements

### Areas for Improvement

1. **File-based persistence** — JSON/CSV does not scale; no indexing, no querying, no ACID transactions
2. **Synchronous plate recognition** — Blocks the detection loop during OCR processing
3. **English-only OCR** — Cannot read Arabic license plates (critical for Egypt/ME deployment)
4. **No violation expiry** — Violations accumulate indefinitely in JSON files
5. **No backup strategy** — File corruption means data loss (despite atomic writes)
6. **No notification delivery** — Email/SMS/webhook handlers exist in config but are not implemented
7. **Django integration incomplete** — ORM code is present but commented out
