# Vision Patrol — AI Detection Models Report

**Date:** February 23, 2026  
**Project:** Vision Patrol — AI-Powered Traffic Monitoring System  
**Report Scope:** All AI/ML detection models and algorithmic detectors  

---

## 1. Overview

Vision Patrol employs **4 custom-trained YOLOv8 models**, **1 pre-trained COCO model**, **1 OCR engine**, and **2 rule-based algorithmic detectors** to cover 6 violation types plus accident detection. All deep learning models use the Ultralytics YOLOv8 framework built on PyTorch.

### Model Inventory

| Model                        | Weight File         | Type                  | Classes                                      | Confidence Threshold |
| ---------------------------- | ------------------- | --------------------- | -------------------------------------------- | -------------------- |
| Vehicle Detection & Tracking | `yolov8s.pt`        | Pre-trained COCO      | person, bicycle, car, motorcycle, bus, truck | 0.5 (default)        |
| Accident Detection           | `accident.pt`       | Custom-trained YOLOv8 | 7 accident types                             | 0.86                 |
| Helmet Detection             | `helmet_ds.pt`      | Custom-trained YOLOv8 | with-helmet, no-helmet                       | 0.55                 |
| License Plate Detection      | `Licenses.pt`       | Custom-trained YOLOv8 | license_plate                                | 0.6                  |
| License Plate OCR            | EasyOCR (English)   | Pre-trained           | Text characters                              | N/A                  |
| Traffic Light Detection      | Algorithmic (HSV)   | Rule-based            | RED, GREEN                                   | N/A                  |
| Parking Detection            | Algorithmic (Timer) | Rule-based            | N/A                                          | N/A                  |

---

## 2. Primary Detection — YOLOv8s (Vehicle Tracking)

**File:** `yolov8s.pt` (loaded in `VideoProcessorMP.py`)  
**Framework:** Ultralytics YOLOv8 Small  
**Role:** Core backbone for all downstream detectors  

### How It Works

The system uses YOLOv8 Small pre-trained on the COCO dataset. It runs `model.track(persist=True)` on every frame to detect and persistently track objects across frames using built-in BoT-SORT tracking.

**Tracked COCO Classes:**
- Vehicles: car (2), motorcycle (3), bus (5), truck (7)
- Vulnerable road users: person (1), bicycle (4)

**Tracked Object Data Structure (per vehicle ID):**
- `center_points` — deque of last 10 center coordinates `(cx, cy)`
- `crossed_lines` — set of line IDs the vehicle has crossed
- `box` — current bounding box `(x1, y1, x2, y2)`
- `class_id` — COCO class ID
- `confidence` — detection confidence score
- `last_seen` — frame counter (timeout after 30 frames unseen)

### Performance Characteristics

- **YOLOv8s** is the "small" variant — balances speed (~40 FPS on GPU) and accuracy (~44.9% mAP on COCO)
- Each `VideoProcessorMP` process loads its own model instance
- GPU memory per instance: ~25–50 MB

---

## 3. Accident Detector

**File:** `Models/Accident_det/Accident_Detector.py`  
**Class:** `AccidentDetector`  
**Model:** `accident.pt` (custom-trained YOLOv8)  
**Default State:** Disabled (toggled with 'a' key)

### 7 Accident Classes

1. `bike_bike_accident`
2. `bike_object_accident`
3. `bike_person_accident`
4. `car_bike_accident`
5. `car_car_accident`
6. `car_object_accident`
7. `car_person_accident`

### Detection Pipeline

```
Frame → Skip check (every 3rd frame) → YOLO inference (conf ≥ 0.86)
  → Filter accident classes → Sort by confidence → Take highest
    → Cooldown check (35s between alerts)
      → Auto-disable check (3 in 80s → disable 10 min)
        → Deduplication (180px radius, last 5 locations)
          → Close-up snapshot (75% margin) → AccidentAlertManager
```

### Key Anti-False-Positive Mechanisms

| Mechanism                 | Parameter                            | Purpose                                                 |
| ------------------------- | ------------------------------------ | ------------------------------------------------------- |
| High confidence threshold | 0.86                                 | Only strong detections trigger alerts                   |
| Alert cooldown            | 35 seconds                           | Prevents repeated alerts for same ongoing accident      |
| Frame skip                | Every 3rd frame                      | Reduces compute; smooths flickering detections          |
| Auto-disable              | 3 accidents in 80s → 10 min cooldown | Prevents cascade of false positives in ambiguous scenes |
| Area deduplication        | 180px radius, deque of 5             | Same physical area doesn't trigger multiple alerts      |
| Vehicle ID tracking       | `vehicles_in_accident` set           | Same vehicle pair doesn't re-trigger                    |

### Snapshot Strategy

The detector extracts a **close-up crop** with 75% margin around the accident bounding box. This gives a focused evidence image while retaining surrounding context.

---

## 4. Helmet Violation Detector

**File:** `Models/Bike_Violations/helmet_detector.py`  
**Class:** `HelmetViolationDetector`  
**Model:** `helmet_ds.pt` (custom-trained YOLOv8)  
**Default State:** Disabled (toggled with 'h' key)

### Two-Stage Detection Pipeline

```
Stage 1: Filter tracked objects for motorcycles (class 3) / bicycles (class 4)
  → Check if vehicle is in DETECTION area (AreaManager)
    → Stage 2: Crop ROI with 50% top padding (helmet region)
      → Run helmet model on ROI
        → Class 0: "with-helmet" → No violation
        → Class 1: "no-helmet" → Flag violation
```

### Key Features

- **Permanent violation tracking:** Once a vehicle ID is flagged as violating, it is never re-checked (`permanent_violations` set)
- **Throttled inference:** Minimum 1 second between helmet checks per vehicle (`min_inference_interval = 1.0`)
- **ROI expansion:** 50% top padding on the motorcycle bounding box to capture the rider's head area
- **Violation trigger:** Motorcycle exits the detection area without a helmet being detected at any point

### Limitations

- The 2-class model (`with-helmet` / `no-helmet`) may struggle with partially visible helmets or unusual helmet colors
- ROI expansion assumes the rider's head is above the motorcycle bounding box — may miss in edge cases (leaning, multiple riders)

---

## 5. Speed Detector

**File:** `Models/Speed Model/speed_detector.py`  
**Class:** `SpeedDetector`  
**Model:** None (algorithmic — uses tracked positions from YOLOv8)  
**Default State:** Disabled until speed lines are defined

### Speed Calculation Algorithm

The speed detector uses pixel-distance-to-real-distance conversion with temporal smoothing:

**Core Formula:**

$$v_{km/h} = \frac{d_{pixels}}{PPM \times \Delta t} \times 3.6$$

Where:
- $d_{pixels}$ = Euclidean distance between two consecutive center points
- $PPM$ = `pixels_per_meter` calibration factor (default: 30)
- $\Delta t$ = Fixed time step $\frac{1}{30}$ seconds (30 FPS assumption)
- $3.6$ = m/s to km/h conversion factor

### Smoothing & Stability

| Parameter               | Value     | Purpose                                       |
| ----------------------- | --------- | --------------------------------------------- |
| `smoothing_window`      | 10 frames | Exponential-weighted moving average window    |
| `max_speed_change`      | 5 km/h    | Maximum allowed speed delta between frames    |
| `min_distance_for_calc` | 3.0 px    | Minimum movement to avoid noise-induced speed |
| Speed cap               | 200 km/h  | Reject physically implausible values          |

### Violation Recording

Speed is **recorded at line crossings** — when a vehicle crosses a line defined as `AreaType.SPEED`, its smoothed speed is compared against `speed_limit` (default: 1 km/h for testing). Each vehicle can only trigger one violation per speed line.

### Known Limitations

- **No perspective correction:** The fixed `pixels_per_meter` scalar does not account for camera angle or lens distortion. Objects farther from camera appear slower.
- **Fixed FPS assumption:** Uses $\frac{1}{30}$s time step regardless of actual frame rate, which may cause inaccuracy with variable frame rates.
- **Testing default:** Speed limit is set to 1 km/h — clearly a testing value, not production-ready.

---

## 6. Traffic Violation Detector

**Files:**  
- `Models/Traffic_violation/masking.py` — HSV traffic light classifier  
- `Models/Traffic_violation/traffic_violation_detector.py` — Violation logic  
- `Models/Traffic_violation/Traffic_detector.py` — Legacy standalone version  

**Class:** `TrafficViolationDetector` (production) + `TrafficLightDetector` (HSV engine)  
**Default State:** Enabled

### Traffic Light Detection (HSV-based)

The system detects traffic light state using color space analysis:

```
Frame ROI (from TRAFFIC_SIGN area) → BGR to HSV conversion
  → Green mask: H[58–179], S[97–255], V[222–255]
  → Red mask: H[0–56], S[43–132], V[184–255]
    → Combined mask → Threshold → Find contours
      → Filter by area (≥50 px) and position (cx < 915)
        → Classify: Green pixels → GREEN / Red pixels → RED
```

### Red-Light Violation Detection

```
Traffic light state = RED?
  → Vehicle crosses TRAFFIC_LINE?
    → Cooldown check (100 frames per vehicle)?
      → Record violation → ViolationManager
```

### Known Limitations

- **Hardcoded HSV ranges** are tuned for specific lighting conditions. Different environments require re-calibration.
- **Hardcoded pixel filter** (`cx >= 915` skip) is scene-specific and non-portable.
- **No deep learning:** HSV thresholding is fragile under varying lighting, shadows, and weather conditions compared to a trained classifier.
- A `calibrate_color_detection()` method exists for parameter tuning but requires manual effort.

---

## 7. Parking Detector

**File:** `Models/Parking Model/parking_Detctor.py`  
**Class:** `ParkingDetector`  
**Model:** None (pure rule-based timer)

### Algorithm

```
Vehicle center point in PARKING area?
  → YES: Start timer (or continue existing)
    → Duration ≥ 15 seconds (violation_time_limit)?
      → Flag violation → Single snapshot → ViolationManager
  → NO: Remove from tracking
```

### Visual Feedback

The detector provides a **color gradient** on the vehicle bounding box:
- **Green** → just entered (0–5 seconds)
- **Yellow** → approaching limit (5–15 seconds)
- **Red** → violation triggered (≥15 seconds)
- **Red dot indicator** appears on violation

### Strengths

- Simple, reliable, and computationally free (no ML inference)
- Leverages the area management system for zone definition
- Single snapshot per violation prevents storage bloat

---

## 8. Wrong Direction Detector

**File:** `Models/Wrong_dir/wrong_det.py`  
**Class:** `WrongDirectionDetector`  
**Model:** None (geometric line-intersection algorithm)  
**Default State:** Disabled until line pairs are configured

### Line-Pair Virtual Gate System

The detector uses **pairs of lines** (entry + exit) to define directional gates:

```
Line Pair Configuration:
  Entry Line (Line 1) ─────────────
                        Traffic Flow →
  Exit Line (Line 2)  ─────────────

Correct Direction: Vehicle crosses Entry line FIRST, then Exit line
Wrong Direction: Vehicle crosses Exit line FIRST, then Entry line
```

### Direction Detection Algorithm

Uses **orientation-based line segment intersection** test:

```
For each vehicle with ≥ 2 tracked positions:
  1. Construct movement vector from last 2 positions
  2. Test intersection with Entry line (cross product method)
  3. Test intersection with Exit line
  4. If Exit line crossed before Entry → WRONG DIRECTION
  5. If Entry line crossed before Exit → FORWARD (correct)
```

The cross product determines which side of a line a point falls on, enabling directionality.

### Features

- **Direction enum:** `FORWARD`, `BACKWARD`, `UNKNOWN`
- **Auto-enable:** Automatically enables when valid line pairs are configured via `AreaManager`
- **Highlight duration:** 1 second visual indicator after detection
- **Per-vehicle state tracking** prevents duplicate violations

---

## 9. License Plate Recognition Pipeline

**Files:**  
- `Violation_Proc/License_recognition/license_plate_detector.py`  
- `Violation_Proc/License_recognition/license_plate_recognizer.py`  

### Two-Stage Pipeline

```
Vehicle violation snapshot
  → Stage 1: YOLO plate detection (Licenses.pt, conf ≥ 0.6)
    → Crop plate region with 10% margin
      → Stage 2: Image preprocessing
        → Resize (if too small) → Contrast enhancement (2.0×) → Sharpness enhancement (2.0×)
          → EasyOCR text recognition (English)
            → Output: plate_text, confidence, plate_image
```

### Routing Decision

| Condition                              | Destination                            |
| -------------------------------------- | -------------------------------------- |
| Plate text exists AND confidence ≥ 0.3 | `unified_violations.json` (confirmed)  |
| Plate text missing OR confidence < 0.3 | `manual_reviews.json` (pending review) |

### Technical Details

- **EasyOCR initialization:** Runs in a background thread for non-blocking startup
- **Thread-safe:** OCR readiness is checked before processing
- **Image preprocessing:** PIL-based contrast and sharpness enhancement improves OCR accuracy on low-quality footage
- **Plate crop saving:** Always saves the plate crop image regardless of OCR success (supports manual review)

### Limitations

- **English-only OCR:** Current configuration does not support Arabic script (critical gap for Egypt/Middle East deployment)
- **Single language model:** No multi-language support configured
- **Synchronous processing:** License plate recognition runs synchronously in the violation recording path, blocking the main detection loop

---

## 10. Summary Assessment

### Strengths

1. **Comprehensive coverage** — 6 violation types + accident detection from a single pipeline
2. **Layered anti-false-positive** mechanisms in accident detection (cooldown, auto-disable, deduplication)
3. **Self-contained detectors** — each module can be independently toggled, configured, and tested
4. **Evidence chain** — every violation produces a vehicle snapshot + plate crop for accountability
5. **Custom-trained models** for domain-specific tasks (accidents, helmets, plates)

### Areas for Improvement

1. **HSV traffic light detection** should be replaced with a trained classifier for robustness across environments
2. **Speed estimation** needs perspective correction (homography) for accuracy at different distances from camera
3. **License plate OCR** needs Arabic language support for ME/Egypt deployment
4. **Model loading** is per-process — no shared GPU memory or batch inference across streams
5. **No model versioning** — no MLflow, W&B, or registry for tracking model iterations
6. **Config file is never parsed** — all thresholds are hardcoded, making tuning require code changes
