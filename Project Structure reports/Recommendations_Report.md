# Vision Patrol — AI & Backend Recommendations Report

**Date:** February 23, 2026  
**Project:** Vision Patrol — AI-Powered Traffic Monitoring System  
**Report Scope:** Technical improvement recommendations for AI models, backend architecture, API design, and system reliability  

---

## 1. Recommendation Priority Matrix

| Priority   | Category | Recommendation                               | Impact                      | Effort |
| ---------- | -------- | -------------------------------------------- | --------------------------- | ------ |
| 🔴 Critical | Backend  | Migrate from JSON/CSV to database            | Scalability, data integrity | High   |
| 🔴 Critical | Backend  | Add authentication & authorization           | Security                    | Medium |
| 🔴 Critical | AI       | Add Arabic OCR support                       | Regional deployment         | Low    |
| 🟡 High     | AI       | Replace HSV traffic light detection with CNN | Accuracy, portability       | Medium |
| 🟡 High     | Backend  | Parse and use config.conf                    | Maintainability             | Low    |
| 🟡 High     | Backend  | Implement structured logging                 | Debugging, auditing         | Low    |
| 🟡 High     | AI       | Add perspective correction for speed         | Accuracy                    | Medium |
| 🟢 Medium   | Backend  | Async license plate processing               | Performance                 | Medium |
| 🟢 Medium   | AI       | Model optimization (TensorRT/ONNX)           | Performance                 | Medium |
| 🟢 Medium   | Backend  | Containerization (Docker)                    | Deployment                  | Medium |
| 🔵 Low      | AI       | Batch inference across streams               | GPU efficiency              | High   |
| 🔵 Low      | Backend  | CI/CD pipeline                               | Development velocity        | Medium |

---

## 2. AI Model Improvements

### 2.1 Replace HSV Traffic Light Detection with Trained Classifier

**Current Problem:**  
The `TrafficLightDetector` in `masking.py` uses hardcoded HSV color ranges (`H[58-179], S[97-255], V[222-255]` for green) and a hardcoded pixel filter (`cx >= 915`). This is:
- Fragile under varying lighting, shadows, weather
- Non-portable (recalibration needed per camera)
- Scene-specific (pixel threshold is fixed)

**Recommendation:**  
Train a YOLOv8 traffic light classifier with 3 classes: `red`, `green`, `yellow`.

**Implementation Approach:**
```
1. Collect traffic light images from target intersections
   - Day/night, sunny/cloudy/rainy conditions
   - Multiple camera angles
2. Label with red/green/yellow classes (use Roboflow or CVAT)
3. Fine-tune YOLOv8-nano (fastest) on the dataset
4. Replace masking.py with model inference
5. Use TRAFFIC_SIGN area as ROI to limit inference region
```

**Expected Improvement:** 30-50% reduction in false positives across different lighting conditions.

### 2.2 Add Perspective Correction for Speed Estimation

**Current Problem:**  
`SpeedDetector` uses a flat `pixels_per_meter` scalar for the entire frame. In reality, pixels near the camera represent smaller real-world distances than pixels farther away. This causes systematic speed errors:
- Vehicles close to camera → speed overestimated
- Vehicles far from camera → speed underestimated

**Recommendation:**  
Implement homography-based perspective correction:

```
1. Define 4 reference points on the road with known real-world distances
2. Compute homography matrix H using cv2.getPerspectiveTransform()
3. Transform pixel coordinates to real-world coordinates before distance calculation
4. Speed formula becomes:
   
   p_world = H × p_pixel
   distance_real = ||p_world_t2 - p_world_t1||
   speed = distance_real / Δt × 3.6
```

**Implementation in `speed_detector.py`:**
```python
class SpeedDetector:
    def __init__(self, ..., calibration_points=None):
        if calibration_points:
            src = np.float32(calibration_points['pixel'])
            dst = np.float32(calibration_points['world'])
            self.homography = cv2.getPerspectiveTransform(src, dst)
        else:
            self.homography = None

    def _pixel_to_world(self, px, py):
        if self.homography is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pt, self.homography)
            return world[0][0]
        return np.array([px / self.pixels_per_meter, py / self.pixels_per_meter])
```

**Calibration can be added to the web UI** — users mark 4 reference points on the road and enter their real-world distances.

### 2.3 Add Arabic License Plate OCR

**Current Problem:**  
EasyOCR is initialized with English only. Egyptian and Middle Eastern license plates contain Arabic numerals and text.

**Recommendation:**  
Enable multi-language OCR:

```python
# Current (English only):
self.reader = easyocr.Reader(['en'], gpu=True)

# Recommended (Arabic + English):
self.reader = easyocr.Reader(['ar', 'en'], gpu=True)
```

**Additional Steps:**
- Fine-tune the plate detection model (`Licenses.pt`) on Egyptian plate formats (rectangular, white/blue)
- Add post-processing for Egyptian plate patterns (e.g., Arabic province codes + digits)
- Consider PaddleOCR as an alternative — better Arabic support in benchmarks

### 2.4 Model Optimization for Production

**Current Problem:**  
YOLOv8 models run in PyTorch (FP32), and each process loads its own model copy.

**Recommendation Stack:**

| Optimization        | Tool                                       | Expected Speedup      |
| ------------------- | ------------------------------------------ | --------------------- |
| FP16 inference      | `model.export(format='engine', half=True)` | 1.5-2×                |
| TensorRT conversion | NVIDIA TensorRT                            | 2-4×                  |
| ONNX export         | `model.export(format='onnx')`              | 1.3-1.5× (CPU)        |
| Batch inference     | Custom pipeline                            | 1.5-2× (multi-stream) |
| Model sharing       | Shared GPU memory across processes         | ~50% memory reduction |

**Implementation Priority:**
1. TensorRT export (one-time conversion, major speedup with NVIDIA GPUs)
2. FP16 — nearly free performance gain
3. Batch inference — requires architecture change

### 2.5 Night/Low-Light Preprocessing

**Current Problem:**  
No image preprocessing for challenging lighting conditions. Nighttime detection accuracy will drop significantly.

**Recommendation:**
```python
def preprocess_low_light(frame):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

Apply CLAHE preprocessing when average frame brightness is below a threshold.

### 2.6 Model Versioning & Experiment Tracking

**Current Problem:**  
Model weights (`.pt` files) are stored directly in the repository with no versioning, no performance metrics history, and no rollback capability.

**Recommendation:**  
Adopt MLflow or Weights & Biases for model lifecycle management:

```
Training → Log metrics (mAP, precision, recall) → Register model version
  → A/B test in shadow mode → Promote to production
    → Monitor drift → Retrain when performance degrades
```

---

## 3. Backend Architecture Improvements

### 3.1 Database Migration (Critical)

**Current Problem:**  
All data is stored in flat JSON and CSV files. This means:
- No indexing → O(n) queries as data grows
- No ACID transactions → potential data corruption under concurrent writes
- No query language → cannot filter/aggregate violations efficiently
- File locking with `threading.Lock` doesn't prevent multi-process corruption

**Recommended Database: PostgreSQL**

**Schema Design:**

```sql
-- Core violations table
CREATE TABLE violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    violation_id VARCHAR(20) UNIQUE NOT NULL,
    type VARCHAR(30) NOT NULL,  -- speed, parking, traffic_light, helmet, wrong_direction
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stream_id VARCHAR(100) NOT NULL,
    camera_location VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    vehicle_id INTEGER,
    license_plate VARCHAR(20),
    plate_confidence DECIMAL(4, 3),
    speed DECIMAL(6, 2),
    speed_limit DECIMAL(6, 2),
    snapshot_path VARCHAR(500),
    plate_image_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'confirmed',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Manual reviews table
CREATE TABLE manual_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    violation_id UUID REFERENCES violations(id),
    status VARCHAR(20) DEFAULT 'pending',
    detected_plate VARCHAR(20),
    corrected_plate VARCHAR(20),
    reviewer_notes TEXT,
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Accident alerts table
CREATE TABLE accident_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    stream_id VARCHAR(100) NOT NULL,
    camera_location VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    accident_class VARCHAR(50),
    severity VARCHAR(20),
    snapshot_path VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Video sources table
CREATE TABLE video_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    source_path VARCHAR(500) NOT NULL,
    is_stream BOOLEAN DEFAULT FALSE,
    location VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    speed_limit DECIMAL(6, 2) DEFAULT 60.0,
    status VARCHAR(20) DEFAULT 'stopped',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_violations_type ON violations(type);
CREATE INDEX idx_violations_timestamp ON violations(timestamp);
CREATE INDEX idx_violations_stream ON violations(stream_id);
CREATE INDEX idx_violations_plate ON violations(license_plate);
CREATE INDEX idx_manual_reviews_status ON manual_reviews(status);
CREATE INDEX idx_accidents_timestamp ON accident_alerts(timestamp);
```

**ORM Recommendation:** SQLAlchemy (async support with `asyncpg`) or the existing Django ORM (prepared code is already commented out).

**Migration Path:**
1. Install PostgreSQL (or use managed: AWS RDS, Supabase)
2. Add SQLAlchemy models mirroring the schema above
3. Implement repository pattern to abstract persistence
4. Migrate existing JSON data with a one-time script
5. Keep CSV export as a feature (not primary storage)

### 3.2 Authentication & Authorization

**Current Problem:**  
Zero authentication on all 30+ API endpoints. Anyone with network access can:
- Start/stop video sources
- Delete violations
- Modify area configurations
- View all data

**Recommendation: JWT + RBAC**

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

ROLES = {
    "admin": ["*"],
    "operator": ["read", "start", "stop", "review"],
    "viewer": ["read"]
}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(role: str):
    async def checker(user = Depends(get_current_user)):
        if role not in ROLES.get(user.get("role", ""), []) and "*" not in ROLES.get(user.get("role", ""), []):
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return checker
```

### 3.3 Parse and Use config.conf

**Current Problem:**  
The `config.conf` file contains 15+ sections with proper configuration values — but **no code ever reads it**. All values are hardcoded throughout the modules.

**Recommendation:**

```python
import configparser

class Config:
    _instance = None

    def __new__(cls, path="config.conf"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = configparser.ConfigParser()
            cls._instance._config.read(path)
        return cls._instance

    @property
    def accident_confidence(self):
        return self._config.getfloat('AccidentDetection', 'confidence_threshold', fallback=0.86)

    @property
    def speed_limit(self):
        return self._config.getfloat('SpeedDetection', 'speed_limit', fallback=60.0)

    # ... etc for all parameters
```

Then inject this singleton into detectors during initialization.

### 3.4 Structured Logging

**Current Problem:**  
All logging is `print()` statements. No log levels, no timestamps, no structured format, no log aggregation.

**Recommendation:**

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "stream_id": getattr(record, 'stream_id', 'system')
        }
        return json.dumps(log_data)

# Setup
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
```

**For production:** Ship logs to ELK stack (Elasticsearch + Logstash + Kibana) or Loki + Grafana.

### 3.5 Async License Plate Processing

**Current Problem:**  
License plate recognition runs **synchronously** in the violation recording path. YOLO inference + EasyOCR takes 200-500ms per violation, blocking the detection loop.

**Recommendation: Background Task Queue**

```
Option A: Celery + Redis
  Violation detected → Quick snapshot save → Enqueue plate recognition task
    → Celery worker picks up task → YOLO + OCR → Update violation record

Option B: FastAPI BackgroundTasks (simpler)
  @app.post("/internal/process-plate")
  async def process_plate(violation_id: str, background_tasks: BackgroundTasks):
      background_tasks.add_task(run_plate_recognition, violation_id)

Option C: asyncio.Queue (current architecture, minimal change)
  Already partially implemented in LicensePlateRecognizer.process_violation()
  → Enable the queue-based path instead of process_violation_sync()
```

**Recommended: Option C** for minimal disruption — the async queue code already exists but is unused.

### 3.6 Health Checks & Monitoring

**Current Problem:**  
No observability. If a video processor hangs or GPU memory leaks, there's no way to know until manual inspection.

**Recommendation:**

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

violations_total = Counter('violations_total', 'Total violations', ['type', 'stream_id'])
frame_processing_time = Histogram('frame_processing_seconds', 'Frame processing time')
active_streams = Gauge('active_video_streams', 'Number of active video streams')
gpu_memory_usage = Gauge('gpu_memory_bytes', 'GPU memory usage in bytes')

# Health endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sources": len([s for s in sources.values() if s.status == "running"]),
        "uptime": time.time() - start_time,
        "gpu_available": torch.cuda.is_available()
    }
```

**Monitoring stack:** Prometheus → Grafana dashboards.

---

## 4. API Design Improvements

### 4.1 Fix Duplicate Endpoint Definitions

**Current Problem:**  
`api_server.py` contains duplicate route definitions for model settings GET/PUT.

**Action:** Audit all routes and remove duplicates.

### 4.2 Add Pagination

**Current Problem:**  
`GET /api/violations` returns ALL violations — unsustainable as data grows.

**Recommendation:**

```python
@app.get("/api/violations")
async def get_violations(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    # With database: SELECT ... LIMIT per_page OFFSET (page-1)*per_page
    return {
        "data": paginated_violations,
        "page": page,
        "per_page": per_page,
        "total": total_count,
        "pages": math.ceil(total_count / per_page)
    }
```

### 4.3 API Versioning

**Recommendation:** Prefix all routes with `/api/v1/` for forward compatibility:

```python
v1 = APIRouter(prefix="/api/v1")

@v1.get("/violations")
async def get_violations():
    ...

app.include_router(v1)
```

### 4.4 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/violations")
@limiter.limit("60/minute")
async def get_violations(request: Request):
    ...
```

---

## 5. Testing Strategy

**Current Problem:**  
`pytest` is in `requirements.txt` but no test files exist.

**Recommended Test Structure:**

```
tests/
├── unit/
│   ├── test_speed_detector.py
│   ├── test_parking_detector.py
│   ├── test_area_manager.py
│   ├── test_violation_manager.py
│   └── test_wrong_direction.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_video_processor.py
│   └── test_plate_recognition.py
└── fixtures/
    ├── sample_frames/
    ├── area_configs/
    └── mock_violations.json
```

**Example test for SpeedDetector:**

```python
def test_speed_calculation():
    detector = SpeedDetector(pixels_per_meter=30)
    tracked = {1: {'center_points': [(100, 100), (130, 100)]}}  # 30px = 1m in 1/30s
    # Expected: 1m / (1/30)s = 30 m/s = 108 km/h
    speed = detector._calculate_raw_speed(1, tracked[1])
    assert 100 < speed < 115  # Allow smoothing tolerance
```

---

## 6. DevOps & Deployment

### 6.1 Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python3", "api_server.py"]
```

### 6.2 docker-compose.yml

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./area_configs:/app/area_configs
      - violation_data:/app/Violation_Proc/violations
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: vision_patrol
      POSTGRES_USER: vp_admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  pgdata:
  violation_data:
```

### 6.3 CI/CD Pipeline (GitHub Actions)

```yaml
name: Vision Patrol CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ -v
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ruff
      - run: ruff check .
```

---

## 7. Summary: Top 10 Actions

| #   | Action                         | Files Affected                                      | Estimated Effort |
| --- | ------------------------------ | --------------------------------------------------- | ---------------- |
| 1   | Add Arabic OCR support         | `license_plate_detector.py`                         | 1-2 hours        |
| 2   | Parse `config.conf` everywhere | All detector files + `api_server.py`                | 1 day            |
| 3   | Add JWT authentication         | `api_server.py`                                     | 1-2 days         |
| 4   | Replace print() with logging   | All files                                           | 1 day            |
| 5   | Enable async plate processing  | `license_plate_recognizer.py`                       | 2-4 hours        |
| 6   | Add PostgreSQL persistence     | `violation_manager.py`, `accident_alert_manager.py` | 3-5 days         |
| 7   | Train traffic light classifier | `masking.py`, `traffic_violation_detector.py`       | 1-2 weeks        |
| 8   | Add homography for speed       | `speed_detector.py`, area config UI                 | 1 week           |
| 9   | Dockerize the application      | New files                                           | 1 day            |
| 10  | Add unit tests                 | New test files                                      | 1 week           |
