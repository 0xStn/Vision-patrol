# Vision Patrol - API Server for Traffic Monitoring System
# Provides RESTful API and WebSocket interface for backend integration

import os
import sys
import uuid
import time
import json
import csv
import asyncio
import threading
import uvicorn
import base64
import cv2
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Query, Body, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator, HttpUrl

# Configure Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add Processing Models directory to path
processing_models_path = os.path.join(current_dir, "Processing Models")
sys.path.append(processing_models_path)

# Add Models directory to path
models_dir = os.path.join(current_dir, "Models")
sys.path.append(models_dir)

# Add Violation_Proc directory to path
violation_proc_dir = os.path.join(current_dir, "Violation_Proc")
sys.path.append(violation_proc_dir)

# Import project modules
from VideoProcessorMP import VideoProcessorMP
from areas import AreaManager, AreaType
from Violation_Proc.violation_manager import ViolationManager
from Violation_Proc.accident_alert_manager import AccidentAlertManager

# Create FastAPI application instance
app = FastAPI(
    title="Vision Patrol API",
    description="""
    API for traffic monitoring system that processes video streams to detect violations and accidents.
    This API allows backend developers to integrate the system with their database and frontend applications.
    """,
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Create directories for storing uploads and snapshots if they don't exist
UPLOAD_DIR = os.path.join(current_dir, "uploads")
SNAPSHOT_DIR = os.path.join(current_dir, "snapshots")
AREA_CONFIG_DIR = os.path.join(current_dir, "area_configs")
STATIC_DIR = os.path.join(current_dir, "static")  # Directory for static web files

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(AREA_CONFIG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)  # Ensure static directory exists

# IMPORTANT: Set up static file serving for snapshots only
# Don't mount the root static files yet - we'll do that after defining all API routes
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "violations": [],
            "accidents": [],
            "system": []
        }

    async def connect(self, websocket: WebSocket, channel: str):
        if channel not in self.active_connections:
            raise ValueError(f"Invalid channel: {channel}")
        await websocket.accept()
        self.active_connections[channel].append(websocket)
        return f"Connected to {channel} channel"

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)

    async def broadcast_to_channel(self, message: dict, channel: str):
        if channel not in self.active_connections:
            return
            
        # Broadcast message to all connections in the channel
        disconnected = []
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                print(f"Error sending message: {str(e)}")
                disconnected.append(connection)
        
        # Clean up disconnected websockets
        for conn in disconnected:
            if conn in self.active_connections[channel]:
                self.active_connections[channel].remove(conn)

# Initialize connection manager
manager = ConnectionManager()

# Global state to track video processors
video_processors = {}
video_source_info = {}
model_settings = {}

# Define models for API
class Coordinates(BaseModel):
    lat: float = 0.0
    lng: float = 0.0

class VideoSource(BaseModel):
    name: str
    source: str
    use_stream: bool = False
    location: str = "Unknown"
    coordinates: Optional[Coordinates] = None
    enabled: bool = True
    speed_limit: Optional[float] = 60.0  # Default to 60 km/h if not specified
    
class VideoFrame(BaseModel):
    source_id: str
    frame_data: str  # Base64 encoded image data

class VideoSourceResponse(BaseModel):
    id: str
    name: str
    source: str
    use_stream: bool
    location: str
    coordinates: Coordinates
    speed_limit: float  # Added speed limit to response
    status: str
    created_at: str
    
class ModelSettings(BaseModel):
    accident_detection: bool = True
    helmet_detection: bool = True
    traffic_violation: bool = True
    speed_detection: bool = True
    parking_detection: bool = True
    wrong_direction: bool = True

class AreaPoint(BaseModel):
    x: int
    y: int

class AreaDefinition(BaseModel):
    type: str  # Must be one of AreaType values
    points: List[List[int]]  # List of [x,y] points

class AreaConfiguration(BaseModel):
    source_id: str
    areas: Dict[str, List[Dict[str, List[List[int]]]]]  # Type -> List of areas with points

class ViolationFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    violation_type: Optional[str] = None
    source_id: Optional[str] = None
    
class AccidentFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    source_id: Optional[str] = None

class ManualReviewWithViolationRequest(BaseModel):
    violation_data: Dict[str, Any]  # Complete violation data
    reason: str = "License plate recognition failed"
    notes: Optional[str] = None

class ManualReviewUpdate(BaseModel):
    license_plate: str
    notes: Optional[str] = None
    reviewer_name: Optional[str] = None

class ManualReviewResponse(BaseModel):
    review_id: str
    violation_id: str
    violation_data: Dict[str, Any]
    reason: str
    notes: Optional[str]
    license_plate: Optional[str]
    reviewer_name: Optional[str]
    created_at: str
    updated_at: Optional[str]
    status: str  # "pending", "completed"

class SystemStatus(BaseModel):
    active_sources: int
    active_connections: Dict[str, int]
    system_uptime: float


    
# Event handlers for violation and accident notifications
class NotificationHandler:
    def __init__(self, connection_manager):
        self.manager = connection_manager
        self.start_time = time.time()
        
    async def handle_violation(self, violation_data):
        """Handle a new violation event and broadcast to WebSocket clients"""
        # Broadcast to violation channel
        await self.manager.broadcast_to_channel({
            "event": "violation",
            "data": violation_data,
            "snapshot_url": f"/snapshots/violations/{violation_data.get('snapshot_path', '')}",
        }, "violations")
        
    async def handle_accident(self, accident_data):
        """Handle a new accident event and broadcast to WebSocket clients"""
        # Broadcast to accident channel
        await self.manager.broadcast_to_channel({
            "event": "accident",
            "data": accident_data,
            "snapshot_url": f"/snapshots/accidents/{accident_data.get('snapshot_path', '')}",
        }, "accidents")
        
    async def broadcast_system_status(self):
        """Broadcast system status to WebSocket clients"""
        status = {
            "active_sources": len(video_processors),
            "active_connections": {
                channel: len(connections) 
                for channel, connections in self.manager.active_connections.items()
            },
            "system_uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.manager.broadcast_to_channel({
            "event": "status_update",
            "data": status
        }, "system")
    
# Initialize notification handler
notification_handler = NotificationHandler(manager)

# Background task to periodically broadcast system status
async def status_broadcast_task():
    while True:
        await notification_handler.broadcast_system_status()
        await asyncio.sleep(10)  # Broadcast every 10 seconds

# Custom ViolationManager for API integration that overrides _update_json_log and other methods
class APIViolationManager(ViolationManager):
    def __init__(self, stream_id, camera_location, coordinates, notification_handler):
        super().__init__(stream_id=stream_id, camera_location=camera_location, coordinates=coordinates)
        self.notification_handler = notification_handler
        
    def _update_json_log(self, violation_record):
        """Override to send violation data to WebSocket clients"""
        # First call the parent method to update the JSON file
        super()._update_json_log(violation_record)
        
        # Then send notification through WebSocket
        asyncio.run(self.notification_handler.handle_violation(violation_record))
        
# Custom AccidentAlertManager for API integration
class APIAccidentAlertManager(AccidentAlertManager):
    def __init__(self, stream_id, camera_location, coordinates, notification_handler):
        super().__init__(stream_id=stream_id, camera_location=camera_location, coordinates=coordinates)
        self.notification_handler = notification_handler
        
    def _update_json_log(self, accident_record):
        """Override to send accident data to WebSocket clients"""
        # First call the parent method to update the JSON file
        super()._update_json_log(accident_record)
        
        # Then send notification through WebSocket
        asyncio.run(self.notification_handler.handle_accident(accident_record))

# Custom VideoProcessorMP for API integration
class APIVideoProcessor:
    def __init__(self, video_id, source, use_stream=False, camera_location="Unknown", coordinates=None, 
                 notification_handler=None, speed_limit=60):
        self.video_id = video_id
        self.source = source
        self.use_stream = use_stream
        self.camera_location = camera_location
        self.coordinates = coordinates or {"lat": 0.0, "lng": 0.0}
        self.notification_handler = notification_handler
        self.status = "initializing"
        self.processor = None
        self.process_thread = None
        self.speed_limit = speed_limit
        
        # Set default model settings
        global model_settings
        model_settings[video_id] = {
            "accident_detection": True,
            "helmet_detection": True,
            "traffic_violation": True,
            "speed_detection": True,
            "parking_detection": True,
            "wrong_direction": True
        }
        
    def start(self):
        """Start video processing in a separate thread"""
        if self.process_thread and self.process_thread.is_alive():
            return False
            
        # Initialize the video processor
        try:
            # Create a custom violation manager that uses our notification handler
            violation_manager = APIViolationManager(
                stream_id=self.video_id,
                camera_location=self.camera_location,
                coordinates=self.coordinates,
                notification_handler=self.notification_handler
            )
            
            # Create the VideoProcessorMP instance with speed limit
            self.processor = VideoProcessorMP(
                video_id=self.video_id,
                source=self.source,
                use_stream=self.use_stream,
                camera_location=self.camera_location,
                coordinates=self.coordinates,
                speed_limit=self.speed_limit
            )
            
            # Replace the violation_manager in the processor
            self.processor.violation_manager = violation_manager
            
            # Start the processor in a separate thread
            self.process_thread = threading.Thread(target=self._run_processor)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            self.status = "running"
            return True
        except Exception as e:
            self.status = f"error: {str(e)}"
            return False
            
    def _run_processor(self):
        """Run the processor in a thread"""
        try:
            self.processor.run()
        except Exception as e:
            self.status = f"error: {str(e)}"
        finally:
            self.status = "stopped"
            
    def stop(self):
        """Stop video processing"""
        try:
            if self.processor:
                self.processor.exit.set()
                
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=5)
                
            self.status = "stopped"
            return True
        except Exception as e:
            self.status = f"error during stop: {str(e)}"
            return False
            
    def set_area_configuration(self, area_config):
        """Set area configuration for the processor"""
        if not self.processor or not hasattr(self.processor, 'area_manager'):
            return False
            
        try:
            # Convert area config to area_manager format and apply
            for area_type_str, areas in area_config.items():
                area_type = getattr(AreaType, area_type_str.upper(), None)
                if area_type:
                    # Clear existing areas of this type
                    self.processor.area_manager.areas[area_type] = []
                    
                    # Add new areas
                    for area in areas:
                        self.processor.area_manager.areas[area_type].append({
                            'points': [tuple(point) for point in area['points']]
                        })
            
            return True
        except Exception as e:
            print(f"Error setting area configuration: {str(e)}")
            return False
            
    def get_area_configuration(self):
        """Get current area configuration from the processor"""
        if not self.processor or not hasattr(self.processor, 'area_manager'):
            return {}

        try:
            # Convert area_manager format to API format
            area_config = {}
            for area_type, areas in self.processor.area_manager.areas.items():
                area_type_str = area_type.name.lower()
                area_config[area_type_str] = []

                for area in areas:
                    if 'points' in area:
                        area_config[area_type_str].append({
                            'points': [list(point) for point in area['points']]
                        })

            return area_config
        except Exception as e:
            print(f"Error getting area configuration: {str(e)}")
            return {}



    def clear_all_area_configurations(self):
        """Clear all area configurations and restart (delete config file and reset)"""
        if not self.processor or not hasattr(self.processor, 'area_manager'):
            return False

        try:
            # Use the area manager's clear and restart method
            success = self.processor.area_manager.clear_all_and_restart()

            if success:
                print(f"Successfully cleared all area configurations for processor {self.video_id}")

            return success
        except Exception as e:
            print(f"Error clearing all area configurations: {str(e)}")
            return False
    def apply_model_settings(self, settings):
        """Apply model settings to the processor"""
        if not self.processor:
            return False
            
        try:
            global model_settings
            model_settings[self.video_id] = settings
            
            # Apply settings directly to the processor's model_settings
            if hasattr(self.processor, 'model_settings'):
                self.processor.model_settings = {
                    "accident_detection": settings.get('accident_detection', True),
                    "helmet_detection": settings.get('helmet_detection', True),
                    "traffic_violation": settings.get('traffic_violation', True),
                    "speed_detection": settings.get('speed_detection', True),
                    "parking_detection": settings.get('parking_detection', True),
                    "wrong_direction": settings.get('wrong_direction', True)
                }
            
            # Also apply to specific detectors that have direct enable/disable properties
            if hasattr(self.processor, 'accident_detector'):
                self.processor.accident_detector.detection_enabled = settings.get('accident_detection', True)
                
            if hasattr(self.processor, 'helmet_detector'):
                self.processor.helmet_detector.detection_enabled = settings.get('helmet_detection', True)
                
            if hasattr(self.processor, 'traffic_violation_detector'):
                self.processor.traffic_violation_detector.detection_enabled = settings.get('traffic_violation', True)
                
            if hasattr(self.processor, 'speed_detector'):
                self.processor.speed_detector.detection_enabled = settings.get('speed_detection', True)
                
            if hasattr(self.processor, 'parking_detector'):
                self.processor.parking_detector.detection_enabled = settings.get('parking_detection', True)
                
            if hasattr(self.processor, 'wrong_direction_detector'):
                self.processor.wrong_direction_detector.detection_enabled = settings.get('wrong_direction', True)
                
            return True
        except Exception as e:
            print(f"Error applying model settings: {str(e)}")
            return False
    
    def get_model_settings(self):
        """Get current model settings"""
        global model_settings
        return model_settings.get(self.video_id, {})

# Function to create and start a video processor
def create_video_processor(video_source: VideoSource):
    """Create and start a video processor for a video source"""
    # Generate a unique ID if not provided
    video_id = str(uuid.uuid4())
    
    # Save video source info
    source_info = {
        "id": video_id,
        "name": video_source.name,
        "source": video_source.source,
        "use_stream": video_source.use_stream,
        "location": video_source.location,
        "coordinates": video_source.coordinates.dict() if video_source.coordinates else {"lat": 0.0, "lng": 0.0},
        "speed_limit": video_source.speed_limit,  # Include speed limit in source info
        "status": "initializing",
        "created_at": datetime.now().isoformat()
    }
    video_source_info[video_id] = source_info
    
    # Create video processor
    processor = APIVideoProcessor(
        video_id=video_id,
        source=video_source.source,
        use_stream=video_source.use_stream,
        camera_location=video_source.location,
        coordinates=source_info["coordinates"],
        notification_handler=notification_handler,
        speed_limit=video_source.speed_limit  # Pass speed_limit to processor
    )
    
    # Start video processor
    success = processor.start()
    if not success:
        # Update status
        source_info["status"] = processor.status
    else:
        # Update status
        source_info["status"] = "running"
        
    # Store processor
    video_processors[video_id] = processor
    
    return video_id, source_info

# Function to get violation data from JSON file
def get_violations(filter_params=None):
    """Get violation data from the ViolationManager JSON file"""
    json_path = os.path.join(violation_proc_dir, "violations", "unified_violations.json")
    
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                violations = json.load(f)
                
            # Apply filters if provided
            if filter_params:
                filtered_violations = []
                for violation in violations:
                    include = True
                    
                    # Filter by date range
                    if filter_params.start_date and violation.get('date') < filter_params.start_date:
                        include = False
                    if filter_params.end_date and violation.get('date') > filter_params.end_date:
                        include = False
                        
                    # Filter by violation type
                    if filter_params.violation_type and violation.get('violation_type') != filter_params.violation_type:
                        include = False
                        
                    # Filter by source ID
                    if filter_params.source_id and violation.get('camera_id') != filter_params.source_id:
                        include = False
                        
                    if include:
                        filtered_violations.append(violation)
                        
                return filtered_violations
            
            return violations
        else:
            return []
    except Exception as e:
        print(f"Error getting violations: {str(e)}")
        return []

# Function to get accident data from JSON file
def get_accidents(filter_params=None):
    """Get accident data from the AccidentAlertManager JSON file"""
    json_path = os.path.join(violation_proc_dir, "accident_alerts", "all_accident_alerts.json")

    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                accidents = json.load(f)

            # Apply filters if provided
            if filter_params:
                filtered_accidents = []
                for accident in accidents:
                    include = True

                    # Filter by date range
                    if filter_params.start_date and accident.get('date') < filter_params.start_date:
                        include = False
                    if filter_params.end_date and accident.get('date') > filter_params.end_date:
                        include = False

                    # Filter by source ID
                    if filter_params.source_id and accident.get('camera_id') != filter_params.source_id:
                        include = False

                    if include:
                        filtered_accidents.append(accident)

                return filtered_accidents

            return accidents
        else:
            return []
    except Exception as e:
        print(f"Error getting accidents: {str(e)}")
        return []

# Manual review functions
def get_manual_reviews():
    """Get all manual review data from JSON file"""
    json_path = os.path.join(violation_proc_dir, "violations", "manual_reviews.json")

    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                reviews = json.load(f)
            return reviews
        else:
            return []
    except Exception as e:
        print(f"Error getting manual reviews: {str(e)}")
        return []

def save_manual_reviews(reviews):
    """Save manual reviews to JSON file"""
    json_path = os.path.join(violation_proc_dir, "violations", "manual_reviews.json")

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Write to temporary file first, then replace
        temp_file = json_path + '.tmp'
        with open(temp_file, 'w') as f:
            json.dump(reviews, f, indent=2)

        # Replace original file
        os.replace(temp_file, json_path)
        return True
    except Exception as e:
        print(f"Error saving manual reviews: {str(e)}")
        return False



def create_manual_review_with_violation_data(violation_data, reason="Manual review required", notes=None):
    """Create a manual review with custom violation data (for cases where violation shouldn't be logged)"""
    try:
        # Check if manual review already exists for this violation
        violation_id = violation_data.get('violation_id')
        if not violation_id:
            return None, "Violation ID is required"

        existing_reviews = get_manual_reviews()
        for review in existing_reviews:
            if review.get('violation_id') == violation_id:
                return None, "Manual review already exists for this violation"

        # Generate unique review ID
        review_id = f"MR_{uuid.uuid4().hex[:8]}"

        # Create review record
        review_record = {
            'review_id': review_id,
            'violation_id': violation_id,
            'violation_data': violation_data,
            'reason': reason,
            'notes': notes,
            'license_plate': None,
            'reviewer_name': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': None,
            'status': 'pending'
        }

        # Add to existing reviews
        existing_reviews.append(review_record)

        # Save to file
        if save_manual_reviews(existing_reviews):
            return review_record, None
        else:
            return None, "Failed to save manual review"

    except Exception as e:
        return None, f"Error creating manual review with violation data: {str(e)}"

def update_manual_review(review_id, license_plate, notes=None, reviewer_name=None):
    """Update manual review with license plate and complete it"""
    try:
        reviews = get_manual_reviews()

        # Check if review exists first
        review_found = False
        for review in reviews:
            if review.get('review_id') == review_id:
                review_found = True
                break

        if not review_found:
            return None, "Manual review not found"

        for review in reviews:
            if review.get('review_id') == review_id:
                # Update fields
                review['license_plate'] = license_plate
                review['notes'] = notes
                review['reviewer_name'] = reviewer_name
                review['updated_at'] = datetime.now().isoformat()
                review['status'] = 'completed'

                # Add violation to database with corrected license plate
                violation_data = review.get('violation_data', {})
                violation_data['license_plate'] = license_plate
                violation_data['manual_review_completed'] = True
                violation_data['manual_review_date'] = datetime.now().isoformat()

                add_violation_to_database(violation_data)

                # Save updated reviews
                if save_manual_reviews(reviews):
                    return review, None
                else:
                    return None, "Failed to save updated manual review"

    except Exception as e:
        return None, f"Error updating manual review: {str(e)}"

def add_violation_to_database(violation_data):
    """Add a violation to the database (JSON and CSV)"""
    try:
        json_path = os.path.join(violation_proc_dir, "violations", "unified_violations.json")

        # Add to JSON
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                violations = json.load(f)
        else:
            violations = []

        violation_id = violation_data.get('violation_id')

        # Check if violation already exists
        violation_exists = any(v.get('violation_id') == violation_id for v in violations)
        if not violation_exists:
            violations.append(violation_data)

            # Save updated violations
            temp_file = json_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(violations, f, indent=2)

            os.replace(temp_file, json_path)
            print(f"Added violation {violation_id} to JSON database")

        # Add to CSV
        csv_path = os.path.join(violation_proc_dir, "violations", "unified_violations.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    violation_data.get('date', ''),
                    violation_data.get('time', ''),
                    violation_data.get('license_plate', ''),
                    violation_data.get('camera_id', ''),
                    violation_data.get('camera_location', ''),
                    violation_data.get('violation_type', ''),
                    violation_data.get('violation_id', ''),
                    violation_data.get('vehicle_id', '')
                ])
            print(f"Added violation {violation_id} to CSV database")

        return True

    except Exception as e:
        print(f"Error adding violation to database: {str(e)}")
        return False

def delete_manual_review(review_id):
    """Delete a manual review"""
    try:
        reviews = get_manual_reviews()
        original_count = len(reviews)

        # Remove the review
        reviews = [review for review in reviews if review.get('review_id') != review_id]

        if len(reviews) == original_count:
            return False, "Manual review not found"

        # Save updated reviews
        if save_manual_reviews(reviews):
            return True, None
        else:
            return False, "Failed to save after deletion"

    except Exception as e:
        return False, f"Error deleting manual review: {str(e)}"

def update_violation_license_plate(violation_id, license_plate):
    """Update the license plate in the original violation record"""
    try:
        json_path = os.path.join(violation_proc_dir, "violations", "unified_violations.json")

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                violations = json.load(f)

            # Find and update the violation
            violation_updated = False
            for violation in violations:
                if violation.get('violation_id') == violation_id:
                    violation['license_plate'] = license_plate
                    violation['manual_review_completed'] = True
                    violation['manual_review_date'] = datetime.now().isoformat()
                    violation_updated = True
                    print(f"Updated violation {violation_id} with license plate: {license_plate}")
                    break

            if not violation_updated:
                print(f"Warning: Violation {violation_id} not found for license plate update")
                return False

            # Save updated violations with atomic operation
            temp_file = json_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(violations, f, indent=2)

            os.replace(temp_file, json_path)

            # Also update CSV if it exists
            csv_path = os.path.join(violation_proc_dir, "violations", "unified_violations.csv")
            if os.path.exists(csv_path):
                update_csv_license_plate(csv_path, violation_id, license_plate)

            return True

    except Exception as e:
        print(f"Error updating violation license plate: {str(e)}")
        return False

def update_csv_license_plate(csv_path, violation_id, license_plate):
    """Update license plate in CSV file"""
    try:
        import csv

        # Read all rows
        rows = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Find the violation_id column index (should be column 6 based on the structure)
        header = rows[0] if rows else []
        violation_id_col = 6  # Default based on current structure
        license_plate_col = 2  # Default based on current structure

        # Try to find the correct column indices
        if 'Violation_ID' in header:
            violation_id_col = header.index('Violation_ID')
        if 'License_Plate' in header:
            license_plate_col = header.index('License_Plate')

        # Update the specific row
        updated = False
        for i, row in enumerate(rows):
            if i > 0 and len(row) > max(violation_id_col, license_plate_col):
                if row[violation_id_col] == violation_id:
                    row[license_plate_col] = license_plate
                    updated = True
                    print(f"Updated CSV row for violation {violation_id}")
                    break

        if updated:
            # Write back with atomic operation
            temp_file = csv_path + '.tmp'
            with open(temp_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

            os.replace(temp_file, csv_path)
        else:
            print(f"Warning: Violation {violation_id} not found in CSV for update")

    except Exception as e:
        print(f"Error updating CSV license plate: {str(e)}")



# API endpoints
@app.get("/api")
async def api_root():
    """API root that returns status - use this to check if API is running"""
    return {"message": "Traffic Monitoring API is running"}

# Video source management endpoints
@app.post("/api/sources", response_model=VideoSourceResponse)
async def add_video_source(video_source: VideoSource):
    """Add a new video source for processing"""
    # Check if source exists and is accessible
    if not video_source.use_stream and not os.path.exists(video_source.source):
        if video_source.source.startswith(("http://", "https://", "rtsp://")):
            # Treat as stream even if use_stream is False
            video_source.use_stream = True
        else:
            raise HTTPException(status_code=404, detail=f"Video source file not found: {video_source.source}")
    
    # Create and start video processor
    try:
        video_id, source_info = create_video_processor(video_source)
        return source_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create video processor: {str(e)}")

@app.get("/api/sources", response_model=List[VideoSourceResponse])
async def list_video_sources():
    """List all video sources"""
    return list(video_source_info.values())

@app.get("/api/sources/{source_id}", response_model=VideoSourceResponse)
async def get_video_source(source_id: str):
    """Get details for a specific video source"""
    if source_id not in video_source_info:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    return video_source_info[source_id]

@app.delete("/api/sources/{source_id}")
async def remove_video_source(source_id: str):
    """Remove a video source and stop processing"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    # Stop video processor
    processor = video_processors[source_id]
    processor.stop()
    
    # Remove from dictionaries
    del video_processors[source_id]
    del video_source_info[source_id]
    
    return {"message": f"Video source removed: {source_id}"}

# Processing control endpoints
@app.post("/api/sources/{source_id}/start")
async def start_processing(source_id: str):
    """Start processing for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    processor = video_processors[source_id]
    success = processor.start()
    
    if success:
        video_source_info[source_id]["status"] = "running"
        return {"message": f"Processing started for: {source_id}"}
    else:
        video_source_info[source_id]["status"] = processor.status
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {processor.status}")

@app.post("/api/sources/{source_id}/stop")
async def stop_processing(source_id: str):
    """Stop processing for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    processor = video_processors[source_id]
    success = processor.stop()
    
    if success:
        video_source_info[source_id]["status"] = "stopped"
        return {"message": f"Processing stopped for: {source_id}"}
    else:
        video_source_info[source_id]["status"] = processor.status
        raise HTTPException(status_code=500, detail=f"Failed to stop processing: {processor.status}")

# Area configuration endpoints
@app.get("/api/sources/{source_id}/areas")
async def get_area_configuration(source_id: str):
    """Get area configuration for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    processor = video_processors[source_id]
    area_config = processor.get_area_configuration()
    
    return {
        "source_id": source_id,
        "areas": area_config
    }

@app.put("/api/sources/{source_id}/areas")
async def update_area_configuration(source_id: str, config: Dict[str, Any]):
    """Update area configuration for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    processor = video_processors[source_id]
    success = processor.set_area_configuration(config)
    
    if success:
        # Save area configuration to file
        config_path = os.path.join(AREA_CONFIG_DIR, f"{source_id}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return {"message": f"Area configuration updated for: {source_id}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to update area configuration")

@app.post("/api/sources/{source_id}/areas/import")
async def import_area_configuration(source_id: str, file: UploadFile = File(...)):
    """Import area configuration for a video source from a file"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    try:
        # Read config from uploaded file
        content = await file.read()
        config = json.loads(content.decode('utf-8'))
        
        # Apply configuration
        processor = video_processors[source_id]
        success = processor.set_area_configuration(config)
        
        if success:
            # Save configuration to file
            config_path = os.path.join(AREA_CONFIG_DIR, f"{source_id}.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            return {"message": f"Area configuration imported for: {source_id}"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to apply imported configuration")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to import configuration: {str(e)}")



@app.post("/api/sources/{source_id}/areas/clear-all")
async def clear_all_area_configurations(source_id: str):
    """Clear all area configurations and restart (delete config file and reset)"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")

    processor = video_processors[source_id]

    # Clear all areas and restart using the processor method
    success = processor.clear_all_area_configurations()

    if success:
        # Also remove the API-side config file if it exists
        import os
        api_config_path = os.path.join(AREA_CONFIG_DIR, f"{source_id}.json")
        if os.path.exists(api_config_path):
            try:
                os.remove(api_config_path)
                print(f"Removed API config file: {api_config_path}")
            except Exception as e:
                print(f"Error removing API config file: {str(e)}")

        return {"message": f"All area configurations cleared and restarted for: {source_id}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to clear area configurations")

@app.get("/api/sources/{source_id}/areas/export")
async def export_area_configuration(source_id: str):
    """Export area configuration for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")

# Model settings endpoints
@app.get("/api/sources/{source_id}/settings")
async def get_source_settings(source_id: str):
    """Get model settings for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
    
    processor = video_processors[source_id]
    
    # Return the model settings
    return {
        "source_id": source_id,
        "models": processor.get_model_settings()
    }

@app.put("/api/sources/{source_id}/settings")
async def update_source_settings(source_id: str, settings: Dict[str, Any] = Body(...)):
    """Update model settings for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
    
    processor = video_processors[source_id]
    
    # Apply model settings
    if "models" in settings:
        success = processor.apply_model_settings(settings["models"])
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to update model settings")
    
    return {"message": f"Settings updated successfully for: {source_id}"}

# Violation data access endpoints
@app.get("/api/violations")
async def get_all_violations():
    """Get all violation data without filtering"""
    violations = get_violations()
    return violations

@app.post("/api/violations")
async def get_filtered_violations(filter_params: ViolationFilter = Body(default=None)):
    """Get violation data with optional filtering"""
    violations = get_violations(filter_params)
    return violations

@app.get("/api/violations/{violation_id}")
async def get_violation_details(violation_id: str):
    """Get details for a specific violation"""
    violations = get_violations()
    for violation in violations:
        if violation.get('violation_id') == violation_id:
            return violation
            
    raise HTTPException(status_code=404, detail=f"Violation not found: {violation_id}")

@app.get("/api/violations/statistics")
async def get_violation_statistics():
    """Get statistics about violations"""
    violations = get_violations()
    
    # Compute statistics
    stats = {
        "total": len(violations),
        "by_type": {},
        "by_source": {},
        "by_date": {}
    }
    
    for violation in violations:
        # Count by type
        violation_type = violation.get('violation_type', 'unknown')
        if violation_type not in stats["by_type"]:
            stats["by_type"][violation_type] = 0
        stats["by_type"][violation_type] += 1
        
        # Count by source
        source_id = violation.get('camera_id', 'unknown')
        if source_id not in stats["by_source"]:
            stats["by_source"][source_id] = 0
        stats["by_source"][source_id] += 1
        
        # Count by date
        date = violation.get('date', 'unknown')
        if date not in stats["by_date"]:
            stats["by_date"][date] = 0
        stats["by_date"][date] += 1
    
    return stats

@app.get("/api/violations/types/{type}")
async def get_violations_by_type(type: str):
    """Get violations of a specific type"""
    filter_params = ViolationFilter(violation_type=type)
    violations = get_violations(filter_params)
    return violations

# Manual review endpoints - Essential CRUD only
@app.post("/api/manual-reviews", response_model=ManualReviewResponse)
async def create_manual_review_with_violation_endpoint(request: ManualReviewWithViolationRequest):
    """Create a manual review with complete violation data"""
    review_record, error = create_manual_review_with_violation_data(
        violation_data=request.violation_data,
        reason=request.reason,
        notes=request.notes
    )

    if error:
        raise HTTPException(status_code=400, detail=error)

    return review_record

@app.get("/api/manual-reviews", response_model=List[ManualReviewResponse])
async def get_all_manual_reviews():
    """Get all manual reviews"""
    reviews = get_manual_reviews()
    return reviews

@app.get("/api/manual-reviews/{review_id}", response_model=ManualReviewResponse)
async def get_manual_review_details(review_id: str):
    """Get details for a specific manual review"""
    reviews = get_manual_reviews()
    for review in reviews:
        if review.get('review_id') == review_id:
            return review

    raise HTTPException(status_code=404, detail=f"Manual review not found: {review_id}")

@app.put("/api/manual-reviews/{review_id}", response_model=ManualReviewResponse)
async def update_manual_review_endpoint(review_id: str, update_data: ManualReviewUpdate):
    """Update a manual review with license plate information"""
    updated_review, error = update_manual_review(
        review_id=review_id,
        license_plate=update_data.license_plate,
        notes=update_data.notes,
        reviewer_name=update_data.reviewer_name
    )

    if error:
        if "not found" in error.lower():
            raise HTTPException(status_code=404, detail=error)
        else:
            raise HTTPException(status_code=400, detail=error)

    return updated_review

@app.delete("/api/manual-reviews/{review_id}")
async def delete_manual_review_endpoint(review_id: str):
    """Delete a manual review"""
    success, error = delete_manual_review(review_id)

    if error:
        if "not found" in error.lower():
            raise HTTPException(status_code=404, detail=error)
        else:
            raise HTTPException(status_code=400, detail=error)

    if not success:
        raise HTTPException(status_code=404, detail=f"Manual review not found: {review_id}")

    return {"message": f"Manual review deleted: {review_id}"}

# Accident alerts access endpoints
@app.post("/api/accidents")
async def get_filtered_accidents(filter_params: AccidentFilter = Body(default=None)):
    """Get accident data with optional filtering"""
    accidents = get_accidents(filter_params)
    return accidents

@app.get("/api/accidents/statistics")
async def get_accident_statistics():
    """Get statistics about accidents"""
    accidents = get_accidents()

    # Compute statistics
    stats = {
        "total": len(accidents),
        "by_source": {},
        "by_date": {}
    }

    for accident in accidents:
        # Count by source
        source_id = accident.get('camera_id', 'unknown')
        if source_id not in stats["by_source"]:
            stats["by_source"][source_id] = 0
        stats["by_source"][source_id] += 1

        # Count by date
        date = accident.get('date', 'unknown')
        if date not in stats["by_date"]:
            stats["by_date"][date] = 0
        stats["by_date"][date] += 1

    return stats

@app.get("/api/accidents/{accident_id}")
async def get_accident_details(accident_id: str):
    """Get details for a specific accident"""
    accidents = get_accidents()
    for accident in accidents:
        if accident.get('accident_id') == accident_id:
            return accident

    raise HTTPException(status_code=404, detail=f"Accident not found: {accident_id}")

# Media access endpoints
@app.get("/api/media/violations/{violation_id}")
async def get_violation_snapshot(violation_id: str):
    """Get snapshot image for a specific violation"""
    violations = get_violations()
    for violation in violations:
        if violation.get('violation_id') == violation_id:
            snapshot_path = violation.get('snapshot_path')
            if snapshot_path:
                # Handle both forward and backward slashes in paths
                normalized_path = snapshot_path.replace('\\', os.sep).replace('/', os.sep)
                full_path = os.path.join(violation_proc_dir, "violations", normalized_path)

                if os.path.exists(full_path):
                    return FileResponse(full_path)
                else:
                    # Try alternative path without violations subdirectory
                    alt_path = os.path.join(violation_proc_dir, normalized_path)
                    if os.path.exists(alt_path):
                        return FileResponse(alt_path)
                    else:
                        raise HTTPException(status_code=404, detail=f"Snapshot not found at: {full_path} or {alt_path}")
            else:
                raise HTTPException(status_code=404, detail=f"No snapshot path available for violation: {violation_id}")

    raise HTTPException(status_code=404, detail=f"Violation not found: {violation_id}")

@app.get("/api/media/accidents/{accident_id}")
async def get_accident_snapshot(accident_id: str):
    """Get snapshot image for a specific accident"""
    accidents = get_accidents()
    for accident in accidents:
        if accident.get('accident_id') == accident_id:
            snapshot_paths = accident.get('snapshot_paths', {})
            closeup_path = snapshot_paths.get('closeup')
            if closeup_path:
                full_path = os.path.join(violation_proc_dir, "accident_alerts", closeup_path)
                if os.path.exists(full_path):
                    return FileResponse(full_path)
                else:
                    raise HTTPException(status_code=404, detail=f"Snapshot not found: {closeup_path}")
            else:
                raise HTTPException(status_code=404, detail=f"No snapshot available for accident: {accident_id}")
            
    raise HTTPException(status_code=404, detail=f"Accident not found: {accident_id}")

@app.get("/api/media/plates/{violation_id}")
async def get_license_plate_snapshot(violation_id: str):
    """Get license plate snapshot for a specific violation"""
    violations = get_violations()
    for violation in violations:
        if violation.get('violation_id') == violation_id:
            # Try both possible field names for plate snapshot path
            plate_path = violation.get('plate_snapshot_path') or violation.get('license_plate_path')
            if plate_path and plate_path.strip():
                # Handle both forward and backward slashes in paths
                normalized_path = plate_path.replace('\\', os.sep).replace('/', os.sep)
                full_path = os.path.join(violation_proc_dir, "violations", normalized_path)

                if os.path.exists(full_path):
                    return FileResponse(full_path)
                else:
                    # Try alternative path without violations subdirectory
                    alt_path = os.path.join(violation_proc_dir, normalized_path)
                    if os.path.exists(alt_path):
                        return FileResponse(alt_path)
                    else:
                        raise HTTPException(status_code=404, detail=f"License plate snapshot not found at: {full_path} or {alt_path}")
            else:
                raise HTTPException(status_code=404, detail=f"No license plate snapshot path available for violation: {violation_id}")

    raise HTTPException(status_code=404, detail=f"Violation not found: {violation_id}")

@app.get("/api/media/manual-reviews/{review_id}/plate")
async def get_manual_review_plate_snapshot(review_id: str):
    """Get license plate snapshot for a manual review"""
    reviews = get_manual_reviews()
    for review in reviews:
        if review.get('review_id') == review_id:
            violation_data = review.get('violation_data', {})
            plate_path = violation_data.get('plate_snapshot_path')
            if plate_path:
                full_path = os.path.join(violation_proc_dir, "violations", plate_path)
                if os.path.exists(full_path):
                    return FileResponse(full_path)
                else:
                    raise HTTPException(status_code=404, detail="Plate snapshot not found")
            else:
                raise HTTPException(status_code=404, detail="No plate snapshot available")
    raise HTTPException(status_code=404, detail="Manual review not found")

@app.get("/api/media/manual-reviews/{review_id}/violation")
async def get_manual_review_violation_snapshot(review_id: str):
    """Get violation snapshot for a manual review"""
    reviews = get_manual_reviews()
    for review in reviews:
        if review.get('review_id') == review_id:
            violation_data = review.get('violation_data', {})
            snapshot_path = violation_data.get('snapshot_path')
            if snapshot_path:
                full_path = os.path.join(violation_proc_dir, "violations", snapshot_path)
                if os.path.exists(full_path):
                    return FileResponse(full_path)
                else:
                    raise HTTPException(status_code=404, detail="Violation snapshot not found")
            else:
                raise HTTPException(status_code=404, detail="No violation snapshot available")
    raise HTTPException(status_code=404, detail="Manual review not found")

# System management endpoints
@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status"""
    return {
        "active_sources": len(video_processors),
        "active_connections": {
            channel: len(connections) 
            for channel, connections in manager.active_connections.items()
        },
        "system_uptime": time.time() - notification_handler.start_time
    }

# WebSocket endpoints
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """WebSocket endpoint for real-time updates"""
    if channel not in ["violations", "accidents", "system"]:
        await websocket.close(code=1008, reason=f"Invalid channel: {channel}")
        return
        
    try:
        await manager.connect(websocket, channel)
        
        # Send initial welcome message
        await websocket.send_json({
            "event": "connected",
            "message": f"Connected to {channel} channel",
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Keep connection alive while receiving messages
            while True:
                # Wait for messages from client (might be ping/pong or commands)
                message = await websocket.receive_text()
                
                # Process any commands here if needed
                if message == "ping":
                    await websocket.send_json({"event": "pong", "timestamp": datetime.now().isoformat()})
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket, channel)
            
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
        except:
            pass

# Video stream WebSocket endpoint
@app.websocket("/ws/video/{source_id}")
async def video_stream_websocket(websocket: WebSocket, source_id: str):
    """WebSocket endpoint for streaming video frames"""
    if source_id not in video_processors:
        await websocket.close(code=1008, reason=f"Video source not found: {source_id}")
        return
        
    processor = video_processors[source_id]
    
    try:
        await websocket.accept()
        
        # Send initial message
        await websocket.send_json({
            "event": "connected",
            "source_id": source_id,
            "timestamp": datetime.now().isoformat()
        })
        
        frame_rate = 5  # Frames per second
        frame_interval = 1.0 / frame_rate
        
        # Stream frames until connection is closed
        try:
            while True:
                start_time = time.time()
                
                # Check if processor is running
                if processor.status != "running" or not processor.processor:
                    await websocket.send_json({
                        "event": "error",
                        "message": "Video source is not running",
                        "timestamp": datetime.now().isoformat()
                    })
                    await asyncio.sleep(1)
                    continue
                
                # Get frame if possible
                if hasattr(processor.processor, 'video_reader') and processor.processor.video_reader:
                    # Read a frame
                    ret, frame = processor.processor.video_reader.read()
                    
                    if not ret or frame is None:
                        # Try to reset and read again
                        processor.processor.video_reader.reset()
                        ret, frame = processor.processor.video_reader.read()
                        
                        if not ret or frame is None:
                            await websocket.send_json({
                                "event": "error",
                                "message": "Failed to read frame from video source",
                                "timestamp": datetime.now().isoformat()
                            })
                            await asyncio.sleep(1)
                            continue
                    
                    # Resize frame for streaming (reduce bandwidth)
                    height, width = frame.shape[:2]
                    max_width = 640
                    if width > max_width:
                        scale = max_width / width
                        new_width = max_width
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame
                    await websocket.send_json({
                        "event": "frame",
                        "source_id": source_id,
                        "frame_data": frame_data,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Calculate sleep time to maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except WebSocketDisconnect:
            print(f"Video stream WebSocket disconnected: {source_id}")
            
    except Exception as e:
        print(f"Video stream WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
        except:
            pass

# Start background tasks on startup
@app.on_event("startup")
async def startup_event():
    # Start status broadcast task
    asyncio.create_task(status_broadcast_task())
    print("System startup completed. WebSocket broadcast task started.")

    # Mount static files AFTER all routes are registered
    # This is crucial to prevent the static files from intercepting API requests
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# Clean up resources on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    # Stop all video processors
    for processor in video_processors.values():
        processor.stop()
    print("System shutdown completed. All video processors stopped.")

# Frame extraction endpoint
@app.get("/api/sources/{source_id}/frame")
async def get_video_frame(source_id: str):
    """Get a single frame from a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail=f"Video source not found: {source_id}")
        
    processor = video_processors[source_id]
    
    try:
        # Get the current frame from the processor if it's running
        if processor.status == "running" and processor.processor:
            # Check if processor has access to the frame
            if hasattr(processor.processor, 'video_reader') and processor.processor.video_reader:
                # Read a frame
                ret, frame = processor.processor.video_reader.read()
                
                if not ret or frame is None:
                    # Try to reset and read again
                    processor.processor.video_reader.reset()
                    ret, frame = processor.processor.video_reader.read()
                    
                    if not ret or frame is None:
                        raise HTTPException(status_code=500, detail="Failed to read frame from video source")
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                # Include frame dimensions
                height, width = frame.shape[:2]
                
                return {
                    "source_id": source_id,
                    "frame_data": frame_data,
                    "width": width,
                    "height": height
                }
            else:
                raise HTTPException(status_code=500, detail="Video reader not initialized")
        else:
            raise HTTPException(status_code=400, detail="Video source is not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting video frame: {str(e)}")

# Add API endpoints for model settings
@app.get("/api/sources/{source_id}/settings", response_model=dict)
def get_source_settings(source_id: str):
    """Get settings for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail="Video source not found")
    
    processor = video_processors[source_id]
    
    # Return the model settings
    return {"models": processor.get_model_settings()}

@app.put("/api/sources/{source_id}/settings", response_model=dict)
def update_source_settings(source_id: str, settings: dict = Body(...)):
    """Update settings for a video source"""
    if source_id not in video_processors:
        raise HTTPException(status_code=404, detail="Video source not found")
    
    processor = video_processors[source_id]
    
    # Apply model settings
    if "models" in settings:
        processor.apply_model_settings(settings["models"])
    
    return {"status": "success", "message": "Settings updated successfully"}

# Run the server if this script is executed
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
