import os
import csv
import json
import time
from datetime import datetime
import cv2
from pathlib import Path
import threading

class AccidentAlertManager:
    """
    Dedicated manager for accident alerts.
    Handles logging, image storage, and alert notifications for the alert system.
    """
    
    def __init__(self, base_dir=None, stream_id="default", camera_location="Unknown", coordinates=None):
        self.stream_id = stream_id
        self.camera_location = camera_location
        self.coordinates = coordinates or {"lat": 0.0, "lng": 0.0}  # Default coordinates
        self.lock = threading.Lock()  # Thread-safe operations
        
        # Setup directories
        if base_dir is None:
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "accident_alerts"
        else:
            base_dir = Path(base_dir) / "accident_alerts"
            
        self.base_dir = base_dir
        self._setup_directories()
        self._setup_logs()
        
        print(f"[{self.stream_id}] Accident Alert Manager initialized for {camera_location}")
        print(f"[{self.stream_id}] Location coordinates: {self.coordinates}")
        
    def _setup_directories(self):
        """Create organized directory structure for accident alerts"""
        # Main accident alerts directory 
        self.base_dir.mkdir(exist_ok=True)
        
        # Create single snapshots directory for all accidents
        self.snapshots_dir = self.base_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
    def _setup_logs(self):
        """Initialize centralized CSV and JSON log files"""
        # Create centralized logs at base directory level
        # No camera-specific or date-specific folders for logs
        
        # Centralized CSV log file with all accident alerts
        self.csv_path = self.base_dir / "all_accident_alerts.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Date', 'Time', 'Camera_ID', 'Camera_Location', 
                    'Latitude', 'Longitude', 'Accident_ID', 'Vehicles_Involved'
                ])
        
        # Centralized JSON log file with all accident alerts
        self.json_path = self.base_dir / "all_accident_alerts.json"
        if not self.json_path.exists():
            with open(self.json_path, 'w') as f:
                json.dump([], f)
    
    def record_accident(self, frame, vehicles_involved, bbox=None, accident_class=None):
        """
        Record a single accident alert snapshot
        
        Args:
            frame: The video frame containing the accident (already cropped closeup)
            vehicles_involved: List of vehicle IDs involved in the accident
            bbox: Optional bounding box of the accident area
            
        Returns:
            Tuple of (accident_id, snapshot_paths)
        """
        with self.lock:  # Thread-safe operation
            # Generate unique accident ID
            timestamp = datetime.now()
            date_str = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H:%M:%S')
            accident_id = f"accident_{self.stream_id}_{int(timestamp.timestamp())}"
            
            # Save the accident snapshot (already a closeup)
            snapshot_paths = self._save_accident_snapshot(frame, accident_id, vehicles_involved)
            
            # Create accident record
            accident_record = {
                'date': date_str,
                'time': time_str,
                'camera_id': self.stream_id,
                'camera_location': self.camera_location,
                'coordinates': self.coordinates,
                'latitude': self.coordinates.get('lat', 0.0),
                'longitude': self.coordinates.get('lng', 0.0),
                'accident_id': accident_id,
                'vehicles_involved': vehicles_involved,  # Keep for CSV but will be removed from JSON
                'snapshot_paths': snapshot_paths
            }
            
            # Update CSV log
            self._update_csv_log(accident_record)
            
            # Update JSON log
            self._update_json_log(accident_record)
            
            print(f"[{self.stream_id}] 🚨 ACCIDENT ALERT 🚨 Recorded: {accident_id}")
            
            return accident_id, snapshot_paths
    
    def _save_accident_snapshot(self, frame, accident_id, vehicles_involved):
        """Save a single snapshot of the accident (already cropped)"""
        try:
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime('%H%M%S')
            
            # Dictionary to hold snapshot path
            snapshot_paths = {}
            
            # Save the closeup snapshot (frame is already the closeup)
            closeup_path = self.snapshots_dir / f"accident_{accident_id}_{timestamp_str}.jpg"
            cv2.imwrite(str(closeup_path), frame)
            snapshot_paths['closeup'] = str(closeup_path.relative_to(self.base_dir))
            
            return snapshot_paths
            
        except Exception as e:
            print(f"[{self.stream_id}] Error saving accident snapshot: {str(e)}")
            return {}
    
    def _update_csv_log(self, accident_record):
        """Update the centralized CSV log with new accident"""
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    accident_record['date'],
                    accident_record['time'],
                    accident_record['camera_id'],
                    accident_record['camera_location'],
                    accident_record['latitude'],
                    accident_record['longitude'],
                    accident_record['accident_id'],
                    '+'.join(map(str, accident_record['vehicles_involved']))
                ])
        except Exception as e:
            print(f"[{self.stream_id}] Error updating accident CSV log: {str(e)}")
    
    def _update_json_log(self, accident_record):
        """Update the centralized JSON log with new accident and database"""
        try:
            # Read existing data
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            # Create a copy of the record without vehicles_involved and accident_class for JSON output
            sanitized_record = accident_record.copy()
            if 'vehicles_involved' in sanitized_record:
                del sanitized_record['vehicles_involved']  # Remove vehicles_involved from JSON
            if 'accident_class' in sanitized_record:
                del sanitized_record['accident_class']  # Remove accident_class from JSON
            
            # Add the sanitized record
            data.append(sanitized_record)
            
            # Write updated data
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # # DATABASE INTEGRATION: Save to Django database if available
            # try:
            #     # Check if Django is set up
            #     import django
            #     if django.conf.settings.configured:
            #         from django.contrib.gis.geos import Point
            #         from traffic_app.models import Accident
                    
            #         # Check if accident already exists
            #         accident_id = accident_record.get('accident_id')
            #         if not Accident.objects.filter(accident_id=accident_id).exists():
            #             # Create Django model instance
            #             accident = Accident(
            #                 accident_id=accident_id,
            #                 date=accident_record.get('date'),
            #                 time=accident_record.get('time'),
            #                 camera_id=accident_record.get('camera_id'),
            #                 camera_location=accident_record.get('camera_location'),
            #                 snapshot_path=accident_record.get('snapshot_paths', {}).get('full', '')
            #             )
                        
            #             # Add coordinates if available
            #             if 'coordinates' in accident_record and accident_record['coordinates']:
            #                 lat = accident_record['coordinates'].get('lat', 0)
            #                 lng = accident_record['coordinates'].get('lng', 0)
            #                 accident.coordinates = Point(lng, lat, srid=4326)
                        
            #             # Save to database
            #             accident.save()
            #             print(f"[{self.stream_id}] Added accident {accident_id} to SQL Server database")
            # except (ImportError, AttributeError):
            #     # Django not available or not configured, skip database integration
            #     pass
            
        except Exception as e:
            print(f"[{self.stream_id}] Error updating accident JSON log: {str(e)}")
    
    def set_camera_coordinates(self, latitude, longitude):
        """Update the camera coordinates for mapping"""
        self.coordinates = {"lat": float(latitude), "lng": float(longitude)}
        print(f"[{self.stream_id}] Camera coordinates updated: {self.coordinates}")
        return self.coordinates
