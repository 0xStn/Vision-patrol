# Vision Patrol - Helmet Violation Detection Module
import cv2
import numpy as np
import os
from datetime import datetime
import csv
from ultralytics import YOLO
import cvzone
import traceback

class HelmetViolationDetector:
    def __init__(self, stream_id="default", violation_manager=None):
        self.stream_id = stream_id
        self.detection_enabled = False  # Default to disabled
        self.violated_ids = []
        self.violation_timeout = 100  # Frames before allowing new violation for same ID
        self.violation_counters = {}  # Track violation counter per ID
        self.tracked_vehicles = {}  # Track motorcycles/bicycles in detection area
        
        # Violation tracking to prevent duplicates
        self.processed_violations = set()  # IDs already logged
        self.permanent_violations = set()  # IDs permanently flagged
        self.violation_timestamps = {}     # Violation occurrence timestamps
        self.last_inference_time = {}      # Last inference time per vehicle
        self.min_inference_interval = 1.0  # Minimum seconds between inferences
        
        # Unified violation manager integration
        self.violation_manager = violation_manager
        
        # Load helmet detection model
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "helmet_ds.pt")
        # Make sure model defaults to None if loading fails
        try:
            self.model = YOLO(self.model_path)
            print(f"[{self.stream_id}] Helmet detector model loaded from {self.model_path}")
            print(f"[{self.stream_id}] Helmet violation detection is DISABLED by default")
        except Exception as e:
            print(f"[{self.stream_id}] Error loading helmet detection model: {str(e)}")
            self.model = None
            
        # Class names from model (may vary based on your model)
        self.helmet_class_names = {
            0: 'with-helmet',
            1: 'no-helmet'
        }
        
        # Don't set up local directories
        self.output_dir = None
        self.log_file = None
    
    def setup_output_directory(self, base_dir=None):
        """Set directory reference but don't create it"""
        # Only create directory if using violation_manager
        if self.violation_manager:
            if base_dir is None:
                # Get the directory where this script is located
                current_dir = os.path.dirname(os.path.abspath(__file__))
                base_dir = os.path.join(current_dir, 'violations')
            
            today_date = datetime.now().strftime('%Y-%m-%d')
            self.output_dir = os.path.join(base_dir, today_date)
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[{self.stream_id}] Helmet violation images will be saved to: {self.output_dir}")
        else:
            self.output_dir = None
            print(f"[{self.stream_id}] Local storage of helmet violations is disabled")
        
        return self.output_dir
    
    def setup_log_file(self):
        """Set log file reference but don't create the file"""
        # Only set up log file if using violation_manager
        if self.violation_manager:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(current_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Create a log file name with date
            today_date = datetime.now().strftime('%Y-%m-%d')
            self.log_file = os.path.join(logs_dir, f'helmet_violations_{today_date}.csv')
            
            # Create log file with headers if it doesn't exist
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Date', 
                        'Time', 
                        'Violation Type', 
                        'Camera ID', 
                        'Vehicle ID', 
                        'Vehicle Type',
                        'Image Path'
                    ])
            
            print(f"[{self.stream_id}] Helmet violations will be logged to: {self.log_file}")
        else:
            self.log_file = None
            print(f"[{self.stream_id}] Local logging of helmet violations is disabled")
            
        return self.log_file
    
    def log_violation(self, violation_type, vehicle_id, image_path, vehicle_type="motorcycle"):
        """Log a violation - use unified manager if available"""
        # Skip if already permanently processed or in current session
        if vehicle_id in self.permanent_violations or vehicle_id in self.processed_violations:
            print(f"[{self.stream_id}] Skipping duplicate helmet violation for vehicle {vehicle_id} (already processed)")
            return False
            
        # Record current violation time
        current_time = datetime.now().timestamp()
        self.violation_timestamps[vehicle_id] = current_time
            
        # Skip local logging if using unified manager
        if self.violation_manager:
            # Mark as processed and permanent
            self.processed_violations.add(vehicle_id)
            self.permanent_violations.add(vehicle_id)  # Add to permanent violations
            print(f"[{self.stream_id}] Vehicle {vehicle_id} added to permanent violations list")
            return True
            
        # Otherwise, use local file for snapshots only - the CSV part is removed
        # Mark as processed and permanent
        self.processed_violations.add(vehicle_id)
        self.permanent_violations.add(vehicle_id)
        print(f"[{self.stream_id}] Vehicle {vehicle_id} added to permanent violations list")
        return True
    
    def process_frame(self, frame, tracked_objects, area_manager):
        """Process the current frame for helmet violations"""
        # Early return if conditions aren't met
        if not self.detection_enabled or frame is None or self.model is None:
            return frame
        
        try:
            # Import here to avoid circular imports
            from areas import AreaType
            
            # Create a defensive copy of frame to prevent issues
            result_frame = frame.copy() if frame is not None else None
            if result_frame is None:
                return frame
                
            current_time = datetime.now().timestamp()
            
            # Only process valid tracked objects dictionary
            if not isinstance(tracked_objects, dict):
                print(f"[{self.stream_id}] Warning: Invalid tracked_objects (not a dictionary)")
                return result_frame
                
            # Filter for motorcycles and bicycles (class IDs 3 and 4 in COCO dataset)
            motorcycle_ids = []
            
            # First pass - update tracked vehicles and identify new ones in detection area
            for track_id, track_info in tracked_objects.items():
                # Skip permanently processed motorcycles/bicycles immediately
                if track_id in self.permanent_violations:
                    continue
                    
                if 'class_id' in track_info:
                    class_id = track_info['class_id']
                    if class_id in [3, 4]:  # motorcycle or bicycle
                        # Check if in detection area
                        if 'center_points' in track_info and track_info['center_points']:
                            center_x, center_y = track_info['center_points'][-1]
                            if area_manager.is_in_area(center_x, center_y, AreaType.DETECTION):
                                motorcycle_ids.append(track_id)
                                
                                # Store/update in tracked vehicles for continuous monitoring
                                if track_id not in self.tracked_vehicles:
                                    self.tracked_vehicles[track_id] = {
                                        'first_seen': datetime.now(),
                                        'helmet_detected': False,
                                        'violation_logged': False,
                                        'box': track_info.get('box', None),
                                        'type': 'motorcycle' if class_id == 3 else 'bicycle',
                                        'in_area': True,
                                        'last_seen': datetime.now()
                                    }
                                else:
                                    self.tracked_vehicles[track_id]['last_seen'] = datetime.now()
                                    self.tracked_vehicles[track_id]['in_area'] = True
                                    self.tracked_vehicles[track_id]['box'] = track_info.get('box', None)
            
            # Update vehicles that are not in the area anymore
            for track_id in list(self.tracked_vehicles.keys()):
                # Skip permanently processed vehicles
                if track_id in self.permanent_violations:
                    continue
                    
                if track_id not in motorcycle_ids:
                    # Vehicle is not in the detection area
                    self.tracked_vehicles[track_id]['in_area'] = False
                    
                    # If vehicle was in area and now left without helmet detection
                    # AND violation hasn't been logged yet AND not already in processed violations
                    if (not self.tracked_vehicles[track_id]['helmet_detected'] and 
                        not self.tracked_vehicles[track_id]['violation_logged'] and
                        track_id not in self.processed_violations):
                        
                        # Handle violation - vehicle left area without helmet
                        if track_id in tracked_objects and 'box' in tracked_objects[track_id]:
                            try:
                                # Get the most recent box coordinates
                                x1, y1, x2, y2 = tracked_objects[track_id]['box']
                                
                                # Save a snapshot of the violation - using enlarged bounding box
                                image_path = self.save_violation_evidence(result_frame, track_id, tracked_objects)
                                
                                # Log the violation - internally handles marking as processed
                                if self.log_violation("NO_HELMET", track_id, image_path, self.tracked_vehicles[track_id]['type']):
                                    # Mark as logged to avoid duplicate logs
                                    self.tracked_vehicles[track_id]['violation_logged'] = True
                                    
                                    # Add to violated IDs if not already there
                                    if track_id not in self.violated_ids:
                                        self.violated_ids.append(track_id)
                                    
                                    # Draw violation box on frame
                                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cvzone.putTextRect(result_frame, f"HELMET VIOLATION", (x1, y1 - 10), 
                                                     scale=1, thickness=2, colorR=(0, 0, 255))
                                    
                                    print(f"[{self.stream_id}] No helmet violation detected for vehicle {track_id}")
                            except Exception as e:
                                print(f"[{self.stream_id}] Error processing helmet violation: {str(e)}")
                                traceback.print_exc()
            
            # Run helmet detection on all motorcycles in detection area
            for track_id in motorcycle_ids:
                # Skip permanently processed vehicles
                if track_id in self.permanent_violations:
                    continue
                    
                # Skip frequent inference on the same vehicle (throttle to once per second)
                last_infer_time = self.last_inference_time.get(track_id, 0)
                if current_time - last_infer_time < self.min_inference_interval:
                    continue
                    
                self.last_inference_time[track_id] = current_time
                
                try:
                    x1, y1, x2, y2 = tracked_objects[track_id]['box']
                    
                    # Extract region of interest for helmet detection
                    # Use a larger ROI to capture both motorcycle and rider
                    if x1 < x2 and y1 < y2:  # Ensure valid box
                        # Add more padding to the top of the bounding box for better helmet detection
                        head_roi_y1 = max(0, y1 - int((y2-y1) * 0.5))  # 50% padding on top
                        # Also ensure we capture enough width in case the rider is leaning
                        head_roi_x1 = max(0, x1 - int((x2-x1) * 0.1))  # 10% padding on sides
                        head_roi_x2 = min(result_frame.shape[1], x2 + int((x2-x1) * 0.1))
                        
                        # Verify ROI bounds are valid
                        if head_roi_y1 >= result_frame.shape[0] or head_roi_x1 >= result_frame.shape[1]:
                            continue
                            
                        head_roi = result_frame[head_roi_y1:y2, head_roi_x1:head_roi_x2]
                        
                        if head_roi.size > 0 and head_roi.shape[0] > 0 and head_roi.shape[1] > 0:  # More thorough check
                            # Run helmet detection on the ROI
                            helmet_results = self.model(head_roi)
                            
                            # Check for helmet detection results
                            if len(helmet_results) > 0 and hasattr(helmet_results[0], 'boxes') and helmet_results[0].boxes is not None:
                                if len(helmet_results[0].boxes) > 0:
                                    boxes = helmet_results[0].boxes.xyxy.cpu().numpy()
                                    cls = helmet_results[0].boxes.cls.cpu().numpy()
                                    
                                    # Draw all helmet detections
                                    helmet_detected = False
                                    for i, box in enumerate(boxes):
                                        h_x1, h_y1, h_x2, h_y2 = box.astype(int)
                                        h_cls = int(cls[i])
                                        
                                        # Adjust coordinates to the original frame
                                        h_x1 += head_roi_x1
                                        h_y1 += head_roi_y1
                                        h_x2 += head_roi_x1
                                        h_y2 += head_roi_y1
                                        
                                        # Get class name and color
                                        h_label = self.helmet_class_names.get(h_cls, f"Class {h_cls}")
                                        
                                        if h_cls == 0:  # with-helmet
                                            h_color = (0, 255, 0)  # Green
                                            helmet_detected = True
                                            self.tracked_vehicles[track_id]['helmet_detected'] = True
                                        else:  # no-helmet
                                            h_color = (0, 0, 255)  # Red
                except Exception as e:
                    print(f"[{self.stream_id}] Error in helmet inference for vehicle {track_id}: {str(e)}")
            
            # Reset temporary processed violations on a timeout basis if needed
            # But keep permanent_violations intact
            
            # Remove old tracked vehicles
            current_time = datetime.now()
            for track_id in list(self.tracked_vehicles.keys()):
                time_diff = (current_time - self.tracked_vehicles[track_id]['last_seen']).total_seconds()
                if time_diff > 5:  # Remove if not seen for 5 seconds
                    del self.tracked_vehicles[track_id]
            
            # Display violation stats on frame
            cv2.putText(result_frame, f"Helmet Violations: {len(self.permanent_violations)}", 
                      (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return result_frame
            
        except Exception as e:
            print(f"[{self.stream_id}] Error in helmet detection process_frame: {str(e)}")
            traceback.print_exc()
            return frame
    
    def save_violation_evidence(self, frame, vehicle_id, tracked_objects):
        """Process violation but save only when using violation_manager"""
        try:
            # Use unified violation manager if available
            if self.violation_manager and vehicle_id in tracked_objects and 'box' in tracked_objects[vehicle_id]:
                vehicle_box = tracked_objects[vehicle_id]['box']
                vehicle_type = "motorcycle"
                if 'class_id' in tracked_objects[vehicle_id]:
                    vehicle_type = "bicycle" if tracked_objects[vehicle_id]['class_id'] == 4 else "motorcycle"
                    
                # Expand the box more to capture both motorcycle and rider
                x1, y1, x2, y2 = vehicle_box
                width, height = x2 - x1, y2 - y1
                
                # More padding especially on top to capture rider's head
                expanded_box = (
                    max(0, x1 - int(width * 0.2)),  # left
                    max(0, y1 - int(height * 0.7)),  # top - increased to capture rider
                    min(frame.shape[1], x2 + int(width * 0.2)),  # right
                    min(frame.shape[0], y2)  # bottom stays the same
                )
                
                try:
                    violation_id, snapshot_paths = self.violation_manager.record_helmet_violation(
                        frame.copy(),  # Use a copy to prevent modifications
                        vehicle_id, 
                        expanded_box, 
                        vehicle_type=vehicle_type
                    )
                    
                    # Improved handling of different return types
                    if snapshot_paths is None:
                        return f"helmet_{self.stream_id}_{violation_id}_none"
                    elif isinstance(snapshot_paths, dict) and 'full' in snapshot_paths:
                        return snapshot_paths['full']
                    elif isinstance(snapshot_paths, str):
                        return snapshot_paths
                    else:
                        return f"helmet_{self.stream_id}_{violation_id}"
                    
                except Exception as e:
                    print(f"[{self.stream_id}] Error in violation manager: {str(e)}")
                    traceback.print_exc()
                    # Return a placeholder path
                    return f"error_helmet_{vehicle_id}_{datetime.now().strftime('%H%M%S')}"
            
            # Just return a placeholder path if not using violation_manager
            return f"disabled_local_storage_{vehicle_id}_{datetime.now().strftime('%H%M%S')}"
            
        except Exception as e:
            print(f"[{self.stream_id}] Error processing helmet violation evidence: {str(e)}")
            traceback.print_exc()
            return f"error_saving_violation_{vehicle_id}_{datetime.now().strftime('%H%M%S')}"
    
    def toggle_detection(self):
        """Toggle helmet violation detection on/off"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        print(f"[{self.stream_id}] Helmet violation detection {status}")
        
        # If detection was just disabled, clear all tracking data
        if not self.detection_enabled:
            self.tracked_vehicles = {}
            self.last_inference_time = {}  # Clear inference times
            # Don't clear permanent violations when disabling - 
            # they should persist until manually reset
            print(f"[{self.stream_id}] Cleared vehicle tracking data")
            
        return self.detection_enabled
    
    def reset_violations(self):
        """Reset all violation counters and tracking data"""
        self.violated_ids = []
        self.tracked_vehicles = {}
        self.processed_violations = set()  # Clear processed violations
        self.permanent_violations = set()  # Clear permanent violations
        self.violation_timestamps = {}     # Clear violation timestamps
        self.last_inference_time = {}      # Clear inference times
        print(f"[{self.stream_id}] Reset all helmet violation records and tracking data")
