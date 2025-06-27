import cv2
import time
import os
import datetime
from areas import AreaType
from pathlib import Path

class ParkingDetector:
    
    def __init__(self, stream_id="default", violation_time_limit=15, violation_manager=None):
       
        self.stream_id = stream_id
        self.violation_time_limit = violation_time_limit
        self.parked_vehicles = {}  # {track_id: {'start_time': timestamp, 'violation': bool, 'location': (x,y)}}
        self.violation_manager = violation_manager
        
        # Don't create log file or output directory
        self.log_file = None
        self.output_dir = None
        
        print(f"[{self.stream_id}] Parking detector initialized with {violation_time_limit} second limit")
    
    def _setup_output_directory(self):
        """Set output directory reference but don't create it"""
        # Only create directory if using violation_manager
        if self.violation_manager:
            self.output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "parking_violations"
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None
    
    def _initialize_log_file(self):
        """Don't create log file, just return reference"""
        # No file creation, just return a path reference
        log_file_path = f"Parking Model/parking_violations_{self.stream_id}.txt"
        return log_file_path
    
    def update_vehicles(self, frame, tracked_objects, area_manager):
       
        current_time = time.time()
        
        # Track vehicles currently in parking areas
        current_parked_ids = set()
        
        # First check all tracked objects
        for track_id, track_info in tracked_objects.items():
            if len(track_info['center_points']) > 0:
                # Get the latest center point
                center_x, center_y = track_info['center_points'][-1]
                
                # Check if the vehicle is in a parking area
                if area_manager.is_in_area(center_x, center_y, AreaType.PARKING):
                    # Add to the current parked vehicles
                    current_parked_ids.add(track_id)
                    
                    # If this is a new vehicle in the parking area, add it to our tracking dict
                    if track_id not in self.parked_vehicles:
                        self.parked_vehicles[track_id] = {
                            'start_time': current_time,
                            'violation': False,
                            'location': (center_x, center_y),
                            'box': track_info['box'],
                            'snapshot_taken': False  # Flag to track if a snapshot was taken
                        }
                    else:
                        # Update location if it's an existing vehicle
                        self.parked_vehicles[track_id]['location'] = (center_x, center_y)
                        self.parked_vehicles[track_id]['box'] = track_info['box']
                
        # Now process all vehicles we're tracking for parking violations
        vehicles_to_remove = []
        
        for track_id, vehicle_data in self.parked_vehicles.items():
            # If vehicle is no longer in a parking area or no longer tracked, mark for removal
            if track_id not in current_parked_ids:
                vehicles_to_remove.append(track_id)
                continue
            
            # Calculate how long the vehicle has been parked
            parked_duration = current_time - vehicle_data['start_time']
            
            # Check for violation
            if parked_duration >= self.violation_time_limit and not vehicle_data['violation']:
                # Mark as violation and log it
                vehicle_data['violation'] = True
                self._log_violation(track_id, vehicle_data['location'])
                
                # Take a snapshot of the violation if it hasn't been taken yet
                if not vehicle_data.get('snapshot_taken', False) and 'box' in vehicle_data:
                    self._save_violation_snapshot(frame, vehicle_data['box'], track_id)
                    vehicle_data['snapshot_taken'] = True
            
            # Draw information on the frame
            self._draw_parking_info(frame, track_id, vehicle_data, parked_duration)
        
        # Remove vehicles that are no longer in parking areas
        for track_id in vehicles_to_remove:
            del self.parked_vehicles[track_id]
        
        return frame
    
    def _draw_parking_info(self, frame, track_id, vehicle_data, parked_duration):
        """Draw minimal parking information on the frame"""
        x1, y1, x2, y2 = vehicle_data['box']
        center_x, center_y = vehicle_data['location']
        
        # Format duration as seconds
        duration_text = f"{int(parked_duration)}s"
        
        # Choose color based on violation status
        if vehicle_data['violation']:
            color = (0, 0, 255)  # Red for violation
            text = f"{duration_text}"  # Only show duration
        else:
            # Gradient from green to yellow to red as time increases
            ratio = min(1.0, parked_duration / self.violation_time_limit)
            b = 0
            g = int(255 * (1 - ratio))
            r = int(255 * ratio)
            color = (b, g, r)
            text = f"{duration_text}"  # Only show duration
        
        # Draw box around vehicle with appropriate color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # Thin line (1px)
        
        # Add simple time text above the bounding box
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # If violation, just show a small indicator
        if vehicle_data['violation']:
            cv2.circle(frame, (x2-10, y1+10), 5, (0, 0, 255), -1)  # Red dot indicator
    
        return frame
    
    def _save_violation_snapshot(self, frame, bbox, vehicle_id):
        """Process violation but save only when using violation_manager"""
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                violation_id, snapshot_path = self.violation_manager.record_parking_violation(
                    frame, vehicle_id, bbox, duration=self.violation_time_limit
                )
                return True
                
            # Otherwise, just log the event without saving files
            print(f"[{self.stream_id}] Parking violation detected for vehicle {vehicle_id} (local saving disabled)")
            return True
            
        except Exception as e:
            print(f"[{self.stream_id}] Error processing parking violation: {str(e)}")
            return False
    
    def _log_violation(self, track_id, location):
        """Log a parking violation but don't write to file"""
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                # Violation will be logged centrally through the violation manager
                return True
            
            # Just print to console without file logging
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{self.stream_id}] Parking violation detected for vehicle {track_id} at {timestamp}")
            return True
        except Exception as e:
            print(f"[{self.stream_id}] Error processing parking violation: {str(e)}")
            return False
    
    def set_violation_time_limit(self, seconds):
        """Update the time limit for parking violations"""
        self.violation_time_limit = seconds
        print(f"[{self.stream_id}] Parking violation time limit updated to {seconds} seconds")
        return True
