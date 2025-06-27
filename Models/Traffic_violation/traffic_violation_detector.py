import cv2
import numpy as np
import os
import time
import uuid
from datetime import datetime
import sys

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from masking import TrafficLightDetector

class TrafficViolationDetector:
    def __init__(self, stream_id="default", violation_manager=None):
        self.stream_id = stream_id
        self.traffic_light_detector = TrafficLightDetector()
        self.detection_enabled = True
        self.violated_ids = []
        self.violation_timeout = 100  # frames before allowing a new violation for the same ID
        self.violation_counters = {}  # track_id -> counter
        self.is_red_light = False
        
        # Use unified violation manager if available
        self.violation_manager = violation_manager
        
        # Avoid creating directories when not using violation_manager
        self.output_dir = None
    
    def setup_output_directory(self, base_dir=None):
        """Setup directory reference but don't create it unless using violation_manager"""
        # Don't create directories unless explicitly using the violation_manager
        if self.violation_manager:
            # If base_dir is None, use the Traffic_violation directory
            if base_dir is None:
                # Get the directory where this script is located (Traffic_violation folder)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                base_dir = os.path.join(current_dir, 'violations')
            
            today_date = datetime.now().strftime('%Y-%m-%d')
            self.output_dir = os.path.join(base_dir, today_date)
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[{self.stream_id}] Violation images will be saved to: {self.output_dir}")
        else:
            self.output_dir = None
            print(f"[{self.stream_id}] Local storage of violation images is disabled")
            
        return self.output_dir
    
    def process_frame(self, frame, tracked_objects, area_manager):
        """
        Process the current frame for traffic violations
        
        Args:
            frame: The video frame to process
            tracked_objects: Dictionary of tracked objects with their positions/history
            area_manager: AreaManager instance for accessing ROI information
        
        Returns:
            Processed frame with violation annotations
        """
        if not self.detection_enabled or frame is None:
            return frame
        
        # Store tracked_objects reference for use in save_violation_evidence
        self.tracked_objects = tracked_objects
        
        result_frame = frame.copy()
        
        # Check if we have traffic sign areas defined
        has_traffic_signs = False
        if hasattr(area_manager, 'areas'):
            from areas import AreaType
            traffic_sign_areas = area_manager.areas.get(AreaType.TRAFFIC_SIGN, [])
            has_traffic_signs = len(traffic_sign_areas) > 0
        
        # Only process traffic lights if we have traffic sign areas defined
        traffic_light_state = None
        if has_traffic_signs:
            # Extract traffic sign ROIs and process them
            for area in traffic_sign_areas:
                points = area.get('points', [])
                if len(points) < 3:  # Need at least 3 points for a valid polygon
                    continue
                
                # Create a mask for the traffic sign area
                mask = np.zeros_like(frame)
                cv2.fillPoly(mask, [np.array(points, np.int32)], (255, 255, 255))
                masked_frame = cv2.bitwise_and(frame, mask)
                
                # Calculate ROI boundaries from polygon points
                x_coords = [pt[0] for pt in points]
                y_coords = [pt[1] for pt in points]
                x_min, x_max = max(0, min(x_coords)), min(frame.shape[1], max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(frame.shape[0], max(y_coords))
                
                # Extract traffic light ROI if we have valid boundaries
                if x_min < x_max and y_min < y_max:
                    traffic_light_roi = masked_frame[y_min:y_max, x_min:x_max]
                    
                    # Process the traffic light ROI
                    _, traffic_light_state = self.traffic_light_detector.process_frame(traffic_light_roi)
                    
                    # Draw the ROI state on the frame
                    if traffic_light_state:
                        color = (0, 255, 0) if traffic_light_state == "GREEN" else (0, 0, 255)
                        cv2.putText(result_frame, f"Traffic Light: {traffic_light_state}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        break  # Stop after first valid detection
        
        # Update the red light flag
        self.is_red_light = (traffic_light_state == "RED")
        
        # Only check for violations if we have a red light
        if self.is_red_light:
            # Check for traffic line violations
            from areas import AreaType
            traffic_lines = area_manager.areas.get(AreaType.TRAFFIC_LINE, [])
            
            # Check each vehicle against traffic lines
            for track_id, track_info in tracked_objects.items():
                # Skip if not enough history points
                if len(track_info.get('center_points', [])) < 2:
                    continue
                
                # Get current and previous positions
                prev_point = track_info['center_points'][-2]
                curr_point = track_info['center_points'][-1]
                
                # Check if crossing any traffic line
                crossing, line_id, direction = area_manager.is_crossing_line(
                    prev_point, curr_point, AreaType.TRAFFIC_LINE)
                
                if crossing:
                    # Check if this is a new violation
                    if track_id not in self.violation_counters or self.violation_counters[track_id] <= 0:
                        self.violation_counters[track_id] = self.violation_timeout
                        
                        # Add to violated IDs
                        if track_id not in self.violated_ids:
                            self.violated_ids.append(track_id)
                            
                        # Save violation evidence
                        self.save_violation_evidence(result_frame, track_id)
                        
                        # Draw violation with thinner box
                        if 'box' in track_info:
                            x1, y1, x2, y2 = track_info['box']
                            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Reduced thickness to 1
                            cv2.putText(result_frame, "RED LIGHT VIOLATION", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
                        print(f"[{self.stream_id}] Red light violation detected for vehicle {track_id} at line {line_id}")
            
            # Update violation counters
            for track_id in list(self.violation_counters.keys()):
                if self.violation_counters[track_id] > 0:
                    self.violation_counters[track_id] -= 1
        
        # Display traffic light status
        light_status = "RED" if self.is_red_light else "GREEN" if traffic_light_state == "GREEN" else "UNKNOWN"
        color = (0, 0, 255) if light_status == "RED" else (0, 255, 0) if light_status == "GREEN" else (255, 255, 255)
        cv2.putText(result_frame, f"Traffic Light: {light_status}", 
                  (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        
        return result_frame
    
    def save_violation_evidence(self, frame, vehicle_id):
        """
        Process violation evidence but save only when using violation_manager
        
        Args:
            frame: The full video frame
            vehicle_id: ID of the violating vehicle
    
        Returns:
            Path to the saved image file or a placeholder string
        """
        # Use unified violation manager if available
        if self.violation_manager and vehicle_id in self.tracked_objects:
            vehicle_box = None
            for track_id, track_info in self.tracked_objects.items():
                if track_id == vehicle_id and 'box' in track_info:
                    vehicle_box = track_info['box']
                    break
                
            if vehicle_box:
                violation_id, snapshot_path = self.violation_manager.record_traffic_violation(
                    frame, vehicle_id, vehicle_box, light_state="RED"
                )
                return snapshot_path
    
        # Just return a placeholder path if not using violation_manager
        timestamp = datetime.now().strftime('%H-%M-%S')
        return f"disabled_local_storage_{vehicle_id}_{timestamp}"
    
    def toggle_detection(self):
        """Toggle traffic violation detection on/off"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        print(f"[{self.stream_id}] Traffic violation detection {status}")
        return self.detection_enabled
    
    def reset_violations(self):
        """Reset all violation counters"""
        self.violated_ids = []
        self.violation_counters = {}
