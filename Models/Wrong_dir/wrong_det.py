import cv2
import numpy as np
import os
import logging
import time
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

# Import the AreaType with proper error handling
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
try:
    from areas import AreaType
except ImportError:
    # Define a backup AreaType if import fails
    class AreaType(Enum):
        DETECTION = 1
        SPEED = 2
        WRONG_DIRECTION = 3
        PARKING = 4
        TRAFFIC_LINE = 5
        TRAFFIC_SIGN = 6
        CUSTOM = 7

class Direction(Enum):
    """Enum for vehicle direction relative to a line pair"""
    FORWARD = 1    # Vehicle moving in allowed direction
    BACKWARD = -1  # Vehicle moving in wrong direction
    UNKNOWN = 0    # Direction not yet determined

class WrongDirectionDetector:
    """
    Detector for vehicles moving in the wrong direction through defined line pairs.
    Uses AreaManager's line definitions to create virtual gates that detect wrong-way traffic.
    """
    def __init__(self, stream_id="default", conf_threshold=0.0, violation_manager=None):
        # Basic configuration
        self.stream_id = stream_id
        self.conf_threshold = conf_threshold
        self.detection_enabled = False
        self.violation_manager = violation_manager
        
        # Disable logging setup
        self.logger = self._setup_dummy_logger()
        
        # State tracking
        self.vehicle_direction_status = {}  # {vehicle_id: {line_pair_id: direction}}
        self.line_pairs = {}                # {pair_id: {entry_line, exit_line, etc.}}
        self.wrong_way_vehicles = set()     # Set of vehicle IDs going wrong way
        self.snapshots_counter = 0
        
        self.logger.info(f"Wrong direction detector initialized")
        self.logger.info(f"Detection will auto-enable when line pairs are configured")
    
    def _setup_dummy_logger(self):
        """Set up a dummy logger that doesn't write to file"""
        logger = logging.getLogger(f"wrong_direction_{self.stream_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Add a NullHandler which does nothing
            logger.addHandler(logging.NullHandler())
        
        return logger
    
    def configure_line_pairs(self, area_manager):
        """Configure line pairs based on lines defined in the AreaManager"""
        try:
            self.line_pairs = {}
            
            if not self._validate_area_manager(area_manager):
                return False
            
            # Get area type to use
            area_type_to_use = self._get_area_type(area_manager)
            
            # Get wrong direction lines
            wrong_dir_lines = self._get_wrong_direction_lines(area_manager, area_type_to_use)
            if not wrong_dir_lines:
                return False
            
            # Create pairs of adjacent lines
            success = self._create_line_pairs(wrong_dir_lines)
            
            # Always enable detection if pairs were successfully created
            if success:
                self.detection_enabled = True
                self.logger.info(f"Wrong direction detection automatically ENABLED with {len(self.line_pairs)} line pairs")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error configuring line pairs: {str(e)}")
            return False
    
    def _validate_area_manager(self, area_manager):
        """Validate the area manager is properly initialized"""
        if area_manager is None or not hasattr(area_manager, 'areas'):
            self.logger.error("Invalid area manager")
            return False
        return True
    
    def _get_area_type(self, area_manager):
        """Get the correct area type from area manager or use imported one"""
        if hasattr(area_manager, 'AreaType') and hasattr(area_manager.AreaType, 'WRONG_DIRECTION'):
            return area_manager.AreaType.WRONG_DIRECTION
        return AreaType.WRONG_DIRECTION
    
    def _get_wrong_direction_lines(self, area_manager, area_type):
        """Get wrong direction lines from area manager"""
        # Create empty list for area type if it doesn't exist
        if area_type not in area_manager.areas:
            area_manager.areas[area_type] = []
            return None
        
        wrong_dir_lines = area_manager.areas[area_type]
        if len(wrong_dir_lines) < 2 or not isinstance(wrong_dir_lines, list):
            return None
        
        return wrong_dir_lines
    
    def _create_line_pairs(self, wrong_dir_lines):
        """Create pairs of entry and exit lines"""
        pair_id = 0
        for i in range(0, len(wrong_dir_lines) - 1, 2):
            if i + 1 >= len(wrong_dir_lines):
                continue
                
            entry_line = wrong_dir_lines[i]
            exit_line = wrong_dir_lines[i + 1]
            
            entry_points = entry_line.get('points', [])
            exit_points = exit_line.get('points', [])
            
            if len(entry_points) == 2 and len(exit_points) == 2:
                self.line_pairs[pair_id] = {
                    'entry_line_id': i,
                    'exit_line_id': i + 1,
                    'entry_line': entry_points,
                    'exit_line': exit_points,
                    'allowed_direction': Direction.FORWARD
                }
                pair_id += 1
        
        return len(self.line_pairs) > 0
    
    def process_frame(self, frame, tracked_objects, area_manager):
        """Process a frame to detect vehicles moving in wrong direction"""
        if frame is None:
            return None
        
        # Show detection status
        processed_frame = self._display_status(frame)
        
        if not self.detection_enabled:
            return processed_frame
        
        # Configure line pairs if needed
        if not self.line_pairs:
            success = self.configure_line_pairs(area_manager)
            if not success:
                return processed_frame
        
        # Draw line pairs
        self._draw_line_pairs(processed_frame)
        
        # Validate tracked_objects
        if not self._validate_tracked_objects(tracked_objects):
            return processed_frame
        
        # Process each vehicle for wrong direction
        self._process_vehicles(processed_frame, tracked_objects)
        
        # Display count of wrong way vehicles
        self._display_wrong_way_count(processed_frame)
        
        return processed_frame
    
    def _display_status(self, frame):
        """Display detection status on frame"""
        processed_frame = frame.copy()
        status_text = "WRONG DIRECTION: " + ("ENABLED" if self.detection_enabled else "DISABLED")
        color = (0, 255, 0) if self.detection_enabled else (0, 0, 255)
        cv2.putText(processed_frame, status_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return processed_frame
    
    def _draw_line_pairs(self, frame):
        """Draw all line pairs on the frame"""
        for _, pair_info in self.line_pairs.items():
            # Draw entry line (red)
            entry_line = pair_info['entry_line']
            cv2.line(frame, entry_line[0], entry_line[1], (0, 0, 255), 1)
            
            # Draw exit line (orange)
            exit_line = pair_info['exit_line']
            cv2.line(frame, exit_line[0], exit_line[1], (0, 165, 255), 1)
            
            # Draw small dots at line endpoints
            for point in entry_line + exit_line:
                cv2.circle(frame, point, 2, (255, 255, 255), -1)
    
    def _validate_tracked_objects(self, tracked_objects):
        """Validate that tracked_objects is a valid dictionary"""
        return tracked_objects and isinstance(tracked_objects, dict)
    
    def _process_vehicles(self, frame, tracked_objects):
        """Process each vehicle to detect wrong direction movement"""
        for vehicle_id, vehicle_info in tracked_objects.items():
            if not self._has_sufficient_tracking_points(vehicle_info):
                continue
            
            # Get current and previous position
            current_pos = vehicle_info['center_points'][-1]
            prev_pos = vehicle_info['center_points'][-2]
            
            # Initialize vehicle direction status if needed
            if vehicle_id not in self.vehicle_direction_status:
                self.vehicle_direction_status[vehicle_id] = {}
            
            # Check each line pair for crossings
            for pair_id, pair_info in self.line_pairs.items():
                # Skip if already processed for this pair
                if self._is_already_processed(vehicle_id, pair_id, frame, vehicle_info):
                    continue
                
                # Check for crossing and handle direction
                self._check_line_crossing(frame, vehicle_id, pair_id, pair_info, prev_pos, current_pos, vehicle_info)
    
    def _has_sufficient_tracking_points(self, vehicle_info):
        """Check if vehicle has enough tracking points for analysis"""
        return 'center_points' in vehicle_info and len(vehicle_info['center_points']) >= 2
    
    def _is_already_processed(self, vehicle_id, pair_id, frame, vehicle_info):
        """Check if vehicle has already been processed for this line pair"""
        if pair_id in self.vehicle_direction_status[vehicle_id]:
            # Mark wrong-way vehicles that have already been identified
            # Only highlight for a limited time (1 second instead of 5 seconds)
            if (self.vehicle_direction_status[vehicle_id][pair_id] == Direction.BACKWARD and 
                vehicle_id in self.wrong_way_vehicles and 
                'bbox' in vehicle_info and 
                'detection_time' in vehicle_info and
                time.time() - vehicle_info.get('detection_time', 0) < 1.0):  # Changed from 5 to 1 second
                self._highlight_wrong_way_vehicle(frame, vehicle_info['bbox'])
            return True
        return False
    
    def _highlight_wrong_way_vehicle(self, frame, bbox):
        """Highlight a vehicle going the wrong way with thinner box"""
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)  # Reduced thickness to 1
        cv2.putText(frame, "WRONG WAY", 
                   (int(bbox[0]), int(bbox[1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)  # Smaller font
    
    def _check_line_crossing(self, frame, vehicle_id, pair_id, pair_info, prev_pos, current_pos, vehicle_info):
        """Check if vehicle is crossing entry or exit line and handle appropriately"""
        # Check line crossings
        entry_crossed = self._line_segments_intersect(
            prev_pos, current_pos, 
            pair_info['entry_line'][0], pair_info['entry_line'][1])
        
        exit_crossed = self._line_segments_intersect(
            prev_pos, current_pos,
            pair_info['exit_line'][0], pair_info['exit_line'][1])
        
        # Handle correct direction (entry line first)
        if entry_crossed:
            self.vehicle_direction_status[vehicle_id][pair_id] = Direction.FORWARD
        
        # Handle wrong direction (exit line first)
        elif exit_crossed:
            self.vehicle_direction_status[vehicle_id][pair_id] = Direction.BACKWARD
            
            # Process new violation
            if vehicle_id not in self.wrong_way_vehicles:
                self._handle_new_violation(frame, vehicle_id, pair_id, vehicle_info)
    
    def _handle_new_violation(self, frame, vehicle_id, pair_id, vehicle_info):
        """Handle a newly detected wrong way violation"""
        self.wrong_way_vehicles.add(vehicle_id)
        
        # Add detection timestamp to the vehicle info
        if 'detection_time' not in vehicle_info:
            vehicle_info['detection_time'] = time.time()
        
        # Log the violation
        self._log_wrong_direction(vehicle_id, pair_id)
        
        # Take snapshot if bbox available
        if 'bbox' in vehicle_info:
            self._save_wrong_way_vehicle(frame, vehicle_info['bbox'], vehicle_id, pair_id)
    
    def _display_wrong_way_count(self, frame):
        """Display count of wrong way vehicles on the frame"""
        cv2.putText(frame, f"Wrong Way Vehicles: {len(self.wrong_way_vehicles)}", 
                  (frame.shape[1] - 200, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """Check if line segments (p1,p2) and (p3,p4) intersect"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0: return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or Counterclockwise
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                   q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases
        if o1 == 0 and on_segment(p1, p3, p2): return True
        if o2 == 0 and on_segment(p1, p4, p2): return True
        if o3 == 0 and on_segment(p3, p1, p4): return True
        if o4 == 0 and on_segment(p3, p2, p4): return True
        
        return False
    
    def _log_wrong_direction(self, vehicle_id, pair_id):
        """Log a wrong direction violation to the log file"""
        log_message = f"Wrong direction violation - Vehicle ID: {vehicle_id}, Line pair: {pair_id}"
        self.logger.info(log_message)
    
    def _save_wrong_way_vehicle(self, frame, bbox, vehicle_id, pair_id=None):
        """Save a snapshot of a vehicle going the wrong way with expanded context"""
        try:
            # Use unified violation manager if available
            if self.violation_manager:
                violation_id, snapshot_path = self.violation_manager.record_wrong_direction(
                    frame, vehicle_id, bbox, line_pair=pair_id
                )
                self.snapshots_counter += 1
                self.logger.info(f"Saved wrong-way violation through manager: {violation_id}")
                return True
                
            # Otherwise, just log that snapshot would have been saved
            self.snapshots_counter += 1
            self.logger.info(f"Would save wrong-way vehicle snapshot (local saving disabled)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing wrong-way vehicle: {str(e)}")
            return False
    
    def clear_tracking_data(self):
        """Clear all tracking data (useful when changing scenes)"""
        self.vehicle_direction_status = {}
        self.wrong_way_vehicles = set()
        self.logger.info("Wrong direction tracking data cleared")
    
    def debug_area_manager(self, area_manager):
        """Print diagnostic information about the area manager configuration"""
        if not hasattr(area_manager, 'areas'):
            self.logger.error("Area manager has no 'areas' attribute")
            return
            
        # Log area counts
        area_type_to_use = AreaType
        if hasattr(area_manager, 'AreaType'):
            area_type_to_use = area_manager.AreaType
            
        for area_type in area_type_to_use:
            if area_type in area_manager.areas:
                count = len(area_manager.areas[area_type])
                self.logger.info(f"{area_type.name}: {count} areas defined")
