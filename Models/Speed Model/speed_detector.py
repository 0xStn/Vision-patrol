import cv2
import numpy as np
import time
import os
from datetime import datetime

class SpeedDetector:
    def __init__(self, stream_id="default", pixels_per_meter=30, smoothing_window=10, font_style="plain", violation_manager=None):
       
        self.stream_id = stream_id
        self.pixels_per_meter = pixels_per_meter  # Calibration parameter
        self.smoothing_window = smoothing_window
        self.enabled = False  # Only enable after speed lines are defined
        self.violation_manager = violation_manager
        self.speed_limit = 1.0  # Set speed limit to 1 km/h
        
        # Set font style based on parameter
        self.font_style = font_style
        if font_style.lower() == "simplex":
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.font_scale = 0.6
            self.font_thickness = 2
        else:  # Default to plain
            self.font = cv2.FONT_HERSHEY_PLAIN
            self.font_scale = 1.2
            self.font_thickness = 1
        
        # Vehicle tracking data
        self.vehicles = {}  # Store vehicle position history
        self.speeds = {}    # Store calculated speeds
        self.recorded_speeds = {}  # Store speeds recorded at the measurement line
        
        # Fixed time step for more consistent speed calculation (33.33ms = ~30 FPS)
        self.fixed_time_step = 1.0 / 30.0  # Fixed time step in seconds
        self.last_frame_time = time.time()
        
        # Speed stability parameters
        self.max_speed_change = 5.0  # Maximum allowed change in speed between calculations (km/h)
        self.min_distance_for_calc = 3.0  # Minimum pixel distance to calculate speed (reduces noise)
        
        # Disable directory creation and CSV logging
        self.disable_csv_logging = True
        
        # Keep track of violations to avoid duplicates
        self.speed_violations = set()  # Set of vehicle IDs that have already been recorded for violation
        
        print(f"[{self.stream_id}] Speed detector initialized with calibration: {pixels_per_meter} pixels/meter")
        print(f"[{self.stream_id}] Speed limit set to: {self.speed_limit} km/h")
        
        # Modified output directory handling for snapshots when needed
        self.output_dir = None
        
    def calculate_speed(self, frame, tracked_objects, fps=30):
        """Calculate speed for all tracked vehicles and identify violations"""
        if not self.enabled or not tracked_objects:
            return frame
        
        # Store current frame for violation snapshots
        self.current_frame = frame.copy()  # Make a clean copy without any annotations
        self.tracked_objects = tracked_objects  # Store reference to all tracked objects
            
        # Calculate time difference using fixed step for stable FPS
        current_time = time.time()
        time_diff = self.fixed_time_step  # Always use 1/30 seconds between frames
        self.last_frame_time = current_time
        
        # Track IDs we've seen in this frame
        current_ids = set()
        
        for obj_id, track_data in tracked_objects.items():
            current_ids.add(obj_id)
            
            if 'center_points' not in track_data or len(track_data['center_points']) < 2:
                continue
                
            # Get current position (latest center point)
            current_pos = track_data['center_points'][-1]
            
            # Initialize tracking for new vehicles
            if obj_id not in self.vehicles:
                self.vehicles[obj_id] = {
                    'positions': track_data['center_points'].copy(),  # Copy existing points
                    'timestamps': [current_time - self.fixed_time_step * i for i in range(len(track_data['center_points'])-1, -1, -1)],
                    'speeds': [],     # List of calculated speeds
                    'last_speed': 0,  # Last stable speed
                    'crossed_lines': set(),  # Lines already crossed (copy from tracked_objects)
                }
                if 'crossed_lines' in track_data:
                    self.vehicles[obj_id]['crossed_lines'] = track_data['crossed_lines'].copy()
            else:
                # Update positions and timestamps
                self.vehicles[obj_id]['positions'].append(current_pos)
                self.vehicles[obj_id]['timestamps'].append(current_time)
                
                # Limit history size
                max_history = self.smoothing_window * 2
                if len(self.vehicles[obj_id]['positions']) > max_history:
                    self.vehicles[obj_id]['positions'] = self.vehicles[obj_id]['positions'][-max_history:]
                    self.vehicles[obj_id]['timestamps'] = self.vehicles[obj_id]['timestamps'][-max_history:]
                    self.vehicles[obj_id]['speeds'] = self.vehicles[obj_id]['speeds'][-max_history:]
                
                # Check if vehicle crossed speed lines recently and calculate speed
                speed_line_crossings = [(line_id, 1) for line_id, line_type in track_data.get('crossed_lines', set()) 
                                     if line_type == 'speed' and (line_id, 'speed') not in self.vehicles[obj_id].get('processed_lines', set())]
                
                if speed_line_crossings:
                    for speed_line_id, direction in speed_line_crossings:
                        # Calculate speed at this crossing line
                        self._calculate_vehicle_speed(obj_id, speed_line_id)
                        
                        # Mark line as processed
                        if 'processed_lines' not in self.vehicles[obj_id]:
                            self.vehicles[obj_id]['processed_lines'] = set()
                        self.vehicles[obj_id]['processed_lines'].add((speed_line_id, 'speed'))
            
            # Get box coordinates for drawing
            if 'box' in track_data:
                x1, y1, x2, y2 = track_data['box']
            else:
                # Estimate box from center points if not available
                center_x, center_y = current_pos
                box_size = 50  # Default box size
                x1, y1 = center_x - box_size//2, center_y - box_size//2
                x2, y2 = center_x + box_size//2, center_y + box_size//2
            
            # Draw speed if available
            if obj_id in self.speeds:
                # Only display speed for active vehicles or for a limited time (1 second only)
                is_visible = obj_id in current_ids
                was_recently_visible = False
                
                # Check if vehicle was recently visible (within last 1 second)
                if 'last_seen_time' in track_data:
                    was_recently_visible = (current_time - track_data['last_seen_time'] < 1.0)  # Changed from 3.0 to 1.0
                
                if is_visible or was_recently_visible:
                    # Use different display for vehicles that have recorded speeds
                    if obj_id in self.recorded_speeds:
                        speed_text = f"{self.recorded_speeds[obj_id]:.1f} km/h"
                        text_color = (0, 0, 255)  # Red for recorded speeds
                    else:
                        speed_text = f"{self.speeds[obj_id]:.1f} km/h"
                        text_color = (0, 255, 0)  # Green for real-time speeds
                    
                    # Use selected font style
                    cv2.putText(
                        frame,
                        speed_text,
                        (x1, y1 - 10),
                        self.font,
                        self.font_scale,
                        text_color,
                        self.font_thickness,
                        cv2.LINE_AA  # Anti-aliased line for smoother appearance
                    )
        
        # Remove vehicles that are no longer in the frame for a while
        ids_to_remove = []
        for vehicle_id in self.vehicles:
            if vehicle_id not in current_ids:
                # If vehicle hasn't been seen for at least 30 frames, remove it
                if len(self.vehicles[vehicle_id]['timestamps']) > 0:
                    if current_time - self.vehicles[vehicle_id]['timestamps'][-1] > 1.0:  # 1 second timeout
                        ids_to_remove.append(vehicle_id)
        
        # Remove stale vehicle records
        for vehicle_id in ids_to_remove:
            if vehicle_id in self.vehicles:
                del self.vehicles[vehicle_id]
            if vehicle_id in self.speeds:
                del self.speeds[vehicle_id]
                
        return frame
    
    def _calculate_vehicle_speed(self, vehicle_id, speed_line_id=None):
        """Calculate speed for a specific vehicle"""
        if vehicle_id not in self.vehicles or len(self.vehicles[vehicle_id]['positions']) < 5:
            return
            
        positions = self.vehicles[vehicle_id]['positions']
        timestamps = self.vehicles[vehicle_id]['timestamps']
        
        # Use at least 5 positions for more stable measurements
        history_index = min(5, len(positions) - 1)
        
        # Calculate pixel distance between first and last position in history
        first_pos = positions[-history_index-1]
        last_pos = positions[-1]
        
        pixel_distance = np.sqrt(
            (last_pos[0] - first_pos[0]) ** 2 + 
            (last_pos[1] - first_pos[1]) ** 2
        )
        
        # Only calculate speed if movement is significant (reduces noise)
        if pixel_distance > self.min_distance_for_calc:
            # Convert to real-world distance (meters)
            distance = pixel_distance / self.pixels_per_meter
            
            # Speed = distance / time (m/s)
            time_elapsed = timestamps[-1] - timestamps[-history_index-1]
            if time_elapsed > 0:
                speed_ms = distance / time_elapsed
                
                # Convert to km/h
                speed_kmh = speed_ms * 3.6
                
                # Limit maximum speed for realism
                speed_kmh = min(speed_kmh, 200.0)
                
                # Apply stability constraint (limit rate of change)
                last_speed = self.vehicles[vehicle_id]['last_speed']
                if last_speed > 0:
                    # Limit speed change to prevent jumps
                    if abs(speed_kmh - last_speed) > self.max_speed_change:
                        if speed_kmh > last_speed:
                            speed_kmh = last_speed + self.max_speed_change
                        else:
                            speed_kmh = last_speed - self.max_speed_change
                
                # Store the calculated speed
                self.vehicles[vehicle_id]['speeds'].append(speed_kmh)
                self.vehicles[vehicle_id]['last_speed'] = speed_kmh
                
                # Apply smoothing - calculate weighted average of recent speed measurements
                # Use exponential weighting to favor more recent measurements
                window_size = min(self.smoothing_window, len(self.vehicles[vehicle_id]['speeds']))
                if window_size > 0:
                    weights = np.exp(np.linspace(0, 1, window_size))
                    weighted_speeds = np.array(self.vehicles[vehicle_id]['speeds'][-window_size:]) * weights
                    avg_speed = weighted_speeds.sum() / weights.sum()
                    
                    # Update speed for display
                    self.speeds[vehicle_id] = avg_speed
                    
                    # If this was a speed line crossing, record the speed
                    if speed_line_id is not None:
                        self.recorded_speeds[vehicle_id] = avg_speed
                        print(f"[{self.stream_id}] Vehicle {vehicle_id} speed at line {speed_line_id}: {avg_speed:.1f} km/h")
                        
                        # Create a unique identifier for this vehicle and speed line crossing
                        violation_key = f"{vehicle_id}_{speed_line_id}"
                        
                        # Record speed violation if over limit (1 km/h) and not already recorded
                        if avg_speed > self.speed_limit and violation_key not in self.speed_violations:
                            try:
                                # Add to violations set to prevent duplicate recordings
                                self.speed_violations.add(violation_key)
                                
                                # Get the vehicle bbox from tracked objects
                                bbox = self._get_vehicle_bbox(vehicle_id)
                                
                                if bbox is not None and hasattr(self, 'current_frame'):
                                    # Record violation through unified manager if available
                                    if self.violation_manager:
                                        violation_id, snapshot_paths = self.violation_manager.record_speed_violation(
                                            self.current_frame, 
                                            vehicle_id, 
                                            bbox,
                                            speed=avg_speed
                                        )
                                        print(f"[{self.stream_id}] Recorded SPEED VIOLATION for vehicle {vehicle_id}: " +
                                             f"{avg_speed:.1f} km/h (limit: {self.speed_limit} km/h) at line {speed_line_id}")
                                    else:
                                        # Standalone approach
                                        self._save_violation_snapshot(self.current_frame, bbox, vehicle_id, avg_speed)
                            except Exception as e:
                                print(f"[{self.stream_id}] Error recording speed violation: {str(e)}")

    def _get_vehicle_bbox(self, vehicle_id):
        """Get the vehicle's bounding box from multiple possible sources"""
        # First try to get from tracked_objects (most reliable and up-to-date)
        if hasattr(self, 'tracked_objects') and vehicle_id in self.tracked_objects:
            if 'box' in self.tracked_objects[vehicle_id]:
                return self.tracked_objects[vehicle_id]['box']
            elif 'bbox' in self.tracked_objects[vehicle_id]:
                return self.tracked_objects[vehicle_id]['bbox']
        
        # Then try from our internal vehicle data
        if vehicle_id in self.vehicles:
            if 'box' in self.vehicles[vehicle_id]:
                return self.vehicles[vehicle_id]['box']
            elif 'bbox' in self.vehicles[vehicle_id]:
                return self.vehicles[vehicle_id]['bbox']
        
        # If we get here, we couldn't find a valid bbox
        return None

    def _save_violation_snapshot(self, frame, bbox, vehicle_id, speed):
        """Save snapshots of speeding vehicle (closeup and context)"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Make a clean copy of the frame
            clean_frame = frame.copy()
            
            # Extract vehicle bbox coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate expanded box dimensions for more context
            width, height = x2 - x1, y2 - y1
            padding_x, padding_y = int(width * 0.6), int(height * 0.6)
            
            # Ensure expanded box stays within frame bounds
            h, w = clean_frame.shape[:2]
            ex1 = max(0, x1 - padding_x)
            ey1 = max(0, y1 - padding_y)
            ex2 = min(w-1, x2 + padding_x)
            ey2 = min(h-1, y2 + padding_y)
            
            if ex2 > ex1 and ey2 > ey1:
                # Crop vehicle image with wider context
                vehicle_img = clean_frame[ey1:ey2, ex1:ex2]
                
                # Add violation text to the snapshot with font thickness 1
                cv2.putText(vehicle_img, f"SPEED: {speed:.1f} km/h", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)  # Changed from 2 to 1
                cv2.putText(vehicle_img, f"LIMIT: {self.speed_limit} km/h", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)  # Changed from 2 to 1
                cv2.putText(vehicle_img, f"Vehicle ID: {vehicle_id}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Create directories if they don't exist
                snapshots_dir = os.path.join(self.output_dir, "snapshots")
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Continue with the existing snapshot saving logic
                # Save closeup snapshot
                closeup_path = os.path.join(snapshots_dir, f"speed_violation_closeup_{self.stream_id}_veh{vehicle_id}_{timestamp}.jpg")
                cv2.imwrite(closeup_path, vehicle_img)
                
                # Mark the violation on the full context image
                cv2.rectangle(clean_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Changed from 2 to 1
                cv2.putText(clean_frame, f"SPEED: {speed:.1f} km/h", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Save full frame context
                full_path = os.path.join(snapshots_dir, f"speed_violation_full_{self.stream_id}_veh{vehicle_id}_{timestamp}.jpg")
                cv2.imwrite(full_path, clean_frame)
                
                print(f"[{self.stream_id}] Saved speed violation snapshots for vehicle {vehicle_id} at {speed:.1f} km/h")
                return True
            else:
                print(f"[{self.stream_id}] Invalid crop dimensions for vehicle {vehicle_id}")
                return False
                
        except Exception as e:
            print(f"[{self.stream_id}] Error saving speed violation snapshot: {str(e)}")
            return False
    
    def enable_csv_logging(self, enable=True):
        """Enable or disable CSV logging (disabled by default)"""
        self.disable_csv_logging = not enable
        status = "enabled" if enable else "disabled"
        print(f"[{self.stream_id}] CSV logging {status}")
        return not self.disable_csv_logging
    
    def set_calibration(self, pixels_per_meter):
        """Set calibration parameter for speed calculation"""
        self.pixels_per_meter = pixels_per_meter
        print(f"[{self.stream_id}] Speed calibration set to {pixels_per_meter} pixels/meter")
    
    def set_speed_limit(self, limit_kmh):
        """Set the speed limit in km/h"""
        self.speed_limit = float(limit_kmh)
        print(f"[{self.stream_id}] Speed limit set to {self.speed_limit} km/h")
    
    def reset_violations(self):
        """Reset the list of recorded speed violations"""
        self.speed_violations.clear()
        print(f"[{self.stream_id}] Speed violation records cleared")
        
    def set_font_style(self, font_style):
       
        if font_style.lower() == "simplex":
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.font_scale = 0.4
            self.font_thickness = 1
        else:  # Default to plain
            self.font = cv2.FONT_HERSHEY_PLAIN
            self.font_scale = 0.4
            self.font_thickness = 1
    
    def enable(self):
        """Enable speed detection after speed lines are defined"""
        self.enabled = True
        print(f"[{self.stream_id}] Speed detection enabled")
        
    def disable(self):
        """Disable speed detection"""
        self.enabled = False
        print(f"[{self.stream_id}] Speed detection disabled")
        
    def is_enabled(self):
        """Check if speed detection is enabled"""
        return self.enabled