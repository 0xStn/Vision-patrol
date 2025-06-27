# Vision Patrol - Accident Detection Module
import cv2
import numpy as np
import time
import threading
import logging
from datetime import datetime
from ultralytics import YOLO
import os
from collections import deque

class AccidentDetector:
    def __init__(self, stream_id="default", model_path=None, conf_threshold=0.86, cooldown=35, frame_skip=3, accident_alert_manager=None):
        
        # Initialize accident model path
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "accident.pt")
            
        # Load accident detection model
        self.model = YOLO(model_path)
        self.stream_id = stream_id
        self.conf_threshold = conf_threshold
        self.cooldown = cooldown  # Cooldown period in seconds
        self.frame_skip = frame_skip
        self.accident_classes = [
            'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
            'car_bike_accident', 'car_car_accident', 'car_object_accident', 'car_person_accident'
        ]
        
        # State management variables
        self.last_alert_time = 0
        self.alert_active = False
        self.frame_counter = 0
        self.snapshot_taken = False
        
        # Accident alert manager for centralized logging
        self.accident_alert_manager = accident_alert_manager
        
        # Detection control flags
        self.detection_enabled = False
        self.alerts_disabled = False
        self.alerts_disabled_until = 0
        
        # Auto-disable functionality parameters
        self.repeated_accidents = {}     # Track repeated accidents by location
        self.accident_timestamps = []    # Store recent accident timestamps
        self.auto_disable_count = 3      # Accidents threshold for auto-disable
        self.auto_disable_window = 80    # Time window in seconds
        self.auto_disable_duration = 600 # Auto-disable duration (10 minutes)
        self.auto_disable_until = 0      # Auto-disable end timestamp
        self.auto_disabled = False       # Flag indicating if model was auto-disabled
        
        # Vehicle ID tracking with more robust tracking
        self.vehicles_in_accident = set()  # Store vehicle IDs involved in the last accident
        self.accidents_history = deque(maxlen=5)  # Store recent accident locations to avoid duplicate alerts
        self.accident_area_radius = 180  # Radius in pixels to consider as the same accident area
        self.current_accident_signature = None  # Track characteristics of current accident
        self.detected_accident = None  # Store current accident detection details

        # Setup dummy logger instead of file logger
        self._setup_dummy_logger()
        
        print(f"[{self.stream_id}] Accident detector initialized with model: {model_path}")
        print(f"[{self.stream_id}] Detection confidence threshold: {self.conf_threshold}")
        print(f"[{self.stream_id}] Alert cooldown period: {self.cooldown} seconds")
        print(f"[{self.stream_id}] Single closeup snapshot mode enabled")
        print(f"[{self.stream_id}] Auto-disable after {self.auto_disable_count} accidents in {self.auto_disable_window} seconds for {self.auto_disable_duration/60} minutes")
        print(f"[{self.stream_id}] Accident detection is DISABLED by default. Press 'a' to toggle.")
        print(f"[{self.stream_id}] Alert response toggle is available using 'r'.")
        
        # Report if we're using a centralized alert manager
        if self.accident_alert_manager:
            print(f"[{self.stream_id}] Using central accident alert manager for logging and snapshots")
    
    def _setup_dummy_logger(self):
        """Set up a dummy logger that doesn't write to file"""
        self.logger = logging.getLogger(f"accident_{self.stream_id}")
        self.logger.setLevel(logging.INFO)
        
        # Add a NullHandler which doesn't write to any file
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    def detect_accidents(self, frame, tracked_objects=None):
        if frame is None:
            return None
            
        # Create a copy to avoid modifying the original frame
        processed_frame = frame.copy()
        
        # Get current time
        current_time = time.time()
        
        # Check if auto-disable period has expired
        if self.auto_disabled and current_time > self.auto_disable_until:
            self.auto_disabled = False
            self.accident_timestamps = []  # Reset accident timestamps
            print(f"[{self.stream_id}] Auto-disable period ended. Re-enabling detection.")
            
        # Check if alerts-disabled period has expired
        if self.alerts_disabled and current_time > self.alerts_disabled_until:
            self.alerts_disabled = False
            print(f"[{self.stream_id}] Alert suppression period ended. Re-enabling accident alerts.")
        
        # Show detection and alert status (minimal)
        self._draw_minimal_status(processed_frame, current_time)
        
        # Skip if accident detection is disabled or auto-disabled
        if (not self.detection_enabled or self.auto_disabled) and not self.alert_active:
            return processed_frame
            
        # Increment frame counter regardless of skipping
        self.frame_counter += 1
        
        # Reset cooldown if expired during active alert
        if self.alert_active and current_time - self.last_alert_time >= self.cooldown:
            self._reset_alert_state()
            print(f"[{self.stream_id}] Cooldown ended. Ready for new alerts.")

        # Skip detection processing based on frame_skip for efficiency
        if self.frame_counter % self.frame_skip != 0 or not self.detection_enabled or self.alert_active or self.auto_disabled:
            return processed_frame

        # Run accident detection on the entire frame
        results = self.model(processed_frame)[0]
        
        # Filter and sort accident detections by confidence
        accident_boxes = []
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = results.names[cls]
            
            # Check if this is an accident with sufficient confidence
            if class_name in self.accident_classes and conf >= self.conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                box_size = (x2 - x1) * (y2 - y1)  # Box area for signature
                accident_boxes.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'box': (x1, y1, x2, y2),
                    'center': box_center,
                    'size': box_size  # Add size as part of accident signature
                })
        
        # Sort by confidence (highest first)
        accident_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Process detections - immediate detection without validation
        if accident_boxes:
            # Get the highest confidence accident detection
            self.detected_accident = accident_boxes[0]
            
            # Draw minimal bounding box for the detected accident (thin line)
            self._draw_minimal_accident_box(processed_frame, self.detected_accident)
            
            # If alerts are not disabled, handle the new accident immediately
            if not self.alerts_disabled:
                self._handle_new_accident(processed_frame, tracked_objects, current_time)
            else:
                # Minimal suppressed alert indicator
                cv2.putText(processed_frame, "ALERT SUPPRESSED", 
                        (processed_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        else:
            self.detected_accident = None
        
        # Clean up old accident records
        self._cleanup_old_data(current_time)
            
        return processed_frame

    def _draw_minimal_status(self, frame, current_time):
        """Draw minimal status indicator only when disabled or auto-disabled"""
        # Detection status - only show if disabled or auto-disabled
        if self.auto_disabled:
            remaining = int(self.auto_disable_until - current_time)
            minutes = remaining // 60
            seconds = remaining % 60
            status_text = f"AUTO-DISABLED ({minutes}:{seconds:02d})"
            cv2.putText(frame, status_text, (20, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        elif not self.detection_enabled:
            cv2.putText(frame, "DETECTION OFF", (20, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def _draw_minimal_accident_box(self, frame, accident):
        """Draw minimal accident detection box - thin line only"""
        x1, y1, x2, y2 = accident['box']
        # Just a thin red rectangle - no text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def _handle_new_accident(self, frame, tracked_objects, current_time):
        """Handle a new accident detection"""
        # Extract accident details
        accident = self.detected_accident
        x1, y1, x2, y2 = accident['box']
        box_center = accident['center']
        box_size = accident['size']
        
        # Create accident signature
        self.current_accident_signature = {
            'center': box_center,
            'box': accident['box'],
            'size': box_size
        }
        
        # Find nearby vehicles that might be involved in the accident
        current_vehicles = set()
        if tracked_objects:
            for track_id, track_info in tracked_objects.items():
                if 'center_points' in track_info and track_info['center_points']:
                    veh_center = track_info['center_points'][-1]  # Latest position
                    # Calculate distance to accident
                    dist = np.sqrt((veh_center[0] - box_center[0])**2 + 
                                  (veh_center[1] - box_center[1])**2)
                    if dist < self.accident_area_radius:  # Vehicle is within accident area
                        current_vehicles.add(track_id)
        
        # Update tracking data
        self.vehicles_in_accident = current_vehicles
        self.accidents_history.append((box_center, current_time, box_size))
        
        # Add current timestamp to accident_timestamps for auto-disable tracking
        self.accident_timestamps.append(current_time)
        
        # Check if we need to auto-disable detection due to frequent accidents
        self._check_auto_disable(current_time)
        
        # Start the accident alert
        self.alert_active = True
        self.last_alert_time = current_time
        
        # Take a single closeup snapshot immediately
        self._take_accident_snapshot(frame, tracked_objects, accident['box'])
        self.snapshot_taken = True
        
        # Log the accident
        self._log_accident(accident['confidence'], current_time, current_vehicles)

    def _check_auto_disable(self, current_time):
        """Check if we should auto-disable detection due to frequent accidents"""
        # Keep only timestamps within the window
        recent_timestamps = [t for t in self.accident_timestamps 
                           if current_time - t <= self.auto_disable_window]
        self.accident_timestamps = recent_timestamps
        
        # Check if we've exceeded the threshold
        if len(self.accident_timestamps) >= self.auto_disable_count:
            self.auto_disabled = True
            self.auto_disable_until = current_time + self.auto_disable_duration
            minutes = self.auto_disable_duration // 60
            print(f"[{self.stream_id}] 🚫 AUTO-DISABLED for {minutes} minutes due to {self.auto_disable_count} accidents in {self.auto_disable_window} seconds")
            
    def _take_accident_snapshot(self, frame, tracked_objects, bbox):
        """Take a single closeup snapshot of the accident area"""
        # Use original frame without any overlays for clean snapshot
        clean_frame = frame.copy()
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            
            # Calculate expanded box with extra wide margin for context
            h, w = clean_frame.shape[:2]
            margin = int(max(x2-x1, y2-y1) * 0.75)  # 75% margin for even wider boxes
            
            # Ensure expanded box stays within frame bounds
            ex1 = max(0, x1 - margin)
            ey1 = max(0, y1 - margin)
            ex2 = min(w-1, x2 + margin)
            ey2 = min(h-1, y2 + margin)
            
            # Extract region if valid
            if ex1 < ex2 and ey1 < ey2:
                # Get closeup area
                accident_closeup = clean_frame[ey1:ey2, ex1:ex2]
                
                # Draw thin bounding box only for involved vehicles
                if tracked_objects and self.vehicles_in_accident:
                    for veh_id in self.vehicles_in_accident:
                        if veh_id in tracked_objects:
                            if 'bbox' in tracked_objects[veh_id]:
                                veh_box = tracked_objects[veh_id]['bbox']
                                if len(veh_box) == 4:
                                    # Adjust coordinates to closeup region
                                    vx1 = max(0, int(veh_box[0]) - ex1)
                                    vy1 = max(0, int(veh_box[1]) - ey1)
                                    vx2 = min(ex2-ex1, int(veh_box[2]) - ex1)
                                    vy2 = min(ey2-ey1, int(veh_box[3]) - ey1)
                                    
                                    # Only draw if box is within the closeup area
                                    if vx1 < vx2 and vy1 < vy2:
                                        # Draw thin blue rectangle with no label
                                        cv2.rectangle(accident_closeup, 
                                                    (vx1, vy1), (vx2, vy2),
                                                    (255, 0, 0), 1)  # Thin line (1px)
                
                # Use accident alert manager to save the snapshot
                if self.accident_alert_manager:
                    self.accident_alert_manager.record_accident(
                        accident_closeup,  # Send the closeup
                        self.vehicles_in_accident,
                        bbox=None  # No need for bbox in closeup
                    )
                else:
                    # Save locally as fallback
                    self._save_accident_locally(accident_closeup)
        else:
            print(f"[{self.stream_id}] No valid bounding box for accident snapshot")

    def _save_accident_locally(self, frame):
        """Log that we would save an accident snapshot but don't actually save it"""
        try:
            # Only save if using accident alert manager
            if self.accident_alert_manager:
                # Create a filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create snapshots directory if needed
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accidents")
                os.makedirs(output_dir, exist_ok=True)
                
                # Create filename
                filename = os.path.join(output_dir, f"accident_{self.stream_id}_{timestamp}.jpg")
                
                # Save the image
                cv2.imwrite(filename, frame)
                print(f"[{self.stream_id}] Saved local snapshot: {os.path.basename(filename)}")
                return True
            else:
                # Just log the event without saving
                print(f"[{self.stream_id}] Accident detected (local saving disabled)")
                return True
        except Exception as e:
            print(f"[{self.stream_id}] Error processing accident snapshot: {str(e)}")
            return False

    def _reset_alert_state(self):
        """Reset the alert state when cooldown expires"""
        self.alert_active = False
        self.current_accident_signature = None
        self.snapshot_taken = False

    def _cleanup_old_data(self, current_time):
        """Clean up old accident records"""
        # Clean up timestamps older than the detection window
        self.accident_timestamps = [t for t in self.accident_timestamps 
                                   if current_time - t <= self.auto_disable_window]

    def _log_accident(self, confidence, current_time, vehicle_ids):
        """Log a new accident detection"""
        # Use accident alert manager if available
        if self.accident_alert_manager:
            print(f"[{self.stream_id}] 🚨 ACCIDENT DETECTED 🚨 (confidence: {confidence:.2f})")
            print(f"[{self.stream_id}] Vehicles involved: {vehicle_ids}")
        else:
            # Minimal basic logging
            vehicle_id_str = f" involving vehicle IDs: {vehicle_ids}" if vehicle_ids else ""
            self._log_detection("ACCIDENT", confidence, vehicle_id_str)
            print(f"[{self.stream_id}] ACCIDENT DETECTED (confidence: {confidence:.2f}){vehicle_id_str}")

    def _log_detection(self, event_type, confidence, additional_info=""):
        """Log accident detection but don't write to file"""
        # Just print to console instead of writing to log file
        print(f"[{self.stream_id}] {event_type} detected with confidence {confidence:.2f}{additional_info}")

    def toggle_detection(self):
        """Toggle accident detection on/off"""
        self.detection_enabled = not self.detection_enabled
        status = "ENABLED" if self.detection_enabled else "DISABLED"
        print(f"[{self.stream_id}] Accident detection {status}")
        return self.detection_enabled
        
    def toggle_alerts(self, duration=60):
        """Toggle alert response - disable alerts temporarily but keep detection running"""
        if self.alerts_disabled:
            # Re-enable alerts
            self.alerts_disabled = False
            print(f"[{self.stream_id}] Accident alerts re-enabled")
        else:
            # Disable alerts with auto-reactivation
            self.alerts_disabled = True
            self.alerts_disabled_until = time.time() + duration
            print(f"[{self.stream_id}] Accident alerts DISABLED for {duration} seconds")
        
        return self.alerts_disabled

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        print(f"Processing video: {video_path}")
        window_name = f"Accident Detection - {os.path.basename(video_path)}"
        cv2.namedWindow(window_name)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame for accident detection
            processed_frame = self.detect_accidents(frame)
            
            # Display the frame
            cv2.imshow(window_name, processed_frame)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()