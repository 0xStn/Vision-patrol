# Vision Patrol - Multiprocessing Video Processor
import cv2
import multiprocessing
import os
import time
import sys
import importlib.util
import numpy as np
from ultralytics import YOLO

# Configure Python paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local modules
from areas import AreaManager, AreaType
from VideoReader import VideoReader

# Set up model directory paths
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, "Models")

# Add individual model directories to sys.path
accident_dir = os.path.join(models_dir, "Accident_det")
if accident_dir not in sys.path:
    sys.path.append(accident_dir)

speed_dir = os.path.join(models_dir, "Speed Model")
if speed_dir not in sys.path:
    sys.path.append(speed_dir)

parking_dir = os.path.join(models_dir, "Parking Model")
if parking_dir not in sys.path:
    sys.path.append(parking_dir)

wrong_dir = os.path.join(models_dir, "Wrong_dir")
if wrong_dir not in sys.path:
    sys.path.append(wrong_dir)

traffic_violation_dir = os.path.join(models_dir, "Traffic_violation")
if traffic_violation_dir not in sys.path:
    sys.path.append(traffic_violation_dir)

helmet_violation_dir = os.path.join(models_dir, "Bike_Violations")
if helmet_violation_dir not in sys.path:
    sys.path.append(helmet_violation_dir)

# Add Violation_Proc directory for the unified violation manager
violation_proc_dir = os.path.join(current_dir, "Violation_Proc")
if violation_proc_dir not in sys.path:
    sys.path.append(violation_proc_dir)

# Import accident detector module
accident_module_path = os.path.join(accident_dir, "Accident_Detector.py")
spec = importlib.util.spec_from_file_location("accident_detector", accident_module_path)
accident_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accident_module)
AccidentDetector = accident_module.AccidentDetector
# Import modules directly from their file paths to handle spaces in directory names
import importlib.util

# Import SpeedDetector module
speed_module_path = os.path.join(speed_dir, "speed_detector.py")
spec = importlib.util.spec_from_file_location("speed_detector", speed_module_path)
speed_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed_module)
SpeedDetector = speed_module.SpeedDetector

# Import ParkingDetector module
parking_module_path = os.path.join(parking_dir, "parking_Detctor.py")
spec = importlib.util.spec_from_file_location("parking_Detctor", parking_module_path)
parking_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parking_module)
ParkingDetector = parking_module.ParkingDetector
# Import WrongDirectionDetector module
wrong_dir_module_path = os.path.join(wrong_dir, "wrong_det.py")
spec = importlib.util.spec_from_file_location("wrong_det", wrong_dir_module_path)
wrong_dir_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrong_dir_module)
WrongDirectionDetector = wrong_dir_module.WrongDirectionDetector

# Import TrafficViolationDetector module
traffic_module_path = os.path.join(traffic_violation_dir, "traffic_violation_detector.py")
spec = importlib.util.spec_from_file_location("traffic_violation_detector", traffic_module_path)
traffic_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(traffic_module)
TrafficViolationDetector = traffic_module.TrafficViolationDetector

# Import HelmetViolationDetector module
helmet_module_path = os.path.join(helmet_violation_dir, "helmet_detector.py")
spec = importlib.util.spec_from_file_location("helmet_detector", helmet_module_path)
helmet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helmet_module)
HelmetViolationDetector = helmet_module.HelmetViolationDetector

# Import violation management modules  
from Violation_Proc.violation_manager import ViolationManager
from Violation_Proc.accident_alert_manager import AccidentAlertManager

# Define vehicle and bike class IDs for YOLOv8
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # car, motorcycle, airplane, bus, truck
BICYCLE_CLASSES = [4]  # bicycle

class VideoProcessorMP(multiprocessing.Process):
    def __init__(self, video_id, source, use_stream=False, camera_location="Unknown", coordinates=None, speed_limit=60):
        multiprocessing.Process.__init__(self)
        
        self.video_id = video_id
        self.source = source
        self.use_stream = use_stream
        self.camera_location = camera_location
        self.camera_coordinates = coordinates or {"lat": 0.0, "lng": 0.0}
        self.exit = multiprocessing.Event()
        
        self.window_name = f"Video Processing - {video_id}"
        self.tracked_objects = {}
        self.track_timeout = 30
        self.display_scale = 0.65
        
        self.original_width = None
        self.original_height = None
        self.display_width = None
        self.display_height = None
        
        # Initialize video reader as None - will be set in initialize_source
        self.video_reader = None
        
        # Control for auto-restart of video when it finishes
        self.auto_restart = True
        
        # Store speed limit parameter
        self.speed_limit = speed_limit
        
        # Add property for enabling/disabling detection models
        self.model_settings = {
            "accident_detection": True,
            "helmet_detection": True,
            "traffic_violation": True, 
            "speed_detection": True,
            "parking_detection": True,
            "wrong_direction": True
        }
    
    def run(self):
        try:
            # Initialize components inside the process
            print(f"[{self.video_id}] Process starting...")
            
            # Create window for this process
            cv2.namedWindow(self.window_name)
            
            # Initialize unified violation manager
            self.violation_manager = ViolationManager(
                stream_id=self.video_id,
                camera_location=self.camera_location,
                coordinates=self.camera_coordinates
            )
            print(f"[{self.video_id}] Unified violation manager initialized")
            
            # Get the accident alert manager reference from the violation manager
            self.accident_alert_manager = self.violation_manager.accident_alert_manager
            print(f"[{self.video_id}] Accident alert manager accessed from violation manager")
            
            # Initialize area manager
            self.area_manager = AreaManager(stream_id=self.video_id)

            # Initialize the video source
            self.initialize_source()
            
            # Read first frame to get dimensions
            ret, first_frame = self.video_reader.read()
            if not ret or first_frame is None:
                raise ValueError(f"Could not read first frame from {self.source}")
            
            # Store original dimensions and calculate display dimensions
            self.original_height, self.original_width = first_frame.shape[:2]
            self.display_width = int(self.original_width * self.display_scale)
            self.display_height = int(self.original_height * self.display_scale)
            
            # Reset video to start
            self.video_reader.reset()
            
            # Create custom mouse callback wrapper to handle coordinate scaling
            def scaled_mouse_callback(event, x, y, flags, param):
                # Scale coordinates back to original frame
                orig_x = int(x / self.display_scale)
                orig_y = int(y / self.display_scale)
                # Pass the scaled coordinates to the area manager's callback
                self.area_manager._mouse_callback(event, orig_x, orig_y, flags, param)
            
            # Set up our scaled mouse callback
            cv2.setMouseCallback(self.window_name, scaled_mouse_callback)
            
            # Initialize accident detector with the dedicated accident alert manager
            self.accident_detector = AccidentDetector(
                stream_id=self.video_id,
                accident_alert_manager=self.accident_alert_manager
            )
            print(f"[{self.video_id}] Accident detector initialized with dedicated alert manager")
            
            # Initialize speed detector with the specific stream ID and speed limit
            self.speed_detector = SpeedDetector(
                stream_id=self.video_id,
                violation_manager=self.violation_manager
            )
            # Set the speed limit after initialization
            self.speed_detector.set_speed_limit(self.speed_limit)
            print(f"[{self.video_id}] Speed detector initialized with speed limit: {self.speed_limit} km/h")
            
            # Initialize parking detector with the specific stream ID
            self.parking_detector = ParkingDetector(
                stream_id=self.video_id, 
                violation_time_limit=15,
                violation_manager=self.violation_manager
            )
            print(f"[{self.video_id}] Parking detector initialized")
            
            # Initialize wrong direction detector with the specific stream ID
            self.wrong_direction_detector = WrongDirectionDetector(
                stream_id=self.video_id,
                violation_manager=self.violation_manager
            )
            print(f"[{self.video_id}] Wrong direction detector initialized")
            
            # Initialize traffic violation detector
            self.traffic_violation_detector = TrafficViolationDetector(
                stream_id=self.video_id,
                violation_manager=self.violation_manager
            )
            print(f"[{self.video_id}] Traffic violation detector initialized")
            
            # Initialize helmet violation detector
            self.helmet_detector = HelmetViolationDetector(
                stream_id=self.video_id,
                violation_manager=self.violation_manager
            )
            print(f"[{self.video_id}] Helmet violation detector initialized")
            
            # Load YOLO model (each process needs its own model instance)
            print(f"[{self.video_id}] Loading YOLO model...")
            self.model = YOLO('yolov8s.pt')

            # Main processing loop
            print(f"[{self.video_id}] Starting processing loop...")
            frame_count = 0
            key = 0  # Initialize key variable to avoid undefined variable errors
              # Make sure area manager is initialized with correct AreaType
            print(f"[{self.video_id}] Area manager initialized with available area types: {[t.name for t in AreaType]}")
            
            # Configure wrong direction detector with area manager
            self.wrong_direction_detector.configure_line_pairs(self.area_manager)
            
            while not self.exit.is_set():
                try:
                    # Read frame using our VideoReader with error handling
                    ret, frame = self.video_reader.read()
                    
                    # Check if we reached the end of the video
                    if not ret or frame is None:
                        print(f"[{self.video_id}] End of video reached or frame reading failed")
                        
                        # If auto-restart is enabled and it's not a stream, reset the video
                        if self.auto_restart and not self.use_stream:
                            print(f"[{self.video_id}] Auto-restarting video...")
                            # Sleep briefly before reset to avoid rapid re-open
                            time.sleep(0.5)
                            
                            # Use try-except to handle reset errors
                            try:
                                self.video_reader.reset()
                            except Exception as e:
                                print(f"[{self.video_id}] Error resetting video: {str(e)}")
                                # Reinitialize video reader
                                self.video_reader.release()
                                self.initialize_source()
                            
                            # Clear tracked objects when restarting
                            self.tracked_objects = {}
                            
                            # Read the first frame after reset
                            ret, frame = self.video_reader.read()
                            if not ret or frame is None:
                                print(f"[{self.video_id}] Failed to restart video, exiting.")
                                break
                        else:
                            # No auto-restart or it's a stream that can't be restarted
                            print(f"[{self.video_id}] Video playback finished.")
                            break
                            
                except Exception as e:
                    print(f"[{self.video_id}] Error reading frame: {str(e)}")
                    # Try to recover by short sleep and continue
                    time.sleep(0.1)
                    continue

                # Create a copy of the frame to work with
                processed_frame = frame.copy()

                # Perform YOLO detection and tracking on the current frame with filtered classes
                results = self.model.track(source=frame, persist=True, classes=VEHICLE_CLASSES + BICYCLE_CLASSES, verbose=False)
                
                # Process detection results - only show boxes within the detection area
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    # Update tracked objects and handle area-based logic
                    self._process_detections(processed_frame, boxes)
                
                # Check if accident detection is enabled before running
                if self.model_settings.get("accident_detection", True):
                    # Run accident detection on the current frame
                    # Pass tracked objects so accident detector can identify which vehicles are involved
                    processed_frame = self.accident_detector.detect_accidents(processed_frame, self.tracked_objects)
                
                # Run speed calculation on the current frame if enabled
                if self.model_settings.get("speed_detection", True):
                    # Check if we have any speed lines defined before enabling the speed detector
                    if AreaType.SPEED in self.area_manager.areas and len(self.area_manager.areas[AreaType.SPEED]) > 0:
                        # Enable speed detection if not already enabled
                        if not self.speed_detector.is_enabled():
                            self.speed_detector.enable()
                            print(f"[{self.video_id}] Speed detection enabled - {len(self.area_manager.areas[AreaType.SPEED])} speed lines defined")
                        
                        # Calculate and display speeds
                        processed_frame = self.speed_detector.calculate_speed(processed_frame, self.tracked_objects)
                
                # Run parking violation detection if enabled
                if self.model_settings.get("parking_detection", True):
                    # Check if we have any parking areas defined
                    if AreaType.PARKING in self.area_manager.areas and len(self.area_manager.areas[AreaType.PARKING]) > 0:
                        # Process parking violations
                        processed_frame = self.parking_detector.update_vehicles(processed_frame, self.tracked_objects, self.area_manager)
                
                # Run wrong direction detection if enabled
                if self.model_settings.get("wrong_direction", True):
                    # Check if we have any wrong direction lines defined
                    if (AreaType.WRONG_DIRECTION in self.area_manager.areas and 
                        len(self.area_manager.areas[AreaType.WRONG_DIRECTION]) >= 2):
                        
                        # Process wrong direction violations                        
                        try:
                            # Configure line pairs periodically
                            if frame_count == 0 or frame_count % 100 == 0:  # Reconfigure periodically
                                self.wrong_direction_detector.configure_line_pairs(self.area_manager)
                                
                            # Run diagnostic only when there's a change in lines
                            if (key == ord('3') and 
                                AreaType.WRONG_DIRECTION in self.area_manager.areas and 
                                len(self.area_manager.areas[AreaType.WRONG_DIRECTION]) % 2 == 0):
                                
                                self.wrong_direction_detector.debug_area_manager(self.area_manager)
                                print(f"[{self.video_id}] Wrong direction detection configured with " +
                                    f"{len(self.area_manager.areas[AreaType.WRONG_DIRECTION])//2} line pairs")
                            
                            processed_frame = self.wrong_direction_detector.process_frame(
                                processed_frame, self.tracked_objects, self.area_manager)
                        except Exception as e:
                            print(f"[{self.video_id}] Error in wrong direction detection: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        # Remove the text display for wrong direction lines
                        # Keeping the logic to check for wrong direction lines but not displaying messages
                        pass
                
                # Run traffic violation detection if enabled
                if self.model_settings.get("traffic_violation", True):
                    # Check if we have any traffic sign areas or traffic lines defined
                    if ((AreaType.TRAFFIC_SIGN in self.area_manager.areas and 
                        len(self.area_manager.areas[AreaType.TRAFFIC_SIGN]) > 0) or
                        (AreaType.TRAFFIC_LINE in self.area_manager.areas and 
                        len(self.area_manager.areas[AreaType.TRAFFIC_LINE]) > 0)):
                        
                        # Process traffic violations
                        try:
                            processed_frame = self.traffic_violation_detector.process_frame(
                                processed_frame, self.tracked_objects, self.area_manager)
                        except Exception as e:
                            print(f"[{self.video_id}] Error in traffic violation detection: {str(e)}")
                            import traceback
                            traceback.print_exc()
                
                # Run helmet violation detection if enabled
                if self.model_settings.get("helmet_detection", True):
                    if AreaType.DETECTION in self.area_manager.areas and len(self.area_manager.areas[AreaType.DETECTION]) > 0:
                        # Process helmet violations
                        try:
                            processed_frame = self.helmet_detector.process_frame(
                                processed_frame, self.tracked_objects, self.area_manager)
                        except Exception as e:
                            print(f"[{self.video_id}] Error in helmet violation detection: {str(e)}")
                            import traceback
                            traceback.print_exc()
                
                # Draw all areas on the frame
                processed_frame = self.area_manager.draw_areas(processed_frame)
                
                # Display source info on frame
                cv2.putText(processed_frame, f"{self.video_id}", (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add indicator for resize mode
                cv2.putText(processed_frame, f"Display Scale: {self.display_scale:.2f}x", 
                          (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Resize the frame for display
                display_frame = self.resize_for_display(processed_frame)
                
                # Show the frame
                cv2.imshow(self.window_name, display_frame)
                
                # Periodically report progress
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"[{self.video_id}] Processed {frame_count} frames")

                # Handle key events for exiting and area management
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or q key - global exit
                    print(f"[{self.video_id}] Exit key pressed")
                    self.exit.set()
                    break
                
                # Handle area management keys
                self.area_manager.handle_key_events(key)
                
                # Handle accident detection toggle
                if key == ord('a'):
                    self.accident_detector.detection_enabled = not self.accident_detector.detection_enabled
                    status = "ENABLED" if self.accident_detector.detection_enabled else "DISABLED"
                    print(f"[{self.video_id}] Accident detection {status}")
                
                # Handle accident alert toggle (new)
                if key == ord('r'):
                    self.accident_detector.toggle_alerts(duration=120)  # Disable alerts for 2 minutes
                    status = "DISABLED" if self.accident_detector.alerts_disabled else "ENABLED"
                    print(f"[{self.video_id}] Accident alerts {status}")
                
                # Handle traffic violation detection toggle
                if key == ord('t'):
                    self.traffic_violation_detector.toggle_detection()
                
                # Handle helmet detection toggle
                if key == ord('h'):
                    self.helmet_detector.toggle_detection()
                    status = "ENABLED" if self.helmet_detector.detection_enabled else "DISABLED"
                    print(f"[{self.video_id}] Helmet detection {status}")
                
                # Handle display scaling controls
                if key == ord('+') or key == ord('='):
                    self.display_scale = min(1.0, self.display_scale + 0.05)
                    self.display_width = int(self.original_width * self.display_scale)
                    self.display_height = int(self.original_height * self.display_scale)
                    print(f"[{self.video_id}] Display scale increased to {self.display_scale:.2f}x")
                elif key == ord('-') or key == ord('_'):
                    self.display_scale = max(0.2, self.display_scale - 0.05)
                    self.display_width = int(self.original_width * self.display_scale)
                    self.display_height = int(self.original_height * self.display_scale)
                    print(f"[{self.video_id}] Display scale decreased to {self.display_scale:.2f}x")
                
                # Toggle auto-restart feature
                elif key == ord('l'):
                    self.auto_restart = not self.auto_restart
                    status = "ENABLED" if self.auto_restart else "DISABLED"
                    print(f"[{self.video_id}] Auto-restart {status}")
                    # Display status on frame
                    cv2.putText(processed_frame, f"Auto-restart: {status}", (20, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Clear all area configurations and restart
                elif key == ord('d'):
                    print(f"[{self.video_id}] Clearing all area configurations and restarting...")

                    # Use the area manager's clear and restart method
                    success = self.area_manager.clear_all_and_restart()

                    if success:
                        print(f"[{self.video_id}] Area configuration cleared and restarted successfully")
                        # Display status on frame
                        cv2.putText(processed_frame, "Area config CLEARED - Ready to define new areas", (20, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    else:
                        print(f"[{self.video_id}] Failed to clear area configuration")
                        cv2.putText(processed_frame, "Failed to clear area config", (20, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
        except Exception as e:
            print(f"[{self.video_id}] Error in process: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up resources
            self.cleanup()
    
    def _process_detections(self, frame, boxes):
        current_frame_tracks = set()
        current_time = time.time()  # Get current time for tracking
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get center point of box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
              # Check if detection is in the detection area
            if not self.area_manager.is_box_in_area((x1, y1, x2, y2), AreaType.DETECTION):
                continue  # Skip if not in detection area
            
            # Draw thin bounding box without class name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thinner line (1px)
            
            # Process tracking info if available
            if hasattr(box, 'id') and box.id is not None:
                track_id = box.id.item()
                current_frame_tracks.add(track_id)
                
                # Get or create tracked object
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        'last_seen': 0,
                        'last_seen_time': current_time , # Add timestamp for timeout
                        'center_points': [],
                        'crossed_lines': set(),  # Keep track of lines already crossed
                        'box': (x1, y1, x2, y2),  # Add box coordinates
                        'confidence': float(box.conf) if hasattr(box, 'conf') else 1.0  # Add confidence value
                    }
                
                track = self.tracked_objects[track_id]
                track['last_seen'] = 0  # Reset frame counter
                track['last_seen_time'] = current_time  # Update last seen timestamp
                track['bbox'] = (x1, y1, x2, y2)  # Update box coordinates in the correct format for wrong_det.py
                track['box'] = (x1, y1, x2, y2)   # Keep the original key as well
                track['confidence'] = float(box.conf) if hasattr(box, 'conf') else 1.0  # Update confidence
                
                # Add class_id to tracked objects for helmet detection
                if hasattr(box, 'cls'):
                    class_id = int(box.cls.item())
                    track['class_id'] = class_id
                
                # Add center point to history (limit to last 10 points)
                track['center_points'].append((center_x, center_y))
                if len(track['center_points']) > 10:
                    track['center_points'].pop(0)
                
                # Draw trajectory if we have enough points
                if len(track['center_points']) > 1:
                    for i in range(1, len(track['center_points'])):
                        cv2.line(frame, 
                                track['center_points'][i-1], 
                                track['center_points'][i], 
                                (0, 255, 255), 1)
                
                # Check if object is crossing any lines
                if len(track['center_points']) >= 2:
                    prev_point = track['center_points'][-2]
                    curr_point = track['center_points'][-1]
                    
                    # Check speed lines
                    speed_crossed, speed_line_id, speed_direction = self.area_manager.is_crossing_line(
                        prev_point, curr_point, AreaType.SPEED)
                    if speed_crossed and (speed_line_id, 'speed') not in track['crossed_lines']:
                        track['crossed_lines'].add((speed_line_id, 'speed'))
                        print(f"[{self.video_id}] Object {track_id} crossed speed line {speed_line_id} in direction {speed_direction}")
                        # Would calculate speed here in a real implementation
                    
                    # Check wrong direction lines
                    wrong_dir_crossed, wrong_dir_line_id, wrong_dir_direction = self.area_manager.is_crossing_line(
                        prev_point, curr_point, AreaType.WRONG_DIRECTION)
                    if wrong_dir_crossed and (wrong_dir_line_id, 'wrong_dir') not in track['crossed_lines']:
                        track['crossed_lines'].add((wrong_dir_line_id, 'wrong_dir'))
                        # Get wrong direction line properties - in a real implementation would check if direction is allowed
                        # For demonstration, we just report the crossing
                        print(f"[{self.video_id}] Object {track_id} crossed direction line {wrong_dir_line_id} in direction {wrong_dir_direction}")
                
                # Check if object is in parking area
                if self.area_manager.is_in_area(center_x, center_y, AreaType.PARKING):
                    # In a real implementation, we would track parking violations here
                    pass
                    
                # Check if object is in traffic line area
                if self.area_manager.is_in_area(center_x, center_y, AreaType.TRAFFIC_LINE):
                    # The traffic violation detector now handles this logic
                    pass
                
                # Check if object is in traffic sign area
                if self.area_manager.is_in_area(center_x, center_y, AreaType.TRAFFIC_SIGN):
                    # The traffic violation detector now handles this logic
                    pass
        
        # Update frame counter for all tracked objects and remove old ones
        tracks_to_remove = []
        for track_id, track in self.tracked_objects.items():
            if track_id not in current_frame_tracks:
                track['last_seen'] += 1
                # Only update last_seen_time if we don't already have one
                if 'last_seen_time' not in track:
                    track['last_seen_time'] = current_time
                
                # Remove if inactive for too long
                if track['last_seen'] > self.track_timeout:
                    tracks_to_remove.append(track_id)
            else:
                # Make sure the vehicle is marked as visible
                track['last_seen_time'] = current_time
          # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
    
    def initialize_source(self):
        """Initialize the video source using our VideoReader class"""
        try:
            print(f"[{self.video_id}] Initializing video source: {self.source}")
            
            # Clean up any existing video reader
            if hasattr(self, 'video_reader') and self.video_reader:
                try:
                    self.video_reader.release()
                except:
                    pass
                
            # Force a short delay before opening a new video to avoid FFmpeg threading issues
            time.sleep(0.2)
            
            # Create a new video reader instance
            self.video_reader = VideoReader(
                source=self.source, 
                use_stream=self.use_stream,
                verbose=True
            )
            
            # Verify that we can read from it
            ret, test_frame = self.video_reader.read()
            if not ret or test_frame is None:
                raise ValueError(f"Could not read initial frame from {self.source}")
                
            # Reset back to start if it's a file
            if not self.use_stream:
                self.video_reader.reset()
                
            print(f"[{self.video_id}] Video source initialized successfully")
            print(f"[{self.video_id}] FPS: {self.video_reader.get_fps()}")
        except Exception as e:
            print(f"[{self.video_id}] Error initializing video source: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up resources used by the video processor"""
        print(f"[{self.video_id}] Cleaning up resources...")
        
        # Release video resources
        try:
            if hasattr(self, 'video_reader') and self.video_reader:
                print(f"[{self.video_id}] Releasing video resources...")
                self.video_reader.release()
        except Exception as e:
            print(f"[{self.video_id}] Error releasing video resource: {str(e)}")
            
        try:
            # Try to close window - only if window_name exists
            if hasattr(self, 'window_name'):
                print(f"[{self.video_id}] Destroying window...")
                cv2.destroyWindow(self.window_name)
        except Exception as e:
            print(f"[{self.video_id}] Error destroying window: {str(e)}")
            
        print(f"[{self.video_id}] Cleanup completed.")
    
    def resize_for_display(self, frame):
        """Resize the frame for display purposes only"""
        if frame is None:
            return None
        
        # Use pre-calculated dimensions for consistent scaling
        return cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
