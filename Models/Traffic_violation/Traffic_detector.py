import cv2
import numpy as np
import pandas as pd
import cvzone
import os
from datetime import datetime

class TrafficViolations:
    def __init__(self, model=None, violation_area=None, traffic_light_area=None):
        self.model = model
        self.violation_area = violation_area or [(324, 313), (283, 374), (854, 392), (864, 322)]
        self.traffic_light_area = traffic_light_area or [(898, 99), (934, 96), (933, 199), (892, 200)]
        self.violated_ids = []
        
        # Default color ranges in HSV for traffic light detection
        self.green_range = {
            'lower': np.array([58, 97, 222]),
            'upper': np.array([179, 255, 255])
        }
        
        self.red_range = {
            'lower': np.array([0, 43, 184]),
            'upper': np.array([56, 132, 255])
        }
        
        self.setup_output_directory()
        
    def setup_output_directory(self, base_dir='saved_images'):
        today_date = datetime.now().strftime('%Y-%m-%d')
        self.output_dir = os.path.join(base_dir, today_date)
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir
        
    def set_violation_area(self, area_coordinates):
        self.violation_area = area_coordinates
        
    def set_traffic_light_area(self, area_coordinates):
        self.traffic_light_area = area_coordinates
    
    def set_color_range(self, color, lower_range, upper_range):
        if color.lower() == 'green':
            self.green_range = {'lower': lower_range, 'upper': upper_range}
        elif color.lower() == 'red':
            self.red_range = {'lower': lower_range, 'upper': upper_range}
        
    def process_frame(self, frame, detection_results=None):
        if frame is None:
            return None, {}
            
        result_frame = frame.copy()
        traffic_light_status = self.detect_traffic_light(result_frame)
        
        violations = []
        if self.model is not None or detection_results is not None:
            violations = self.detect_violations(result_frame, detection_results, traffic_light_status)
        
        cv2.polylines(result_frame, [np.array(self.violation_area, np.int32)], True, (0, 255, 0), 2)
        
        return result_frame, {
            'traffic_light': traffic_light_status,
            'violations': violations
        }
    
    def detect_traffic_light(self, frame):
        # Create mask for traffic light region
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [np.array(self.traffic_light_area, np.int32)], (255, 255, 255))
        masked_frame = cv2.bitwise_and(frame, mask)
        
        # Calculate ROI boundaries
        x_coords = [pt[0] for pt in self.traffic_light_area]
        y_coords = [pt[1] for pt in self.traffic_light_area]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Extract traffic light ROI
        traffic_light_roi = masked_frame[y_min:y_max, x_min:x_max]
        
        # Draw traffic light boundary
        cv2.polylines(frame, [np.array(self.traffic_light_area, np.int32)], True, (0, 255, 255), 2)
        
        # Process the traffic light ROI to detect status
        _, detected_label = self._process_traffic_light_roi(traffic_light_roi)
        return detected_label
    
    def _process_traffic_light_roi(self, frame, min_contour_area=50):
        if frame is None or frame.size == 0:
            return None, None
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for both color ranges
        green_mask = cv2.inRange(hsv, self.green_range['lower'], self.green_range['upper'])
        red_mask = cv2.inRange(hsv, self.red_range['lower'], self.red_range['upper'])
        
        # Combine the two masks
        combined_mask = cv2.bitwise_or(green_mask, red_mask)
        _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)
        detected_label = None
        
        # Find contours
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.contourArea(c) > min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w // 2
                cy = y + h // 2
                
                # Skip contours out of region of interest
                if cx >= 915:
                    continue
                
                # Determine the color of the contour
                if cv2.countNonZero(green_mask[y:y+h, x:x+w]) > 0:
                    color = (0, 255, 0)  # Green color for the rectangle
                    text_color = (0, 255, 0)  # Green text
                    label = "GREEN"
                elif cv2.countNonZero(red_mask[y:y+h, x:x+w]) > 0:
                    color = (0, 0, 255)  # Red color for the rectangle
                    text_color = (0, 0, 255)  # Red text
                    label = "RED"
                else:
                    continue
                
                detected_label = label
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw the center point
                cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)
                
                # Display text
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        return frame, detected_label
    
    def detect_violations(self, frame, detection_results=None, traffic_light_status=None):
        violations = []
        
        if detection_results is None and self.model is not None:
            detection_results = self.model(frame)
            a = detection_results[0].boxes.data.cpu()
            px = pd.DataFrame(a).astype("float")
        else:
            return violations
        
        # Process each detected vehicle
        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            confidence = float(row[4])
            class_id = int(row[5])
            
            # Check if object is a vehicle (car, truck, bus, motorcycle)
            if class_id in [2, 3, 5, 7]:  # Standard COCO indices for vehicles
                # Calculate center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                
                # Check if vehicle is in violation zone
                in_violation_zone = cv2.pointPolygonTest(np.array(self.violation_area, np.int32), (cx, cy), False)
                
                # Assign a simple ID based on index for demo purposes
                vehicle_id = index + 1
                
                if in_violation_zone >= 0:
                    if traffic_light_status == "RED":
                        # Mark violation
                        cvzone.putTextRect(frame, f'VIOLATION {vehicle_id}', (x1, y1), 2, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Save evidence if it's a new violation
                        if vehicle_id not in self.violated_ids:
                            self.save_violation_evidence(frame, vehicle_id)
                            self.violated_ids.append(vehicle_id)
                            
                        violations.append({
                            'id': vehicle_id,
                            'bbox': (x1, y1, x2, y2),
                            'center': (cx, cy),
                            'confidence': confidence,
                            'class_id': class_id
                        })
                    else:
                        # Not a violation if light is not red
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return violations
    
    def save_violation_evidence(self, frame, vehicle_id):
        timestamp = datetime.now().strftime('%H-%M-%S')
        file_path = os.path.join(self.output_dir, f'violation_{vehicle_id}_{timestamp}.jpg')
        cv2.imwrite(file_path, frame)
        return file_path
    
    def reset_violations(self):
        self.violated_ids = []
        
    def calibrate_color_detection(self, frame, roi=None):
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(hsv)
        h_min, h_max = np.min(h), np.max(h)
        s_min, s_max = np.min(s), np.max(s)
        v_min, v_max = np.min(v), np.max(v)
        
        return {
            'suggested_range': {
                'lower': np.array([h_min, s_min, v_min]),
                'upper': np.array([h_max, s_max, v_max])
            }
        }

# For backward compatibility
def process_frame(frame):
    detector = TrafficViolations()
    traffic_light_roi = frame.copy()
    return detector._process_traffic_light_roi(traffic_light_roi)