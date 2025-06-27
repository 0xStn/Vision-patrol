import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self):
        self.green_range = {
            'lower': np.array([58, 97, 222]),
            'upper': np.array([179, 255, 255])
        }
        
        self.red_range = {
            'lower': np.array([0, 43, 184]),
            'upper': np.array([56, 132, 255])
        }
        
        # Add current traffic light state and flags
        self.current_state = None  # None, "RED", or "GREEN"
        self.is_red = False  # Flag for quick access to red light state
        
    def set_color_range(self, color, lower_range, upper_range):
        if color.lower() == 'green':
            self.green_range = {'lower': lower_range, 'upper': upper_range}
        elif color.lower() == 'red':
            self.red_range = {'lower': lower_range, 'upper': upper_range}
    
    def process_frame(self, frame, min_contour_area=50):
        if frame is None or frame.size == 0:
            return None, None
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        green_mask = cv2.inRange(hsv, self.green_range['lower'], self.green_range['upper'])
        red_mask = cv2.inRange(hsv, self.red_range['lower'], self.red_range['upper'])
        
        combined_mask = cv2.bitwise_or(green_mask, red_mask)
        _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)
        detected_label = None
        
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.contourArea(c) > min_contour_area:
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w // 2
                cy = y + h // 2
                
                if cx >= 915:
                    continue
                
                if cv2.countNonZero(green_mask[y:y+h, x:x+w]) > 0:
                    color = (0, 255, 0)
                    text_color = (0, 255, 0)
                    label = "GREEN"
                elif cv2.countNonZero(red_mask[y:y+h, x:x+w]) > 0:
                    color = (0, 0, 255)
                    text_color = (0, 0, 255)
                    label = "RED"
                else:
                    continue
                
                detected_label = label
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Update the current state and flag
        self.current_state = detected_label
        self.is_red = (detected_label == "RED")
        
        return frame, detected_label
    
    # Get traffic light state
    def get_traffic_light_state(self):
        return self.current_state, self.is_red
    
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

def process_frame(frame):
    """Legacy function for backward compatibility"""
    detector = TrafficLightDetector()
    return detector.process_frame(frame)