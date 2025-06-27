import cv2
import numpy as np
import pickle
import os
import json
from enum import Enum, auto

class AreaType(Enum):
    """Enum for different types of areas/ROIs"""
    DETECTION = auto()        # Object detection area
    SPEED = auto()            # Speed recognition lines
    WRONG_DIRECTION = auto()  # Wrong direction lines
    PARKING = auto()          # Parking ROI
    TRAFFIC_LINE = auto()     # Traffic line detection
    TRAFFIC_SIGN = auto()     # Traffic sign recognition
    CUSTOM = auto()           # Custom area type

class AreaManager:
    """
    Class to manage different types of ROIs and lines for various video analytics models
    """
    def __init__(self, stream_id="default"):
        self.stream_id = stream_id
        self.areas = {}  # Dictionary to store different areas by their type
        self.active_area_type = AreaType.DETECTION  # Default area type
        self.is_defining_area = False
        self.temp_points = []  # Temporary points while defining an area
        self.temp_point = None  # To show the line being drawn
        self.area_colors = {
            AreaType.DETECTION: (0, 255, 0),       # Green
            AreaType.SPEED: (255, 0, 0),           # Red
            AreaType.WRONG_DIRECTION: (0, 0, 255), # Blue
            AreaType.PARKING: (255, 255, 0),       # Yellow
            AreaType.TRAFFIC_LINE: (255, 0, 255),  # Magenta
            AreaType.TRAFFIC_SIGN: (0, 255, 255),  # Cyan
            AreaType.CUSTOM: (128, 128, 128)       # Gray
        }
        
        # Load existing area configurations
        self.load_areas()
    
    def setup_mouse_callback(self, window_name):
        """Set up mouse callback for the given window"""
        cv2.setMouseCallback(window_name, self._mouse_callback)
    
    def set_active_area_type(self, area_type):
        """Set the type of area being defined"""
        if not isinstance(area_type, AreaType):
            raise ValueError("area_type must be an AreaType enum value")
        self.active_area_type = area_type
        print(f"[{self.stream_id}] Active area type set to {area_type.name}")
        return True
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for area definition"""
        if not self.is_defining_area and event == cv2.EVENT_LBUTTONDOWN:
            # Start defining a new area
            self.is_defining_area = True
            self.temp_points = [(x, y)]
            if self.active_area_type not in self.areas:
                self.areas[self.active_area_type] = []
            print(f"[{self.stream_id}] Started defining area of type {self.active_area_type.name}")
            
        elif self.is_defining_area and event == cv2.EVENT_LBUTTONDOWN:
            # Add point to current area
            self.temp_points.append((x, y))
            print(f"[{self.stream_id}] Added point {len(self.temp_points)} to {self.active_area_type.name} area")
            
        elif self.is_defining_area and event == cv2.EVENT_MOUSEMOVE:
            # Track mouse movement
            self.temp_point = (x, y)
            
        elif self.is_defining_area and event == cv2.EVENT_RBUTTONDOWN:
            # Complete the area
            if len(self.temp_points) >= 3:  # Need at least 3 points for a polygon
                # Close the polygon by connecting to the first point
                area = self.temp_points + [self.temp_points[0]]
                self.areas[self.active_area_type].append({
                    'points': area,
                    'type': self.active_area_type.name,
                    'enabled': True,
                    'properties': {}  # Additional properties for specific area types
                })
                self.is_defining_area = False
                self.temp_points = []
                self.temp_point = None
                self.save_areas()
                print(f"[{self.stream_id}] Completed {self.active_area_type.name} area definition")
            elif len(self.temp_points) == 2 and self.active_area_type in [AreaType.SPEED, AreaType.WRONG_DIRECTION, AreaType.TRAFFIC_LINE]:
                # For line-based areas, 2 points are enough
                self.areas[self.active_area_type].append({
                    'points': self.temp_points.copy(),
                    'type': self.active_area_type.name,
                    'enabled': True,
                    'properties': {}
                })
                self.is_defining_area = False
                self.temp_points = []
                self.temp_point = None
                self.save_areas()
                print(f"[{self.stream_id}] Completed {self.active_area_type.name} line definition")
            else:
                print(f"[{self.stream_id}] Need at least 3 points for a polygon area or 2 points for a line")
    
    def handle_key_events(self, key=None):
        """Handle key events for area management"""
        if key is None:
            return True
            
        # Number keys 1-7 to select area type
        if key == ord('1'):
            self.set_active_area_type(AreaType.DETECTION)
            return True
        elif key == ord('2'):
            self.set_active_area_type(AreaType.SPEED)
            return True
        elif key == ord('3'):
            self.set_active_area_type(AreaType.WRONG_DIRECTION)
            return True
        elif key == ord('4'):
            self.set_active_area_type(AreaType.PARKING)
            return True
        elif key == ord('5'):
            self.set_active_area_type(AreaType.TRAFFIC_LINE)
            return True
        elif key == ord('6'):
            self.set_active_area_type(AreaType.TRAFFIC_SIGN)
            return True
        elif key == ord('7'):
            self.set_active_area_type(AreaType.CUSTOM)
            return True
        elif key == ord('c'):  # Clear current area type
            if self.active_area_type in self.areas:
                self.areas[self.active_area_type] = []
                self.is_defining_area = False
                self.temp_points = []
                self.temp_point = None
                self.save_areas()
                print(f"[{self.stream_id}] Cleared all areas of type {self.active_area_type.name}")
            return True
        elif key == ord('r'):  # Reset current area being defined
            self.is_defining_area = False
            self.temp_points = []
            self.temp_point = None
            print(f"[{self.stream_id}] Reset current area definition")
            return True
            
        return True
    
    def draw_areas(self, frame):
        """Draw all defined areas on the frame"""
        # Draw saved areas
        for area_type, areas in self.areas.items():
            color = self.area_colors.get(area_type, (255, 255, 255))
            
            for area in areas:
                points = area['points']
                
                # Draw line or polygon based on number of points
                if len(points) == 2:  # Line
                    cv2.line(frame, points[0], points[1], color, 1)
                    
                    # Add circles at endpoints (smaller)
                    cv2.circle(frame, points[0], 1, color, -1)
                    cv2.circle(frame, points[1], 1, color, -1)
                    
                else:  # Polygon
                    # Draw lines between consecutive points
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i + 1], color, 1)
                    
                    # Draw points
                    for point in points[:-1]:  # Skip last point which is duplicate of first
                        cv2.circle(frame, point, 1, color, -1)
        
        # Draw area being defined
        if self.is_defining_area and self.temp_points:
            color = self.area_colors.get(self.active_area_type, (255, 255, 255))
            
            # Draw lines between consecutive points
            for i in range(len(self.temp_points) - 1):
                cv2.line(frame, self.temp_points[i], self.temp_points[i + 1], color, 1)
            
            # Draw line to current mouse position
            if self.temp_point and len(self.temp_points) > 0:
                cv2.line(frame, self.temp_points[-1], self.temp_point, color, 1)
            
            # Draw points
            for point in self.temp_points:
                cv2.circle(frame, point, 1, color, -1)
        
        # Add instructions
        if self.is_defining_area:
            text = f"Defining {self.active_area_type.name} area - Left-click: Add point, Right-click: Complete, 'r': Reset"
            cv2.putText(frame, text, (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            text = f"Active area type: {self.active_area_type.name} - Press 1-7 to change, Left-click to start, 'd' to clear all"
            cv2.putText(frame, text, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _draw_arrow(self, frame, pt1, pt2, color, arrow_size=15):
        """Draw an arrow from pt1 to pt2"""
        # This function is disabled as per user request
        # No arrows will be drawn
        return
    
    def is_in_area(self, x, y, area_type=None):
        """
        Check if a point (x, y) is inside any area of the specified type
        If area_type is None, checks against the active area type
        """
        check_area_type = area_type if area_type is not None else self.active_area_type
        
        if check_area_type not in self.areas:
            return True  # If no areas defined, consider all points valid
        
        for area in self.areas[check_area_type]:
            if not area.get('enabled', True):
                continue
                
            points = area['points']
            if len(points) <= 2:
                continue  # Skip lines, only check polygons
                
            # Convert points to numpy array for pointPolygonTest
            points_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            
            # Check if point is inside polygon
            distance = cv2.pointPolygonTest(points_np, (float(x), float(y)), False)
            if distance >= 0:
                return True
        
        return False
    
    def is_box_in_area(self, box, area_type=None):        
        # Check if a bounding box (x1, y1, x2, y2) is at least partially inside any area of specified type
       
        check_area_type = area_type if area_type is not None else self.active_area_type
        
        if check_area_type not in self.areas or not self.areas[check_area_type]:
            return True  # If no areas defined, consider all boxes valid
        
        x1, y1, x2, y2 = box
        
        for area in self.areas[check_area_type]:
            if not area.get('enabled', True):
                continue
                
            points = area['points']
            if len(points) <= 2:
                continue  # Skip lines, only check polygons
            
            # Check if any corner of the box is inside the area
            corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
            for corner in corners:
                if self._is_point_in_polygon(corner, points):
                    return True
            
            # Check if the box contains any point of the area
            for point in points:
                if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                    return True
                    
            # Check if any edge of the box intersects with any edge of the area
            box_edges = [
                [(x1, y1), (x1, y2)],  # Left edge
                [(x1, y2), (x2, y2)],  # Bottom edge
                [(x2, y2), (x2, y1)],  # Right edge
                [(x2, y1), (x1, y1)]   # Top edge
            ]
            
            area_edges = []
            for i in range(len(points) - 1):
                area_edges.append([points[i], points[i+1]])
            
            for box_edge in box_edges:
                for area_edge in area_edges:
                    if self._line_segments_intersect(box_edge[0], box_edge[1], area_edge[0], area_edge[1]):
                        return True
        
        return False
    
    def _is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        
        for i in range(1, n):
            p2x, p2y = polygon[i]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments (p1-p2 and p3-p4) intersect"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
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
    
    def is_crossing_line(self, prev_point, curr_point, line_type=None):
        """
        Check if a moving object crossed any line of the specified type
        Returns: 
        - (True, line_id, direction) if crossed, where direction is 1 (positive) or -1 (negative)
        - (False, None, None) if not crossed
        """
        check_line_type = line_type if line_type is not None else AreaType.SPEED
        
        if check_line_type not in self.areas:
            return False, None, None
        
        for i, area in enumerate(self.areas[check_line_type]):
            if not area.get('enabled', True):
                continue
                
            points = area['points']
            if len(points) != 2:
                continue  # Only check lines with exactly 2 points
            
            # Check if the movement path intersects with the line
            if self._line_segments_intersect(prev_point, curr_point, points[0], points[1]):
                # Determine the direction of crossing
                # Create vectors
                line_vec = (points[1][0] - points[0][0], points[1][1] - points[0][1])
                movement_vec = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
                
                # Calculate cross product z-component to determine direction
                cross_product = line_vec[0] * movement_vec[1] - line_vec[1] * movement_vec[0]
                direction = 1 if cross_product > 0 else -1
                
                return True, i, direction
        
        return False, None, None
    
    def save_areas(self):
        """Save all area configurations to a file"""
        try:
            # Create directory if it doesn't exist
            if not os.path.exists('area_configs'):
                os.makedirs('area_configs')
                
            filename = f"area_configs/areas_{self.stream_id}.json"
            
            # Convert areas dict to a format that can be easily serialized
            serializable_areas = {}
            for area_type, areas in self.areas.items():
                serializable_areas[area_type.name] = areas
            
            with open(filename, 'w') as f:
                json.dump(serializable_areas, f, indent=2)
                
            print(f"[{self.stream_id}] Areas saved to {filename}")
            return True
        except Exception as e:
            print(f"[{self.stream_id}] Error saving areas: {str(e)}")
            return False
    
    def load_areas(self):
        """Load area configurations from a file"""
        try:
            filename = f"area_configs/areas_{self.stream_id}.json"
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    serialized_areas = json.load(f)
                
                # Convert loaded data back to our format
                for area_type_name, areas in serialized_areas.items():
                    try:
                        area_type = AreaType[area_type_name]
                        self.areas[area_type] = areas
                    except KeyError:
                        print(f"[{self.stream_id}] Unknown area type: {area_type_name}")
                
                print(f"[{self.stream_id}] Areas loaded from {filename}")
                return True
            else:
                print(f"[{self.stream_id}] No saved areas found")
                return False
        except Exception as e:
            print(f"[{self.stream_id}] Error loading areas: {str(e)}")
            return False
    
    def export_areas_for_api(self):
        """Export areas in a format suitable for a web API"""
        api_data = {
            "stream_id": self.stream_id,
            "areas": {}
        }
        
        for area_type, areas in self.areas.items():
            api_data["areas"][area_type.name] = []
            
            for i, area in enumerate(areas):
                api_area = {
                    "id": i,
                    "type": area_type.name,
                    "points": area['points'],
                    "enabled": area.get('enabled', True),
                    "properties": area.get('properties', {})
                }
                api_data["areas"][area_type.name].append(api_area)
        
        return api_data
    
    def import_areas_from_api(self, api_data):
        """Import areas from an API data structure"""
        if "areas" not in api_data:
            return False
            
        # Clear existing areas
        self.areas = {}
        
        for area_type_name, areas in api_data["areas"].items():
            try:
                area_type = AreaType[area_type_name]
                self.areas[area_type] = []
                
                for area in areas:
                    self.areas[area_type].append({
                        "points": area["points"],
                        "type": area_type_name,
                        "enabled": area.get("enabled", True),
                        "properties": area.get("properties", {})
                    })
            except KeyError:
                print(f"[{self.stream_id}] Unknown area type: {area_type_name}")
        
        self.save_areas()
        return True



    def clear_all_and_restart(self):
        """
        Completely clear all area configurations, delete the config file, and restart
        """
        try:
            # Clear all areas from memory
            self.areas = {}

            # Reset area manager state
            self.is_defining_area = False
            self.temp_points = []
            self.temp_point = None
            self.active_area_type = AreaType.DETECTION  # Reset to default

            # Delete the configuration file
            filename = f"area_configs/areas_{self.stream_id}.json"
            if os.path.exists(filename):
                os.remove(filename)
                print(f"[{self.stream_id}] Deleted area configuration file: {filename}")

            print(f"[{self.stream_id}] All area configurations cleared and restarted")
            return True

        except Exception as e:
            print(f"[{self.stream_id}] Error clearing and restarting areas: {str(e)}")
            return False
