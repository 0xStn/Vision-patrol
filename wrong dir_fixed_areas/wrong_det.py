# wrong_direction.py
import cv2
import numpy as np
import cvzone
import os
from datetime import datetime

# Define vehicle classes (COCO class IDs for car, motorcycle, bus, truck)
VEHICLES = [2, 3, 5, 7]

# Define areas to monitor for wrong direction movement
AREA1 = [[661, 170], [923, 118], [1073, 133], [884, 211]]
AREA2 = [[1112, 242], [1098, 300], [1234, 382], [1230, 483], [692, 310], [938, 201]]

# Directory to save wrong-way car images
SAVE_DIR = "wrong_way_cars"
os.makedirs(SAVE_DIR, exist_ok=True)

def detect_wrong_direction(frame, results, car_status, wrong_way_cars):
    wrong_way_count = len(wrong_way_cars)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()    # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()   # Class IDs
        ids = result.boxes.id.cpu().numpy()        # Object IDs (tracking IDs)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            obj_class = int(classes[i])
            obj_id = int(ids[i])

            # Check if the detected object is a vehicle
            if obj_class in VEHICLES:
                cx = (x1 + x2) // 2  # Center X of the bounding box
                cy = y2              # Bottom Y of the bounding box

                # Check if the vehicle is in AREA1 or AREA2
                in_area1 = cv2.pointPolygonTest(np.array(AREA1, np.int32), (cx, cy), False) >= 0
                in_area2 = cv2.pointPolygonTest(np.array(AREA2, np.int32), (cx, cy), False) >= 0

                if obj_id not in car_status:
                    car_status[obj_id] = {'in_area1': False, 'in_area2': False, 'wrong_way': False, 'saves': False}

                # Update car status based on areas
                if in_area1:
                    car_status[obj_id]['in_area1'] = True
                if in_area2:
                    car_status[obj_id]['in_area2'] = True

                # Detect wrong-way movement
                if car_status[obj_id]['in_area1'] and in_area2 and not car_status[obj_id]['wrong_way']:
                    car_status[obj_id]['wrong_way'] = True

                    # Save image of wrong-way car
                    if not car_status[obj_id]['saves']:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        car_image_path = os.path.join(SAVE_DIR, f'car_{obj_id}_{timestamp}.png')
                        car_image = frame[y1:y2, x1:x2]
                        cv2.imwrite(car_image_path, car_image)
                        car_status[obj_id]['saves'] = True
                        wrong_way_cars.add(obj_id)

                # Draw bounding box and wrong-way label if applicable
                cv2.rectangle(frame, (x1, y1), (x2, y2), (144, 238, 144), 1, cv2.LINE_AA)
                if car_status[obj_id]['wrong_way']:
                    cvzone.putTextRect(frame, f'Wrong Way {obj_id}', (x1, y1 - 20), 1, 1, colorR=(0, 0, 255))

    # Visualize monitored areas
    cv2.polylines(frame, [np.array(AREA1, np.int32)], True, (255, 255, 255), 1)
    cv2.polylines(frame, [np.array(AREA2, np.int32)], True, (255, 255, 255), 1)

    return frame, wrong_way_count
