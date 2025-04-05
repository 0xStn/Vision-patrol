# main.py
import cv2
from ultralytics import YOLO
from wrong_det import detect_wrong_direction

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video file
cap = cv2.VideoCapture('Wrongtest.mp4')

# Track the status of cars
car_status = {}
wrong_way_cars = set()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for consistency
    frame = cv2.resize(frame, (1280, 720))

    # Perform YOLO tracking on the current frame
    results = model.track(source=frame, persist=True)

    # Detect wrong-way movement and update the frame
    frame, wrong_way_count = detect_wrong_direction(frame, results, car_status, wrong_way_cars)
    
    # Show the frame
    cv2.imshow("Wrong Direction Detection", frame)

    # Exit when the 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
