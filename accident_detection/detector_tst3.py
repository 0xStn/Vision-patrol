import cv2
import numpy as np
import time
import threading
import logging
from datetime import datetime
from ultralytics import YOLO
import winsound
import os

class AccidentDetector:
    def __init__(self, model_path, conf_threshold=0.88, cooldown=15, frame_skip=3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.cooldown = cooldown  # 15-second alert cooldown
        self.frame_skip = frame_skip
        self.accident_classes = [
            'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
            'car_bike_accident', 'car_car_accident', 'car_object_accident', 'car_person_accident'
        ]
        
        # State management
        self.last_alert_time = 0
        self.snapshots_taken = 0
        self.alert_active = False
        self.save_snapshots = False  # Flag to enable saving snapshots
        self.frame_counter_after_alert = 0  # Counter for frames after alert

        # Setup Logging
        logging.basicConfig(filename="accident_log.txt", level=logging.INFO, 
                          format="%(asctime)s - %(levelname)s - %(message)s")

    def process_video(self, video_path, output_dir="accidents"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        os.makedirs(output_dir, exist_ok=True)
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % self.frame_skip != 0:
                continue

            results = self.model(frame)[0]
            self._process_frame(frame, results, output_dir)

            cv2.imshow('Accident Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame, results, output_dir):
        current_time = time.time()
        
        # Reset cooldown if expired
        if self.alert_active and (current_time - self.last_alert_time >= self.cooldown):
            self.alert_active = False
            self.save_snapshots = False
            self.snapshots_taken = 0
            self.frame_counter_after_alert = 0
            print("Cooldown ended. Ready for new alerts.")

        # Save every alternate frame after alert
        if self.save_snapshots and self.snapshots_taken < 5:
            if self.frame_counter_after_alert % 2 == 0:  # Save every alternate frame
                self._save_accident(frame, output_dir)
                self.snapshots_taken += 1
                print(f"Saved snapshot {self.snapshots_taken}/5")
            self.frame_counter_after_alert += 1

            if self.snapshots_taken >= 5:
                self.save_snapshots = False  # Stop saving after 5 snapshots

        # Check for accidents
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            class_name = results.names[cls]

            if class_name in self.accident_classes and conf >= self.conf_threshold:
                self._handle_accident(frame, output_dir, class_name, conf, current_time)

    def _handle_accident(self, frame, output_dir, class_name, confidence, current_time):
        if not self.alert_active:
            # New accident detected - start cooldown period
            self.alert_active = True
            self.last_alert_time = current_time
            self.save_snapshots = True  # Enable saving snapshots
            self.snapshots_taken = 0  # Reset snapshot counter
            self.frame_counter_after_alert = 0  # Reset frame counter
            self._trigger_alert()
            self._log_detection(class_name, confidence)
            print("New alert triggered. Saving 5 alternate frames.")

    def _draw_alert_box(self, frame):
        cv2.rectangle(frame, (20, 20), (600, 100), (0, 0, 255), -1)
        cv2.putText(frame, "ACCIDENT DETECTED!", (50, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    def _save_accident(self, frame, output_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"accident_{timestamp}_{self.snapshots_taken}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved snapshot {self.snapshots_taken + 1}/5")

    def _trigger_alert(self):
        threading.Thread(target=winsound.Beep, args=(1000, 500)).start()

    def _log_detection(self, class_name, confidence):
        logging.info(f"Accident detected: {class_name} with confidence {confidence:.2f}")

if __name__ == "__main__":
    detector = AccidentDetector("accident.pt")  
    detector.process_video("accident6.mp4")