import os
import pandas as pd
from datetime import datetime
import numpy as np
import torch
from time import time
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class ObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.saved_ids = []
        self.classwise_counts = {}
        self.region_initialized = False
        self.spd = {}
        self.trkd_ids = []
        self.trk_pt = {}
        self.trk_pp = {}
        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)

        # Initialize CSV data storage
        self.csv_filename = self.get_daily_filename()
        self.create_csv()

    def get_daily_filename(self):
        """Generate a filename based on the current date."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"vehicle_count_data_{current_date}.csv"
        return filename

    def create_csv(self):
        """Create the CSV file with proper headers if it doesn't exist."""
        if not os.path.exists(self.csv_filename):
            header = ["Track ID", "Label", "Action", "Speed (km/h)", "Class", "Date", "Time"]
            df = pd.DataFrame(columns=header)
            df.to_csv(self.csv_filename, index=False)
            print(f"CSV file created: {self.csv_filename} with headers.")

    def save_label_to_file(self, track_id, label, action, speed, class_name):
        """Save the label, track_id, action, speed, date, time, and class name to a CSV file in a table format."""
        # If speed is a tensor, convert it to a scalar
        if isinstance(speed, torch.Tensor):  # If speed is a PyTorch tensor
            speed = speed.item()
        elif isinstance(speed, np.ndarray):  # If speed is a numpy array
            speed = speed.item()

        # Ensure speed is a float and round it to an integer
        speed = int(round(speed))

        current_time = datetime.now()
        current_date = current_time.date()
        current_time_str = current_time.strftime("%H:%M:%S")

        # Prepare data to save to CSV
        data = {
            "Track ID": track_id,
            "Label": label,
            "Action": action,
            "Speed (km/h)": speed,
            "Class": class_name,
            "Date": current_date,
            "Time": current_time_str
        }

        # Convert to a DataFrame for easier appending
        df = pd.DataFrame([data])

        # Append data to the CSV file
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)
        print(f"Data for track_id {track_id} saved to CSV file {self.csv_filename}.")

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """Count objects and update file based on centroid movements."""
        if prev_position is None or track_id in self.counted_ids:
            return

        action = None
        speed = None  # Initialize speed

        # Handle linear region counting
        if len(self.region) == 2:
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    if current_centroid[0] > prev_position[0]:
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                        action = "IN"
                    else:
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                        action = "OUT"
                else:
                    if current_centroid[1] > prev_position[1]:
                        self.in_count += 1
                        self.classwise_counts[self.names[cls]]["IN"] += 1
                        action = "IN"
                    else:
                        self.out_count += 1
                        self.classwise_counts[self.names[cls]]["OUT"] += 1
                        action = "OUT"
                self.counted_ids.append(track_id)

        # Handle polygonal region counting
        elif len(self.region) > 2:
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                if current_centroid[0] > prev_position[0]:
                    self.in_count += 1
                    self.classwise_counts[self.names[cls]]["IN"] += 1
                    action = "IN"
                else:
                    self.out_count += 1
                    self.classwise_counts[self.names[cls]]["OUT"] += 1
                    action = "OUT"
                self.counted_ids.append(track_id)

        if action:
            label = f"{self.names[cls]} ID: {track_id}"
            speed = self.spd.get(track_id, 0)  # Get speed if available, else 0
            self.save_label_to_file(track_id, label, action, speed, self.names[cls])

    def store_classwise_counts(self, cls):
        """Initialize count dictionary for a given class."""
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        """Display the counts and actions on the image."""
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
                                  f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

        for track_id in self.track_ids:
            if track_id in self.counted_ids:
                in_count = self.in_count
                label = f"ID:{track_id} count at number {in_count}"

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            track_index = self.track_ids.index(track_id)
            cls = self.clss[track_index]
            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            combine_label = f"{speed_label}, ID: {track_id}"
            self.annotator.box_label(self.boxes[self.track_ids.index(track_id)], label=combine_label, color=(255, 0, 0))

    def count(self, im0):
        """Main counting function to track objects and store counts in the file."""
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)
        self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)
            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            speed_label = f"{int(self.spd[track_id])} km/h" if track_id in self.spd else self.names[int(cls)]
            self.annotator.draw_centroid_and_tracks(self.track_line, color=colors(int(track_id), True), track_thickness=self.line_width)
            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                direction = "known"
            else:
                direction = "unknown"
            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    self.spd[track_id] = np.abs(self.track_line[-1][1] - self.trk_pp[track_id][1]) / time_difference

            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]
            current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
            self.count_objects(current_centroid, track_id, prev_position, cls)

        self.display_counts(im0)
        return im0
