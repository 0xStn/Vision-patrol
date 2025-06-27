# Vision Patrol - Violation Manager
import os
import csv
import json
import time
import uuid
from datetime import datetime
import cv2
from pathlib import Path
import threading
from .accident_alert_manager import AccidentAlertManager
from .License_recognition.license_plate_recognizer import LicensePlateRecognizer

class ViolationManager:
    """
    Centralized manager for traffic violations (excluding accidents).
    Handles logging, image storage, and reporting for Vision Patrol system.
    """
    
    def __init__(self, base_dir=None, stream_id="default", camera_location="Unknown", coordinates=None):
        self.stream_id = stream_id
        self.camera_location = camera_location
        
        # Initialize coordinates with default values if not provided
        if coordinates is None:
            self.coordinates = {"lat": 0.0, "lng": 0.0}
        elif isinstance(coordinates, dict) and "lat" in coordinates and "lng" in coordinates:
            # Ensure coordinate values are proper floats
            try:
                self.coordinates = {
                    "lat": float(coordinates["lat"]), 
                    "lng": float(coordinates["lng"])
                }
            except (ValueError, TypeError):
                print(f"[{stream_id}] Warning: Invalid coordinate values. Using defaults.")
                self.coordinates = {"lat": 0.0, "lng": 0.0}
        else:
            print(f"[{stream_id}] Warning: Coordinates not in expected format. Using defaults.")
            self.coordinates = {"lat": 0.0, "lng": 0.0}

        self.lock = threading.Lock()  # Thread-safe operations
        
        # Counter for unique violation IDs
        self.violation_counter = 0
        
        # Setup directories
        if base_dir is None:
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "violations"
        else:
            base_dir = Path(base_dir)
            
        self.base_dir = base_dir
        self._setup_directories()
        self._setup_logs()
        
        # Create accident alert manager for accident handling
        self.accident_alert_manager = AccidentAlertManager(
            stream_id=stream_id,
            camera_location=camera_location,
            coordinates=self.coordinates
        )
        
        # Initialize license plate recognizer
        self.license_plate_recognizer = LicensePlateRecognizer(base_dir=self.base_dir)
        
        print(f"[{self.stream_id}] Unified Violation Manager initialized for {camera_location}")
        
    def _setup_directories(self):
        """Create organized directory structure for violation snapshots"""
        # Main violations directory 
        self.base_dir.mkdir(exist_ok=True)
        
        # Create a single snapshots directory for all violation types
        self.snapshots_dir = self.base_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        # A single directory for all violation types with no date or camera subfolders
        self.violation_dirs = {}
        for violation_type in ["parking", "speed", "wrong_direction", 
                              "traffic_light", "helmet"]:
            # Instead of creating type-specific folders, just point all types to the same snapshots directory
            self.violation_dirs[violation_type] = self.snapshots_dir
        
        # Add plates directory - single directory with no date subfolders
        self.plates_dir = self.base_dir / "plates"
        self.plates_dir.mkdir(exist_ok=True)
        
    def _setup_logs(self):
        """Initialize centralized CSV and JSON log files"""
        # Use a single CSV file for all violations directly in violations folder
        self.csv_path = self.base_dir / "unified_violations.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Date', 'Time', 'License_Plate', 'Camera_ID', 
                    'Camera_Location', 'Violation_Type', 'Violation_ID',
                    'Vehicle_ID'
                ])
        
        # Use a single JSON file for all violations directly in violations folder
        self.json_path = self.base_dir / "unified_violations.json"
        if not self.json_path.exists():
            with open(self.json_path, 'w') as f:
                json.dump([], f)
    
    # Methods for recording different violation types
    
    def record_parking_violation(self, frame, vehicle_id, bbox, duration=None):
        """Record a parking violation"""
        return self._record_violation(frame, vehicle_id, bbox, "parking", 
                                     extra_info={"duration": duration})
    
    def record_speed_violation(self, frame, vehicle_id, bbox, speed=None):
        """Record a speed violation"""
        return self._record_violation(frame, vehicle_id, bbox, "speed", 
                                     extra_info={"speed": speed})
    
    def record_wrong_direction(self, frame, vehicle_id, bbox, line_pair=None):
        """Record a wrong direction violation"""
        return self._record_violation(frame, vehicle_id, bbox, "wrong_direction", 
                                     extra_info={"line_pair": line_pair})
    
    def record_traffic_violation(self, frame, vehicle_id, bbox, light_state=None):
        """Record a traffic light violation"""
        return self._record_violation(frame, vehicle_id, bbox, "traffic_light", 
                                     extra_info={"light_state": light_state})
    
    def record_accident(self, frame, vehicles_involved, bbox=None, accident_class=None):
        """
        Record a traffic accident - delegate to accident alert manager
        """
        # Just delegate to the accident alert manager
        return self.accident_alert_manager.record_accident(
            frame, vehicles_involved, bbox, accident_class
        )
    
    def record_helmet_violation(self, frame, vehicle_id, bbox, vehicle_type=None):
        """Record a helmet violation"""
        return self._record_violation(frame, vehicle_id, bbox, "helmet", 
                                     extra_info={"vehicle_type": vehicle_type})
    
    def _record_violation(self, frame, vehicle_id, bbox, violation_type,
                         extra_info=None, save_full_frame=False):
        """
        Core method to record any type of violation
        NOW DOES LICENSE PLATE RECOGNITION FIRST, THEN DECIDES WHERE TO SAVE
        Returns the violation ID and snapshot path
        """
        with self.lock:  # Thread-safe operation
            # Generate unique violation ID with counter and UUID to ensure uniqueness
            timestamp = datetime.now()
            date_str = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H:%M:%S')

            # Increment counter for each violation
            self.violation_counter += 1

            # Create simplified unique ID using counter and short UUID - no prefixes
            unique_id = f"{self.violation_counter:06d}_{uuid.uuid4().hex[:6]}"
            # Store the full violation type info in the record but use simple ID
            violation_id = unique_id

            # Save violation snapshot (only car closeup with wider margin for non-accidents)
            snapshot_path = self._save_violation_snapshot(
                frame, bbox, vehicle_id, violation_type, violation_id, save_full_frame)

            # Create violation record template
            violation_record = {
                'date': date_str,
                'time': time_str,
                'license_plate': '',  # Will be filled by recognition
                'camera_id': self.stream_id,
                'camera_location': self.camera_location,
                'location_coordinates': self.coordinates,
                'violation_type': violation_type,
                'violation_id': violation_id,
                'vehicle_id': vehicle_id,
                'snapshot_path': snapshot_path,
                'extra_info': extra_info or {}
            }

            print(f"[{self.stream_id}] Processing {violation_type} violation: {violation_id}")

            # DO LICENSE PLATE RECOGNITION FIRST (SYNCHRONOUSLY)
            try:
                plate_result = self.license_plate_recognizer.process_violation_sync(
                    violation_record, frame, bbox
                )

                if plate_result:
                    plate_text = plate_result.get('license_plate', '')
                    confidence = plate_result.get('plate_confidence', 0.0)
                    plate_snapshot_path = plate_result.get('plate_snapshot_path', '')

                    # Update violation record with plate info
                    violation_record['license_plate'] = plate_text
                    violation_record['plate_snapshot_path'] = plate_snapshot_path

                    # Decide where to save based on license plate recognition results
                    needs_manual_review = False
                    reason = ""

                    if not plate_text:
                        needs_manual_review = True
                        if confidence > 0.0:
                            reason = f"License plate detected (conf: {confidence:.2f}) but OCR failed to read text"
                        else:
                            reason = "License plate detection failed - no plate found"
                    elif confidence < 0.3:
                        needs_manual_review = True
                        reason = f"License plate detection confidence too low ({confidence:.2f} < 0.3)"

                    if needs_manual_review:
                        print(f"[{self.stream_id}] {reason} for {violation_id}, creating manual review")
                        self._create_manual_review_directly(violation_record, confidence, reason)
                    else:
                        print(f"[{self.stream_id}] Good license plate recognition for {violation_id}: {plate_text}")
                        # Save as normal violation
                        self._save_violation_to_database(violation_record)

                else:
                    # License plate recognition completely failed
                    print(f"[{self.stream_id}] License plate recognition failed completely for {violation_id}, creating manual review")
                    self._create_manual_review_directly(violation_record, 0.0, "License plate recognition system failure")

            except Exception as e:
                print(f"[{self.stream_id}] Error in license plate recognition for {violation_id}: {str(e)}")
                # If license plate processing fails, create manual review
                self._create_manual_review_directly(violation_record, 0.0, f"License plate processing error: {str(e)}")

            return violation_id, snapshot_path
    
    def _update_plate_in_json(self, updated_record):
        """
        Update an existing violation record in the JSON file with license plate info
        Called by the license plate recognizer when processing is complete
        """
        try:
            # Get the violation ID of the record to update
            violation_id = updated_record.get('violation_id')
            if not violation_id:
                print(f"[{self.stream_id}] Missing violation ID in update record")
                return

            # Check if license plate recognition failed or has low confidence
            plate_text = updated_record.get('license_plate', '')
            confidence = updated_record.get('plate_confidence', 0.0)

            # Determine if this should go to manual review
            # Case 1: No license plate text detected (OCR failed)
            # Case 2: License plate detection confidence is below 30%
            needs_manual_review = False
            reason = ""

            if not plate_text:
                needs_manual_review = True
                if confidence > 0.0:
                    reason = f"License plate detected (conf: {confidence:.2f}) but OCR failed to read text"
                else:
                    reason = "License plate detection failed - no plate found"
            elif confidence < 0.3:
                needs_manual_review = True
                reason = f"License plate detection confidence too low ({confidence:.2f} < 0.3)"

            if needs_manual_review:
                print(f"[{self.stream_id}] {reason} for {violation_id}, creating manual review instead")
                self._create_manual_review_for_failed_recognition(violation_id, updated_record, confidence, reason)

                # Remove the violation from violations database since it should be in manual review
                self._remove_violation_from_database(violation_id)
                return

            # If we have a good license plate (text exists and confidence >= 0.3), proceed with normal violation recording
            # Read existing data
            with open(self.json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[{self.stream_id}] Error decoding JSON file")
                    return

            # Find and update the specific record - keep confidence out of JSON
            updated = False
            for i, record in enumerate(data):
                if record.get('violation_id') == violation_id:
                    # Update license plate fields
                    record['license_plate'] = updated_record.get('license_plate', '')

                    # Ensure plate snapshot path is included in the record
                    if 'plate_snapshot_path' in updated_record:
                        record['plate_snapshot_path'] = updated_record.get('plate_snapshot_path', '')

                    # Remove confidence from JSON if it exists
                    if 'plate_confidence' in record:
                        del record['plate_confidence']

                    updated = True
                    break

            if not updated:
                print(f"[{self.stream_id}] Violation record {violation_id} not found for plate update")
                return

            # Write updated data with atomic operation
            temp_file = str(self.json_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Replace the original file with the temp file
            os.replace(temp_file, self.json_path)

            # Update CSV file - now includes confidence
            self._update_plate_in_csv(violation_id, updated_record.get('license_plate', ''),
                                     updated_record.get('plate_confidence', 0.0))

            print(f"[{self.stream_id}] Updated license plate in records for {violation_id}: {updated_record.get('license_plate', '')}")
            print(f"[{self.stream_id}] Plate image path: {updated_record.get('plate_snapshot_path', 'Not available')}")

        except Exception as e:
            print(f"[{self.stream_id}] Error updating license plate in JSON: {str(e)}")

    def _create_manual_review_for_violation(self, violation_id, updated_record, confidence):
        """
        Create a manual review entry for a violation with failed or low-confidence license plate recognition
        """
        try:
            # Determine reason based on the failure type
            plate_text = updated_record.get('license_plate', '')
            if not plate_text:
                reason = "License plate recognition failed - no text detected"
            else:
                reason = f"License plate recognition low confidence ({confidence:.2f}) - manual verification needed"

            # Get the full violation record
            violations_data = []
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    violations_data = json.load(f)

            violation_data = None
            for violation in violations_data:
                if violation.get('violation_id') == violation_id:
                    violation_data = violation
                    break

            if not violation_data:
                print(f"[{self.stream_id}] Could not find violation data for manual review creation: {violation_id}")
                return

            # Create manual review record
            review_id = f"MR_{uuid.uuid4().hex[:8]}"
            review_record = {
                'review_id': review_id,
                'violation_id': violation_id,
                'violation_data': violation_data,
                'reason': reason,
                'notes': f"Automatic creation due to license plate recognition failure. Confidence: {confidence:.2f}",
                'license_plate': None,
                'reviewer_name': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': None,
                'status': 'pending'
            }

            # Save manual review
            manual_reviews_path = os.path.join(self.base_dir, "manual_reviews.json")
            existing_reviews = []

            # Load existing reviews
            if os.path.exists(manual_reviews_path):
                try:
                    with open(manual_reviews_path, 'r') as f:
                        existing_reviews = json.load(f)
                except json.JSONDecodeError:
                    existing_reviews = []

            # Check if manual review already exists for this violation
            review_exists = any(review.get('violation_id') == violation_id for review in existing_reviews)
            if review_exists:
                print(f"[{self.stream_id}] Manual review already exists for violation: {violation_id}")
                return

            # Add new review
            existing_reviews.append(review_record)

            # Save to file
            temp_file = manual_reviews_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(existing_reviews, f, indent=2)

            os.replace(temp_file, manual_reviews_path)

            print(f"[{self.stream_id}] Created manual review {review_id} for violation {violation_id}")

        except Exception as e:
            print(f"[{self.stream_id}] Error creating manual review: {str(e)}")

    def _create_manual_review_for_failed_recognition(self, violation_id, updated_record, confidence, reason=None):
        """
        Create a manual review entry for a violation with failed license plate recognition
        This is called when license plate recognition fails completely
        """
        try:
            # Use provided reason or determine based on the failure type
            if reason is None:
                plate_text = updated_record.get('license_plate', '')
                if not plate_text:
                    if confidence > 0.0:
                        reason = f"License plate detected (conf: {confidence:.2f}) but OCR failed to read text"
                    else:
                        reason = "License plate detection failed - no plate found"
                else:
                    reason = f"License plate detection confidence too low ({confidence:.2f} < 0.3)"

            # Get the violation record that was temporarily stored
            violation_data = self._get_temp_violation_data(violation_id)

            if not violation_data:
                print(f"[{self.stream_id}] Could not find violation data for manual review creation: {violation_id}")
                return

            # Add plate snapshot path to violation data
            if 'plate_snapshot_path' in updated_record:
                violation_data['plate_snapshot_path'] = updated_record.get('plate_snapshot_path', '')

            # Create manual review record
            review_id = f"MR_{uuid.uuid4().hex[:8]}"
            review_record = {
                'review_id': review_id,
                'violation_id': violation_id,
                'violation_data': violation_data,
                'reason': reason,
                'notes': f"Automatic creation: {reason}. Detection confidence: {confidence:.2f}",
                'license_plate': None,
                'reviewer_name': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': None,
                'status': 'pending'
            }

            # Save manual review
            manual_reviews_path = os.path.join(self.base_dir, "manual_reviews.json")
            existing_reviews = []

            # Load existing reviews
            if os.path.exists(manual_reviews_path):
                try:
                    with open(manual_reviews_path, 'r') as f:
                        existing_reviews = json.load(f)
                except json.JSONDecodeError:
                    existing_reviews = []

            # Check if manual review already exists for this violation
            review_exists = any(review.get('violation_id') == violation_id for review in existing_reviews)
            if review_exists:
                print(f"[{self.stream_id}] Manual review already exists for violation: {violation_id}")
                return

            # Add new review
            existing_reviews.append(review_record)

            # Save to file
            temp_file = manual_reviews_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(existing_reviews, f, indent=2)

            os.replace(temp_file, manual_reviews_path)

            print(f"[{self.stream_id}] Created manual review {review_id} for failed recognition {violation_id}")

        except Exception as e:
            print(f"[{self.stream_id}] Error creating manual review for failed recognition: {str(e)}")

    def cleanup_empty_license_plates(self):
        """
        Cleanup function to move any violations with empty license plates to manual review
        This catches cases where the async license plate processing failed or didn't complete
        """
        try:
            if not os.path.exists(self.json_path):
                return

            with open(self.json_path, 'r') as f:
                violations = json.load(f)

            violations_to_move = []

            # Find violations with empty license plates
            for violation in violations:
                plate_text = violation.get('license_plate', '')
                if not plate_text:
                    violations_to_move.append(violation)

            if violations_to_move:
                print(f"[{self.stream_id}] Found {len(violations_to_move)} violations with empty license plates to move to manual review")

                for violation in violations_to_move:
                    violation_id = violation.get('violation_id')
                    print(f"[{self.stream_id}] Moving violation {violation_id} to manual review (cleanup)")

                    # Create manual review
                    self._create_manual_review_for_failed_recognition(
                        violation_id,
                        {'license_plate': '', 'plate_confidence': 0.0},
                        0.0,
                        "License plate processing incomplete - moved during cleanup"
                    )

                    # Remove from violations database
                    self._remove_violation_from_database(violation_id)

        except Exception as e:
            print(f"[{self.stream_id}] Error during cleanup of empty license plates: {str(e)}")

    def _create_manual_review_directly(self, violation_record, confidence, reason):
        """
        Create a manual review directly from violation record (no database save first)
        This is the new approach - decide before saving to database
        """
        try:
            violation_id = violation_record.get('violation_id')

            # Create manual review record
            review_id = f"MR_{uuid.uuid4().hex[:8]}"
            review_record = {
                'review_id': review_id,
                'violation_id': violation_id,
                'violation_data': violation_record,
                'reason': reason,
                'notes': f"Automatic creation: {reason}. Detection confidence: {confidence:.2f}",
                'license_plate': None,
                'reviewer_name': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': None,
                'status': 'pending'
            }

            # Save manual review
            manual_reviews_path = os.path.join(self.base_dir, "manual_reviews.json")
            existing_reviews = []

            # Load existing reviews
            if os.path.exists(manual_reviews_path):
                try:
                    with open(manual_reviews_path, 'r') as f:
                        existing_reviews = json.load(f)
                except json.JSONDecodeError:
                    existing_reviews = []

            # Check if manual review already exists for this violation
            review_exists = any(review.get('violation_id') == violation_id for review in existing_reviews)
            if review_exists:
                print(f"[{self.stream_id}] Manual review already exists for violation: {violation_id}")
                return

            # Add new review
            existing_reviews.append(review_record)

            # Save to file
            temp_file = manual_reviews_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(existing_reviews, f, indent=2)

            os.replace(temp_file, manual_reviews_path)

            print(f"[{self.stream_id}] Created manual review {review_id} for {violation_id} (direct)")

        except Exception as e:
            print(f"[{self.stream_id}] Error creating manual review directly: {str(e)}")

    def _save_violation_to_database(self, violation_record):
        """
        Save a violation record to the database (JSON and CSV)
        This is called only for violations with good license plate recognition
        """
        try:
            # Update CSV log
            self._update_csv_log(violation_record)

            # Update JSON log
            self._update_json_log(violation_record)

            violation_id = violation_record.get('violation_id')
            plate_text = violation_record.get('license_plate', '')
            print(f"[{self.stream_id}] Saved violation {violation_id} to database with license plate: {plate_text}")

        except Exception as e:
            print(f"[{self.stream_id}] Error saving violation to database: {str(e)}")

    def _get_temp_violation_data(self, violation_id):
        """
        Get violation data that was temporarily stored before license plate processing
        """
        try:
            # Check if violation exists in current JSON (it might have been added temporarily)
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    violations = json.load(f)

                for violation in violations:
                    if violation.get('violation_id') == violation_id:
                        return violation

            # If not found, we need to reconstruct it from available data
            # This shouldn't happen in normal flow, but just in case
            print(f"[{self.stream_id}] Warning: Could not find temp violation data for {violation_id}")
            return None

        except Exception as e:
            print(f"[{self.stream_id}] Error getting temp violation data: {str(e)}")
            return None

    def _remove_violation_from_database(self, violation_id):
        """
        Remove a violation from the violations database (JSON and CSV)
        This is called when a violation should go to manual review instead
        """
        try:
            # Remove from JSON
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    violations = json.load(f)

                # Filter out the violation
                original_count = len(violations)
                violations = [v for v in violations if v.get('violation_id') != violation_id]

                if len(violations) < original_count:
                    # Save updated violations
                    temp_file = str(self.json_path) + '.tmp'
                    with open(temp_file, 'w') as f:
                        json.dump(violations, f, indent=2)

                    os.replace(temp_file, self.json_path)
                    print(f"[{self.stream_id}] Removed violation {violation_id} from JSON database")

            # Remove from CSV
            if os.path.exists(self.csv_path):
                rows = []
                with open(self.csv_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                # Filter out the violation (violation_id is in column 6)
                original_count = len(rows)
                filtered_rows = [rows[0]]  # Keep header

                for i, row in enumerate(rows):
                    if i > 0:  # Skip header
                        if len(row) > 6 and row[6] != violation_id:
                            filtered_rows.append(row)

                if len(filtered_rows) < original_count:
                    # Save updated CSV
                    temp_file = str(self.csv_path) + '.tmp'
                    with open(temp_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(filtered_rows)

                    os.replace(temp_file, self.csv_path)
                    print(f"[{self.stream_id}] Removed violation {violation_id} from CSV database")

        except Exception as e:
            print(f"[{self.stream_id}] Error removing violation from database: {str(e)}")

    def get_manual_reviews(self, filter_status=None):
        """
        Get all manual reviews for this stream, optionally filtered by status
        """
        try:
            manual_reviews_path = os.path.join(self.base_dir, "manual_reviews.json")

            if not os.path.exists(manual_reviews_path):
                return []

            with open(manual_reviews_path, 'r') as f:
                reviews = json.load(f)

            # Filter by stream_id and optionally by status
            filtered_reviews = []
            for review in reviews:
                violation_data = review.get('violation_data', {})
                if violation_data.get('camera_id') == self.stream_id:
                    if filter_status is None or review.get('status') == filter_status:
                        filtered_reviews.append(review)

            return filtered_reviews

        except Exception as e:
            print(f"[{self.stream_id}] Error getting manual reviews: {str(e)}")
            return []

    def update_manual_review(self, review_id, license_plate=None, notes=None, reviewer_name=None):
        """
        Update a manual review with corrected license plate information
        """
        try:
            manual_reviews_path = os.path.join(self.base_dir, "manual_reviews.json")

            if not os.path.exists(manual_reviews_path):
                return False, "Manual reviews file not found"

            with open(manual_reviews_path, 'r') as f:
                reviews = json.load(f)

            # Find and update the review
            review_found = False
            for review in reviews:
                if review.get('review_id') == review_id:
                    review_found = True

                    # Update fields
                    if license_plate is not None:
                        review['license_plate'] = license_plate
                    if notes is not None:
                        review['notes'] = notes
                    if reviewer_name is not None:
                        review['reviewer_name'] = reviewer_name

                    review['updated_at'] = datetime.now().isoformat()

                    # Update status to completed if license plate is provided
                    if license_plate:
                        review['status'] = 'completed'

                        # Add the violation to the database with the corrected license plate
                        violation_id = review.get('violation_id')
                        if violation_id:
                            self._add_violation_from_manual_review(review, license_plate)

                    break

            if not review_found:
                return False, "Manual review not found"

            # Save updated reviews
            temp_file = manual_reviews_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(reviews, f, indent=2)

            os.replace(temp_file, manual_reviews_path)

            print(f"[{self.stream_id}] Updated manual review {review_id}")
            return True, None

        except Exception as e:
            print(f"[{self.stream_id}] Error updating manual review: {str(e)}")
            return False, f"Error updating manual review: {str(e)}"

    def _add_violation_from_manual_review(self, review, license_plate):
        """
        Add a violation to the database from a completed manual review
        This creates a proper violation record with the manually corrected license plate
        """
        try:
            violation_data = review.get('violation_data', {})
            violation_id = review.get('violation_id')

            # Update violation data with corrected license plate
            violation_data['license_plate'] = license_plate
            violation_data['manual_review_completed'] = True
            violation_data['manual_review_date'] = datetime.now().isoformat()

            # Add to JSON file
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    violations = json.load(f)
            else:
                violations = []

            # Check if violation already exists (shouldn't happen, but just in case)
            violation_exists = any(v.get('violation_id') == violation_id for v in violations)
            if not violation_exists:
                violations.append(violation_data)

                # Save updated violations
                temp_file = str(self.json_path) + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(violations, f, indent=2)

                os.replace(temp_file, self.json_path)
                print(f"[{self.stream_id}] Added violation {violation_id} to JSON database with license plate: {license_plate}")

            # Add to CSV file
            self._add_violation_to_csv(violation_data)

        except Exception as e:
            print(f"[{self.stream_id}] Error adding violation from manual review: {str(e)}")

    def _add_violation_to_csv(self, violation_record):
        """Add a violation record to the CSV file"""
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    violation_record.get('date', ''),
                    violation_record.get('time', ''),
                    violation_record.get('license_plate', ''),
                    violation_record.get('camera_id', ''),
                    violation_record.get('camera_location', ''),
                    violation_record.get('violation_type', ''),
                    violation_record.get('violation_id', ''),
                    violation_record.get('vehicle_id', '')  # This might be missing, that's ok
                ])
            print(f"[{self.stream_id}] Added violation {violation_record.get('violation_id')} to CSV database")
        except Exception as e:
            print(f"[{self.stream_id}] Error adding violation to CSV: {str(e)}")

    def delete_manual_review(self, review_id):
        """
        Delete a manual review
        """
        try:
            manual_reviews_path = os.path.join(self.base_dir, "manual_reviews.json")

            if not os.path.exists(manual_reviews_path):
                return False, "Manual reviews file not found"

            with open(manual_reviews_path, 'r') as f:
                reviews = json.load(f)

            # Remove the review
            original_count = len(reviews)
            reviews = [review for review in reviews if review.get('review_id') != review_id]

            if len(reviews) == original_count:
                return False, "Manual review not found"

            # Save updated reviews
            temp_file = manual_reviews_path + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(reviews, f, indent=2)

            os.replace(temp_file, manual_reviews_path)

            print(f"[{self.stream_id}] Deleted manual review {review_id}")
            return True, None

        except Exception as e:
            print(f"[{self.stream_id}] Error deleting manual review: {str(e)}")
            return False, f"Error deleting manual review: {str(e)}"
    
    def _update_plate_in_csv(self, violation_id, plate_text, confidence=0.0):
        """
        Update the CSV file with the license plate text and confidence
        Note: This is a simple implementation - a production system might use a database
        """
        try:
            # Read all CSV data
            rows = []
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Check if confidence column exists in header
            header = rows[0]
            has_confidence_col = 'Plate_Confidence' in header
            
            # Add confidence column if it doesn't exist
            confidence_idx = -1
            if not has_confidence_col:
                header.append('Plate_Confidence')
                confidence_idx = len(header) - 1
                rows[0] = header
            else:
                confidence_idx = header.index('Plate_Confidence')
            
            # Find and update the record (search for violation ID in column 6)
            updated = False
            for i, row in enumerate(rows):
                if i > 0 and len(row) > 6 and row[6] == violation_id:
                    row[2] = plate_text  # License plate is in column 3 (index 2)
                    
                    # Make sure row is long enough for confidence
                    while len(row) <= confidence_idx:
                        row.append('')
                    
                    # Update confidence
                    row[confidence_idx] = f"{confidence:.4f}"
                    updated = True
                    break
            
            if updated:
                # Write back the updated data
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
        
        except Exception as e:
            print(f"[{self.stream_id}] Error updating license plate in CSV: {str(e)}")
    
    def _save_violation_snapshot(self, frame, bbox, vehicle_id, violation_type, violation_id, save_full_frame=False):
        """Save a snapshot of the violation - focused on vehicle with wider margin"""
        try:
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime('%H%M%S')
            # All violations now go to the same snapshots directory
            
            # For violations where we want full frame, save it
            if save_full_frame:
                full_path = self.snapshots_dir / f"{violation_type}_full_{violation_id}_{timestamp_str}.jpg"
                cv2.imwrite(str(full_path), frame)
                return str(full_path.relative_to(self.base_dir))
            
            # For all other violations, save the cropped vehicle with reduced margin
            if bbox is not None:
                # Extract vehicle with margin
                x1, y1, x2, y2 = bbox
                # Calculate expanded box with 45% margin (reduced by 15% from previous 60%)
                width, height = x2 - x1, y2 - y1
                margin_x, margin_y = int(width * 0.45), int(height * 0.45)
                
                # Ensure expanded box stays within frame bounds
                h, w = frame.shape[:2]
                ex1 = max(0, x1 - margin_x)
                ey1 = max(0, y1 - margin_y)
                ex2 = min(w-1, x2 + margin_x)
                ey2 = min(h-1, y2 + margin_y)
                
                if ex1 < ex2 and ey1 < ey2:  # Valid box
                    vehicle_img = frame[ey1:ey2, ex1:ex2]
                    
                    # Add violation information to image with thinner font
                    # Use violation ID for simpler display
                    violation_text = f"{violation_type.upper()} VIOLATION - ID: {violation_id}"
                    cv2.putText(vehicle_img, violation_text, 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Font thickness set to 1
                    
                    # Save vehicle closeup image - use simplified ID in filename
                    # All violation types go into the same directory now
                    closeup_path = self.snapshots_dir / f"{violation_type}_vehicle_{violation_id}_{timestamp_str}.jpg"
                    cv2.imwrite(str(closeup_path), vehicle_img)
                    return str(closeup_path.relative_to(self.base_dir))
        
            # Fallback if bbox is invalid - save full frame
            full_path = self.snapshots_dir / f"{violation_type}_fallback_{violation_id}_{timestamp_str}.jpg"
            cv2.imwrite(str(full_path), frame)
            return str(full_path.relative_to(self.base_dir))
        
        except Exception as e:
            print(f"[{self.stream_id}] Error saving violation snapshot: {str(e)}")
            # Return a placeholder path if we couldn't save the image
            return f"error_snapshot_{violation_type}_{violation_id}_{timestamp_str}"
        
    def _update_csv_log(self, violation_record):
        """Update the CSV log with new violation"""
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    violation_record['date'],
                    violation_record['time'],
                    violation_record['license_plate'],
                    violation_record['camera_id'],
                    violation_record['camera_location'],
                    violation_record['violation_type'],
                    violation_record['violation_id'],
                    violation_record['vehicle_id']
                    # Coordinates are only added to JSON, not CSV
                ])
        except Exception as e:
            print(f"[{self.stream_id}] Error updating CSV log: {str(e)}")
    
    def _update_json_log(self, violation_record):
        """Update the unified JSON log with new violation and database"""
        try:
            # Original JSON file logic (keep for backward compatibility)
            with open(self.json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
            
            # Create a copy of the record without vehicle_id for JSON output
            sanitized_record = violation_record.copy()
            if 'vehicle_id' in sanitized_record:
                del sanitized_record['vehicle_id']
            
            # Add the sanitized record
            data.append(sanitized_record)
            
            # Write updated data with atomic operation
            temp_file = str(self.json_path) + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Replace the original file with the temp file
            os.replace(temp_file, self.json_path)
            
            # # DATABASE INTEGRATION: Save to Django database if available
            # try:
            #     # Check if Django is set up
            #     import django
            #     if django.conf.settings.configured:
            #         from django.contrib.gis.geos import Point
            #         from traffic_app.models import Violation
                    
            #         # Check if violation already exists
            #         violation_id = violation_record.get('violation_id')
            #         if not Violation.objects.filter(violation_id=violation_id).exists():
            #             # Create Django model instance
            #             violation = Violation(
            #                 violation_id=violation_id,
            #                 date=violation_record.get('date'),
            #                 time=violation_record.get('time'),
            #                 license_plate=violation_record.get('license_plate', ''),
            #                 plate_confidence=violation_record.get('plate_confidence', 0.0),
            #                 camera_id=violation_record.get('camera_id'),
            #                 camera_location=violation_record.get('camera_location'),
            #                 violation_type=violation_record.get('violation_type'),
            #                 snapshot_path=violation_record.get('snapshot_path', ''),
            #                 plate_snapshot_path=violation_record.get('plate_snapshot_path', '')
            #             )
                        
            #             # Add coordinates if available
            #             if 'coordinates' in violation_record and violation_record['coordinates']:
            #                 lat = violation_record['coordinates'].get('lat', 0)
            #                 lng = violation_record['coordinates'].get('lng', 0)
            #                 violation.coordinates = Point(lng, lat, srid=4326)
                            
            #             # Save to database
            #             violation.save()
            #             print(f"[{self.stream_id}] Added violation {violation_id} to SQL Server database")
            # except (ImportError, AttributeError):
            #     # Django not available or not configured, skip database integration
            #     pass
            
        except Exception as e:
            print(f"[{self.stream_id}] Error updating JSON log: {str(e)}")
    
    def set_camera_coordinates(self, latitude, longitude):
        """Update the camera coordinates for mapping"""
        try:
            lat = float(latitude)
            lng = float(longitude)
            
            # Log the coordinate change
            print(f"[{self.stream_id}] Camera coordinates updated: {lat}, {lng}")
            
            self.coordinates = {"lat": lat, "lng": lng}
            
            # Also update coordinates in accident alert manager
            if hasattr(self, 'accident_alert_manager'):
                self.accident_alert_manager.set_camera_coordinates(latitude, longitude)
        except (ValueError, TypeError) as e:
            print(f"[{self.stream_id}] Invalid coordinates provided ({e}), using defaults")
            self.coordinates = {"lat": 0.0, "lng": 0.0}

        return self.coordinates
