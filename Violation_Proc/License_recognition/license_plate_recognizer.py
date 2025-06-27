import os
import time
import threading
import queue
import cv2
from pathlib import Path
from datetime import datetime
from .license_plate_detector import LicensePlateDetector

class LicensePlateRecognizer:
    """
    Manages asynchronous license plate recognition for violation snapshots.
    """
    
    def __init__(self, base_dir=None, max_queue_size=100):
        # Initialize license plate detector
        self.detector = LicensePlateDetector()
        
        # Setup directories
        if base_dir is None:
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "violations"
        else:
            base_dir = Path(base_dir)
        
        self.base_dir = base_dir
        self.plates_dir = base_dir / "plates"
        self.plates_dir.mkdir(exist_ok=True)
        
        # Processing queue for async operation
        self.processing_queue = queue.Queue(maxsize=max_queue_size)
        
        # Start worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        print(f"License plate recognizer initialized, processing thread started")
    
    def process_violation(self, violation_record, frame, bbox, json_updater_callback):
        """
        Queue a violation for license plate processing
        
        Args:
            violation_record: Dict containing the violation record
            frame: The frame image containing the vehicle
            bbox: Vehicle bounding box
            json_updater_callback: Function to call to update the JSON record
        """
        try:
            # Create a request object for the queue
            request = {
                'violation_record': violation_record.copy(),
                'frame': frame.copy(),
                'bbox': bbox,
                'callback': json_updater_callback,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to add to the queue, but don't block if full
            try:
                self.processing_queue.put(request, block=False)
                print(f"Queued license plate recognition for violation {violation_record.get('violation_id')}")
            except queue.Full:
                print(f"License plate processing queue is full, skipping violation {violation_record.get('violation_id')}")
                
        except Exception as e:
            print(f"Error queueing license plate recognition: {str(e)}")

    def process_violation_sync(self, violation_record, frame, bbox):
        """
        Process license plate recognition synchronously (blocking)
        Returns the updated violation record with plate info or None if failed

        Args:
            violation_record: Dict containing the violation record
            frame: The frame image containing the vehicle
            bbox: Vehicle bounding box

        Returns:
            Dict with license plate info or None if processing failed
        """
        try:
            violation_id = violation_record.get('violation_id', 'unknown')
            print(f"Processing license plate synchronously for violation {violation_id}")

            # Process the license plate
            plate_text, plate_bbox, plate_img, confidence = self.detector.detect_license_plate(frame, bbox)

            # Prepare result
            result = {
                'license_plate': plate_text,
                'plate_confidence': float(confidence),
                'plate_ocr_success': bool(plate_text),
                'plate_snapshot_path': ''
            }

            # Always save the plate image if it was detected, even if OCR failed
            if plate_img is not None:
                try:
                    # Generate plate image filename - directly to plates directory without date folder
                    status = "detected" if not plate_text else "recognized"
                    plate_filename = f"plate_{violation_id}_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    plate_path = self.plates_dir / plate_filename

                    # Save the plate image
                    cv2.imwrite(str(plate_path), plate_img)

                    # Get relative path for the JSON record
                    rel_plate_path = str(plate_path.relative_to(self.base_dir))
                    result['plate_snapshot_path'] = rel_plate_path

                    if plate_text:
                        print(f"License plate recognized for violation {violation_id}: {plate_text} (conf: {confidence:.2f})")
                    else:
                        print(f"License plate detected but not readable for violation {violation_id} - saved image for manual review")

                    print(f"License plate image saved at: {rel_plate_path}")

                except Exception as e:
                    print(f"Error saving license plate image: {str(e)}")
                    # Continue with processing even if image save fails
            else:
                print(f"No license plate detected for violation {violation_id}")

            return result

        except Exception as e:
            print(f"Error in synchronous license plate processing: {str(e)}")
            return None
    
    def _process_queue(self):
        """Worker thread to process license plate recognition queue"""
        while self.running:
            try:
                # Get a request from the queue with timeout
                request = self.processing_queue.get(timeout=1.0)
                
                # Extract request data
                violation_record = request['violation_record']
                frame = request['frame']
                bbox = request['bbox']
                callback = request['callback']
                
                # Process the license plate
                plate_text, plate_bbox, plate_img, confidence = self.detector.detect_license_plate(frame, bbox)
                
                # Always save the plate image if it was detected, even if OCR failed
                if plate_img is not None:
                    try:
                        # Generate plate image filename - directly to plates directory without date folder
                        violation_id = violation_record.get('violation_id', 'unknown')
                        status = "detected" if not plate_text else "recognized"
                        plate_filename = f"plate_{violation_id}_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        plate_path = self.plates_dir / plate_filename
                        
                        # Save the plate image
                        cv2.imwrite(str(plate_path), plate_img)
                        
                        # Get relative path for the JSON record
                        rel_plate_path = str(plate_path.relative_to(self.base_dir))
                        
                        # Update violation record with plate info
                        violation_record['license_plate'] = plate_text
                        violation_record['plate_snapshot_path'] = rel_plate_path
                        violation_record['plate_confidence'] = float(confidence)
                        violation_record['plate_ocr_success'] = bool(plate_text)
                        
                        # Call the callback to update the JSON/CSV records
                        if callback:
                            callback(violation_record)
                            
                        if plate_text:
                            print(f"License plate recognized for violation {violation_id}: {plate_text} (conf: {confidence:.2f})")
                        else:
                            print(f"License plate detected but not readable for violation {violation_id} - saved image for manual review")
                        
                        print(f"License plate image saved at: {rel_plate_path}")
                    
                    except Exception as e:
                        print(f"Error processing license plate: {str(e)}")
                else:
                    print(f"No license plate detected for violation {violation_record.get('violation_id', 'unknown')}")
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except queue.Empty:
                # Just continue if no items in queue
                pass
            except Exception as e:
                print(f"Error in license plate processing thread: {str(e)}")
                time.sleep(1)  # Prevent high CPU usage in case of repeated errors
    
    def shutdown(self):
        """Shutdown the processing thread"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        print("License plate recognizer shutdown")
