import cv2
import numpy as np
import os
import torch
from pathlib import Path
import easyocr
import threading
import time
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO  # Use ultralytics for YOLOv8 directly

class LicensePlateDetector:
    """
    License plate detection and recognition using a pre-trained model and EasyOCR.
    """
    
    def __init__(self, model_path=None, ocr_langs=None):
        self.model_loaded = False
        self.model = None
        self.ocr_reader = None
        self.ocr_ready = False
        self.lock = threading.Lock()
        
        # EasyOCR language settings
        self.ocr_langs = ocr_langs or ['en']
        
        # Default model path if none provided
        if model_path is None:
            model_path = str(Path(__file__).parent / "Licenses.pt")
        
        # Load object detection model for license plates
        self._load_model(model_path)
        
        # Initialize EasyOCR in a background thread
        threading.Thread(target=self._initialize_ocr, daemon=True).start()
    
    def _load_model(self, model_path):
        """Load the license plate detection model"""
        try:
            if os.path.exists(model_path):
                # Use YOLO from ultralytics directly for YOLOv8 model
                self.model = YOLO(model_path)
                if self.model:
                    self.model_loaded = True
                    print(f"License plate detector model loaded from {model_path}")
            else:
                print(f"License plate model not found at {model_path}")
        except Exception as e:
            print(f"Failed to load license plate detection model: {str(e)}")
            self.model = None
    
    def _initialize_ocr(self):
        """Initialize EasyOCR in a background thread"""
        try:
            print(f"Initializing EasyOCR with languages: {self.ocr_langs}")
            self.ocr_reader = easyocr.Reader(self.ocr_langs, gpu=torch.cuda.is_available())
            with self.lock:
                self.ocr_ready = True
            print("EasyOCR initialization complete")
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {str(e)}")
    
    def detect_license_plate(self, frame, vehicle_bbox=None):
        """
        Detect license plate in a frame or vehicle region
        
        Args:
            frame: The input image
            vehicle_bbox: Optional bounding box of vehicle to focus detection
        
        Returns:
            plate_text: Recognized license plate text or empty string
            plate_bbox: Coordinates of license plate in frame or None
            plate_img: Cropped image of the plate or None
            confidence: Detection confidence or 0.0
        """
        # Check if models are loaded
        if not self.model_loaded:
            return "", None, None, 0.0
            
        # If OCR isn't ready yet, we'll still detect plates but return empty text
        with self.lock:
            ocr_ready = self.ocr_ready
        
        try:
            # Create a working copy of the frame
            img = frame.copy()
            
            # If vehicle bbox is provided, crop the image to that region with margin
            if vehicle_bbox is not None:
                x1, y1, x2, y2 = vehicle_bbox
                # Add some margin around the vehicle
                height, width = frame.shape[:2]
                margin_x = int((x2 - x1) * 0.1)  # 10% margin
                margin_y = int((y2 - y1) * 0.1)  # 10% margin
                
                # Ensure we stay within image bounds
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(width, x2 + margin_x)
                y2 = min(height, y2 + margin_y)
                
                # Crop the image
                if x1 < x2 and y1 < y2:
                    img = frame[y1:y2, x1:x2]
                
            # Run detection model on the image - Use YOLOv8 syntax directly
            results = self.model(img)
            
            # No plates found
            if len(results) == 0 or len(results[0].boxes) == 0:
                return "", None, None, 0.0
            
            # Get the most confident plate detection
            best_detection = None
            best_confidence = 0
            
            # Process all detected plates using YOLOv8 format
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                xyxy = box.xyxy.cpu().numpy()[0]  # Get bounding box in xyxy format
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, 'conf') else 0.0
                
                # Filter by minimum confidence
                if conf > best_confidence:
                    best_confidence = conf
                    best_detection = (int(x1), int(y1), int(x2), int(y2), conf)
            
            if best_detection is None:
                return "", None, None, 0.0
            
            # Extract coordinates of the best detection
            px1, py1, px2, py2, conf = best_detection
            
            # Adjust coordinates if we cropped the image using vehicle_bbox
            if vehicle_bbox is not None:
                vx1, vy1, _, _ = vehicle_bbox
                px1 += vx1 - margin_x
                px2 += vx1 - margin_x
                py1 += vy1 - margin_y
                py2 += vy1 - margin_y
            
            # Ensure coordinates are within the original image
            height, width = frame.shape[:2]
            px1 = max(0, min(width-1, px1))
            py1 = max(0, min(height-1, py1))
            px2 = max(0, min(width-1, px2))
            py2 = max(0, min(height-1, py2))
            
            # Extract the plate image for OCR
            if px1 < px2 and py1 < py2:
                plate_img = frame[py1:py2, px1:px2].copy()
                
                # Apply image pre-processing to improve OCR
                plate_img = self._preprocess_plate_image(plate_img)
                
                # Perform OCR on the plate image only if OCR is ready
                plate_text = ""
                if ocr_ready:
                    try:
                        plate_text = self._recognize_plate_text(plate_img)
                    except Exception as e:
                        print(f"OCR Error: {str(e)}")
                        # Continue with empty text but still return the plate image
            
            # Return the plate image even if OCR fails or isn't ready
            # This allows for human review or retrying OCR later
            return plate_text, (px1, py1, px2, py2), plate_img, conf
    
        except Exception as e:
            print(f"Error in license plate detection: {str(e)}")
        
        return "", None, None, 0.0
    
    def _preprocess_plate_image(self, plate_img):
        """
        Apply preprocessing to improve license plate image quality for OCR
        """
        try:
            # Convert to PIL image for easier processing
            pil_img = Image.fromarray(plate_img)
            
            # Resize to a good size for OCR if too small
            if pil_img.width < 100 or pil_img.height < 30:
                factor = max(100 / pil_img.width, 30 / pil_img.height)
                new_width = int(pil_img.width * factor)
                new_height = int(pil_img.height * factor)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(2.0)
            
            # Convert back to OpenCV format
            processed_img = np.array(pil_img)
            
            return processed_img
            
        except Exception as e:
            print(f"Error in preprocessing license plate: {str(e)}")
            # Return original image if processing fails
            return plate_img
    
    def _recognize_plate_text(self, plate_img):
        """
        Recognize text on the license plate image using EasyOCR
        """
        try:
            # Run OCR
            results = self.ocr_reader.readtext(plate_img)
            
            # Combine all detected text
            texts = []
            for detection in results:
                text = detection[1]
                # Simple filtering to keep only letters and digits
                filtered_text = ''.join(c for c in text if c.isalnum())
                if filtered_text:
                    texts.append(filtered_text)
            
            # Join all detected texts
            plate_text = ' '.join(texts)
            
            # Remove extra spaces and common OCR errors
            plate_text = plate_text.strip().replace(' ', '')
            
            return plate_text
            
        except Exception as e:
            print(f"Error in OCR: {str(e)}")
            return ""
