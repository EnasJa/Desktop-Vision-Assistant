import os
import time
import logging
import cv2
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO

from config import (
    YOLO_MODEL, 
    CONFIDENCE_THRESHOLD, 
    MAX_DETECTIONS,
    MODELS_DIR,
    CAMERA_WIDTH,
    CAMERA_HEIGHT
)

logger = logging.getLogger(__name__)

class VisionProcessor:
    def __init__(self):
        logger.info(f"Initializing vision processor with YOLOv8 model: {YOLO_MODEL}")
        model_path = self._get_model_path(YOLO_MODEL)
        self.model = YOLO(model_path)
        self.camera = None
        self.camera_index = 0
        self.processing = False
        self.frame_queue = Queue(maxsize=1)  # Only keep the latest frame
        self.result_queue = Queue(maxsize=1)  # Only keep the latest result
        self.processing_thread = None
        
    def _get_model_path(self, model_name):
        """Get the path to the YOLOv8 model, download if needed"""
        # If it's a full path, return as is
        if os.path.exists(model_name):
            return model_name
            
        # Check if it's in the models directory
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            return model_path
            
        # If the model doesn't exist, YOLO will download it automatically
        return model_name
        
    def start_camera(self, camera_index=0):
        """Start the camera capture"""
        try:
            logger.info(f"Starting camera with index: {camera_index}")
            self.camera_index = camera_index
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera with index {camera_index}")
                return False
                
            logger.info("Camera started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
            
    def stop_camera(self):
        """Stop the camera capture"""
        if self.camera and self.camera.isOpened():
            logger.info("Stopping camera")
            self.camera.release()
            self.camera = None
            
    def start_processing(self):
        """Start the vision processing thread"""
        if self.processing:
            logger.warning("Vision processing already running")
            return
            
        if not self.camera or not self.camera.isOpened():
            logger.error("Camera not started, cannot begin processing")
            return False
            
        logger.info("Starting vision processing")
        self.processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        return True
        
    def stop_processing(self):
        """Stop the vision processing thread"""
        logger.info("Stopping vision processing")
        self.processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            
    def get_frame(self):
        """Get the latest frame if available"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
        
    def get_detection_results(self):
        """Get the latest detection results if available"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
        
    def _processing_loop(self):
        """Main processing loop for vision detection"""
        logger.info("Vision processing loop started")
        while self.processing and self.camera and self.camera.isOpened():
            # Read frame from camera
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("Failed to get frame from camera")
                time.sleep(0.1)
                continue
                
            # Update the latest frame
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # Clear the queue and put the new frame
                _ = self.frame_queue.get()
                self.frame_queue.put(frame)
                
            # Process frame with YOLO
            try:
                results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
                
                # Process results
                detections = []
                for i, detection in enumerate(results.boxes.data.tolist()):
                    if i >= MAX_DETECTIONS:  # Limit the number of detections
                        break
                        
                    x1, y1, x2, y2, confidence, class_id = detection
                    class_id = int(class_id)
                    class_name = results.names[class_id]
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
                
                # Update the latest results
                if not self.result_queue.full():
                    self.result_queue.put(detections)
                else:
                    # Clear the queue and put the new results
                    _ = self.result_queue.get()
                    self.result_queue.put(detections)
                    
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                
            # Sleep a small amount to prevent CPU overload
            time.sleep(0.01)
            
    def process_single_image(self, image):
        """Process a single image and return detections"""
        try:
            logger.info("Processing single image")
            results = self.model(image, conf=CONFIDENCE_THRESHOLD)[0]
            
            # Process results
            detections = []
            for i, detection in enumerate(results.boxes.data.tolist()):
                if i >= MAX_DETECTIONS:  # Limit the number of detections
                    break
                    
                x1, y1, x2, y2, confidence, class_id = detection
                class_id = int(class_id)
                class_name = results.names[class_id]
                
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                
            return detections
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return []
            
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on the frame"""
        if frame is None or detections is None:
            return frame
            
        # Make a copy to avoid modifying the original
        output_frame = frame.copy()
        
        for detection in detections:
            # Extract detection info
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(
                output_frame, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                (0, 255, 0), 
                2
            )
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                output_frame,
                (bbox[0], bbox[1] - label_size[1] - 10),
                (bbox[0] + label_size[0], bbox[1]),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                output_frame,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
        return output_frame


# Simple test function
def test_vision_processing():
    vision = VisionProcessor()
    if vision.start_camera():
        print("Camera started. Starting vision processing...")
        vision.start_processing()
        
        try:
            # Run for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                frame = vision.get_frame()
                detections = vision.get_detection_results()
                
                if frame is not None and detections is not None:
                    frame_with_detections = vision.draw_detections(frame, detections)
                    cv2.imshow("YOLOv8 Detections", frame_with_detections)
                    
                    # Print detections
                    print(f"Detected {len(detections)} objects:")
                    for det in detections:
                        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
                        
                    # Exit if 'q' is pressed
                    if cv2.waitKey(1) == ord('q'):
                        break
                        
                time.sleep(0.1)
                
        finally:
            vision.stop_processing()
            vision.stop_camera()
            cv2.destroyAllWindows()
            print("Vision test completed")
    else:
        print("Failed to start camera")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Test the vision processing
    test_vision_processing()