from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np

class InferenceModel:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the YOLO model."""
        self.model = YOLO(model_path)
        self.prepare_model()
        self.object_map = {
            0: 'person', 1: 'bicycle', 2: 'car', 
            # ... (rest of the object map)
        }
        
    def prepare_model(self):
        """Prepare the model for inference by exporting to NCNN format."""
        self.model.export(format='ncnn', int8=True, dynamic=True)
        self.model = YOLO('yolov8n_ncnn_model')
        
    def process_image(self, image):
        """Process a single image and return results with annotations."""
        results = self.model(image)
        detections = results[0].boxes.cls.tolist()
        
        # Count detections
        class_counts = dict(Counter(detections))
        readable_counts = {
            self.object_map.get(int(cls), f"Class {cls}"): count 
            for cls, count in class_counts.items()
        }
        
        # Create annotated image
        annotated_img = results[0].plot()
        
        return annotated_img, readable_counts
        
    def process_frame(self, frame):
        """Process a video frame and return annotated frame."""
        results = self.model(frame)
        detections = results[0].boxes.cls.tolist()
        
        # Count class 0 (person) detections
        class_0_count = detections.count(0)
        
        # Annotate frame
        result = results[0].plot()
        cv2.putText(
            result, 
            f"Class 0 Count: {class_0_count}", 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return result