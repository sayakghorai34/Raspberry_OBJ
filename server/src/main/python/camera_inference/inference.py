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
           0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
           5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
           10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
           15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
           20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
           25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
           30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
           35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
           40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
           45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
           50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
           55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
           60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
           65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
           70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
           75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
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
            f"Human Count: {class_0_count}", 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        return result
