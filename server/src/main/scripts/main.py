import argparse
import cv2
import os
from http import server
import io
import logging
import numpy as np
import socketserver
from threading import Condition, Thread
from io import BytesIO
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from ultralytics import YOLO
import tempfile
import base64
import json
from collections import Counter

# Parse command line arguments
parser = argparse.ArgumentParser(description="Server for camera streaming, video, and image processing")
parser.add_argument("--resolution", type=str, help="Video resolution in WIDTHxHEIGHT format (default: 640x640)", default="640x640")
parser.add_argument("--port", type=int, help="Port number for the server (default: 8000)", default=8000)
args = parser.parse_args()

# Parse resolution
resolution = tuple(map(int, args.resolution.split('x')))
model = YOLO('yolov8n.pt')  # Use YOLOv8n for lightweight inference
model.export(format='ncnn', int8=True, dynamic=True)
model = YOLO('yolov8n_ncnn_model')

# Create a directory for saving annotated files
SAVE_DIR = '/tmp/annotated_output'
object_map = {
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

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Utility functions
def save_annotated_image(annotated_img, file_name):
    file_path = os.path.join(SAVE_DIR, file_name)
    cv2.imwrite(file_path, annotated_img)
    return file_path


def save_annotated_video(annotated_frames, original_filename):
    output_filename = os.path.splitext(original_filename)[0] + '_annotated.mp4'  # Rename with "_annotated.mp4"
    output_path = os.path.join(SAVE_DIR, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = annotated_frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    for frame in annotated_frames:
        out.write(frame)
    out.release()
    return output_path


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()
        self.inference_thread = None
        self.frame_buffer = None

    def run_inference(self, img):
        results = model(img)
        detections = results[0].boxes.cls.tolist()

        # Count class 0 detections
        class_0_count = detections.count(0)

        # Display the count on the image
        result = results[0].plot()
        cv2.putText(result, f"Class 0 Count: {class_0_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buf = cv2.imencode('.jpg', result)
        buf = buf.tobytes()

        with self.condition:
            self.frame_buffer = buf
            self.condition.notify_all()

    def write(self, buf):
        img_array = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = Thread(target=self.run_inference, args=(img,))
            self.inference_thread.start()

        with self.condition:
            if self.frame_buffer is not None:
                self.frame = self.frame_buffer
            else:
                self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/upload/video':
            length = int(self.headers['Content-Length'])
            body = self.rfile.read(length)

            # Save the uploaded video temporarily
            video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_file.write(body)
            video_file.close()
            video_path = video_file.name

            # Process the video
            cap = cv2.VideoCapture(video_path)
            annotated_frames = []
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Create a memory buffer for the output video
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = model(frame)[0].plot()  # Annotate the frame
                writer.write(result)

            cap.release()
            writer.release()

            # Return video from memory
            with open(temp_output.name, "rb") as video_file:
                video_data = video_file.read()

            os.remove(video_path)  # Clean up the temporary uploaded video
            os.remove(temp_output.name)  # Clean up the temporary annotated video

            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.end_headers()
            self.wfile.write(video_data)

        elif self.path == '/upload/image':
            length = int(self.headers['Content-Length'])
            body = self.rfile.read(length)

            # Save image temporarily
            image_file = tempfile.NamedTemporaryFile(delete=False)
            image_file.write(body)
            image_file.close()
            image_path = image_file.name

            # Process the image using YOLO
            img = cv2.imread(image_path)
            results = model(img)  # Apply YOLO inference
            result_img = results[0].plot()  # Annotated image

            # Extract detected objects
            detections = results[0].boxes.cls.tolist()  # List of detected class IDs
            class_counts = dict(Counter(detections))  # Count each class ID
            readable_counts = {object_map.get(int(cls), f"Class {cls}"): count 
                               for cls, count in class_counts.items()}

            # Save the annotated image
            output_image_path = save_annotated_image(result_img, os.path.basename(image_path) + '_annotated.jpg')

            # Encode the image as base64
            with open(output_image_path, "rb") as img_file:
                base64_string = base64.b64encode(img_file.read()).decode('utf-8')

            # Create the JSON response
            response = {
                "message": "Image processed",
                "image_data": base64_string,
                "class_counts": readable_counts
            }

            # Send the response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# Configure and start the camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": resolution}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', args.port)
    server = StreamingServer(address, StreamingHandler)
    print(f"Server running on port {args.port}")
    server.serve_forever()
finally:
    picam2.stop_recording()
