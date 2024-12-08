from http import server
import socketserver
import logging
import tempfile
import json
import base64
import cv2
import os
from .utils import save_annotated_image, save_annotated_video

class StreamingHandler(server.BaseHTTPRequestHandler):
    def __init__(self, *args, output=None, inference_model=None, **kwargs):
        self.output = output
        self.inference_model = inference_model
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/stream':
            self._handle_stream()
        else:
            self.send_error(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/upload/video':
            self._handle_video_upload()
        elif self.path == '/upload/image':
            self._handle_image_upload()
        else:
            self.send_error(404)
            self.end_headers()

    def _handle_stream(self):
        """Handle streaming request."""
        self.send_response(200)
        self.send_header('Age', 0)
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        try:
            while True:
                with self.output.condition:
                    self.output.condition.wait()
                    frame = self.output.frame
                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
        except Exception as e:
            logging.warning(
                'Removed streaming client %s: %s', 
                self.client_address, 
                str(e)
            )

    def _handle_video_upload(self):
        """Handle video upload and processing."""
        length = int(self.headers['Content-Length'])
        body = self.rfile.read(length)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
            video_file.write(body)
            video_path = video_file.name

        # Process video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        annotated_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = self.inference_model.process_frame_dynamic(frame)
            annotated_frames.append(result)

        cap.release()
        output_path = save_annotated_video(annotated_frames, os.path.basename(video_path))
        # Encode vid
        video_data = None
        with open(output_path, "rb") as video_file:
            video_data = base64.b64encode(video_file.read()).decode('utf-8')

        # Cleanup
        os.remove(video_path)
        os.remove(output_path)

        # Prepare response
        response = {
            "message": "Video processed",
            "video_data": video_data
        }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())



    def _handle_image_upload(self):
        """Handle image upload and processing."""
        length = int(self.headers['Content-Length'])
        body = self.rfile.read(length)

        with tempfile.NamedTemporaryFile(delete=False) as image_file:
            image_file.write(body)
            image_path = image_file.name

        # Process image
        img = cv2.imread(image_path)
        annotated_img, readable_counts = self.inference_model.process_image(img)

        # Save and encode result
        output_path = save_annotated_image(
            annotated_img, 
            os.path.basename(image_path) + '_annotated.jpg'
        )

        with open(output_path, "rb") as img_file:
            base64_string = base64.b64encode(img_file.read()).decode('utf-8')

        # Prepare response
        response = {
            "message": "Image processed",
            "image_data": base64_string,
            "class_counts": readable_counts
        }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

        # Cleanup
        os.remove(image_path)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True