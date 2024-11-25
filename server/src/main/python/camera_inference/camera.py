from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

class CameraManager:
    def __init__(self, resolution=(640, 640)):
        """Initialize the camera with given resolution."""
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.configure_camera()
        
    def configure_camera(self):
        """Configure the camera with specified resolution."""
        self.picam2.configure(
            self.picam2.create_video_configuration(
                main={"size": self.resolution}
            )
        )
    
    def start_recording(self, output):
        """Start recording with the specified output."""
        self.picam2.start_recording(JpegEncoder(), FileOutput(output))
        
    def stop_recording(self):
        """Stop recording."""
        self.picam2.stop_recording()