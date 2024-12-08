from threading import Condition, Thread
import io
import numpy as np
import cv2

class StreamingOutput(io.BufferedIOBase):
    def __init__(self, inference_model):
        """Initialize streaming output with inference model."""
        self.frame = None
        self.condition = Condition()
        self.inference_thread = None
        self.frame_buffer = None
        self.inference_model = inference_model

    def run_inference(self, img):
        """Run inference on the image and update frame buffer."""
        result = self.inference_model.process_frame_fixed(img)
        _, buf = cv2.imencode('.jpg', result)
        buf = buf.tobytes()

        with self.condition:
            self.frame_buffer = buf
            self.condition.notify_all()

    def write(self, buf):
        """Write frame data and trigger inference."""
        img_array = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = Thread(
                target=self.run_inference, 
                args=(img,)
            )
            self.inference_thread.start()

        with self.condition:
            if self.frame_buffer is not None:
                self.frame = self.frame_buffer
            else:
                self.frame = buf
            self.condition.notify_all()