import argparse
from src.main.python.camera_inference.camera import CameraManager
from src.main.python.camera_inference.inference import InferenceModel
from src.main.python.camera_inference.streaming import StreamingOutput
from src.main.python.camera_inference.server import StreamingServer, StreamingHandler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Server for camera streaming, video, and image processing"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Video resolution in WIDTHxHEIGHT format (default: 640x640)",
        default="640x640"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the server (default: 8000)",
        default=8000
    )
    return parser.parse_args()

def main():
    """Main function to run the server."""
    args = parse_args()
    resolution = tuple(map(int, args.resolution.split('x')))
    
    # Initialize components
    inference_model = InferenceModel()
    camera_manager = CameraManager(resolution)
    streaming_output = StreamingOutput(inference_model)
    
    # Start camera recording
    camera_manager.start_recording(streaming_output)
    
    try:
        # Start server
        address = ('', args.port)
        handler = lambda *args, **kwargs: StreamingHandler(
            *args,
            output=streaming_output,
            inference_model=inference_model,
            **kwargs
        )
        server = StreamingServer(address, handler)
        print(f"Server running on port {args.port}")
        server.serve_forever()
    finally:
        camera_manager.stop_recording()

if __name__ == '__main__':
    main()
