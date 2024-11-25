import os
import cv2

SAVE_DIR = '/tmp/annotated_output'

def ensure_save_dir():
    """Ensure the save directory exists."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def save_annotated_image(annotated_img, file_name):
    """Save an annotated image and return the file path."""
    ensure_save_dir()
    file_path = os.path.join(SAVE_DIR, file_name)
    cv2.imwrite(file_path, annotated_img)
    return file_path

def save_annotated_video(annotated_frames, original_filename):
    """Save annotated video frames and return the file path."""
    ensure_save_dir()
    output_filename = os.path.splitext(original_filename)[0] + '_annotated.mp4'
    output_path = os.path.join(SAVE_DIR, output_filename)
    
    if annotated_frames:
        height, width, _ = annotated_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        for frame in annotated_frames:
            out.write(frame)
        out.release()
    
    return output_path