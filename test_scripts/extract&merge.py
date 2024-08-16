import cv2
import time
from PIL import Image

def extract_frames_from_video(video_path, bounding_boxes, interval_ms=30):
    """
    Extract frames from a video, crop regions using bounding boxes, and return cropped images.

    Parameters:
    - video_path: Path to the video file.
    - bounding_boxes: List of tuples with bounding box coordinates (x_min, y_min, x_max, y_max).
    - interval_ms: Time interval between frames in milliseconds.

    Returns:
    - cropped_images: List of cropped images from the video frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    cropped_images = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(frame_rate * interval_ms / 1000)

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process every nth frame based on interval_frames
        if frame_count % interval_frames == 0:
            # Convert the frame to PIL image format
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            for box in bounding_boxes:
                x_min, y_min, x_max, y_max = box
                # Crop the image using the bounding box
                cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
                cropped_images.append(cropped_image)
        
        frame_count += 1

    cap.release()
    return cropped_images

def merge_images(images):
    """
    Merge a list of images horizontally into a single image.

    Parameters:
    - images: List of PIL Image objects.

    Returns:
    - merged_image: The merged image.
    """
    # Calculate the total width and maximum height for the merged image
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Create a new blank image with the calculated width and height
    merged_image = Image.new('RGB', (total_width, max_height))
    
    # Paste images into the merged image
    x_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return merged_image


# Example usage
if __name__ == "__main__":
    video_path = 'video.mp4'  # Path to the video file
    bounding_boxes = [(30, 50, 180, 200), (200, 80, 350, 250), (360, 120, 480, 300)]

    # Extract frames from the video and crop the regions
    cropped_images = extract_frames_from_video(video_path, bounding_boxes, interval_ms=30)

    # Merge the cropped images
    merged_image = merge_images(cropped_images)

    # Save or show the merged image
    merged_image.save("merged_image.jpg")
    merged_image.show()
