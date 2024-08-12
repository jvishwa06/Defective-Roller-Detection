import streamlit as st
import cv2
from ultralytics import YOLO
import torch

# Set up the Streamlit app
st.title("Defect Detection in Rollers")
st.sidebar.title("Settings")

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Sidebar options for defect size consideration
size_based_detection = st.sidebar.radio(
    "Detection Mode", 
    ("Detect Based on Size", "Detect Without Size Consideration")
)

# Sidebar slider to adjust the defect size threshold (only visible if size-based detection is selected)
if size_based_detection == "Detect Based on Size":
    min_defect_size = st.sidebar.slider("Minimum Defect Size (cm)", 1, 10, 1)
else:
    min_defect_size = 0  # No size threshold if detection without size consideration

# Define the pixel to cm ratio (set this according to your camera setup)
pixel_to_cm_ratio = 0.1  # Example: 1 pixel = 0.1 cm

# Streamlit webcam input
st.sidebar.write("Adjust the defect size threshold using the slider.")
source = st.sidebar.selectbox("Select Source", ["Webcam", "Video", "Image"])
if source == "Webcam":
    source = 0
elif source == "Video":
    source = st.sidebar.text_input("Enter Video Path", "path/to/video.mp4")
elif source == "Image":
    source = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Initialize the video capture or image depending on the source
cap = None
if isinstance(source, int) or source.endswith(".mp4"):
    cap = cv2.VideoCapture(source)

def process_frame(frame, min_defect_size):
    results = model.predict(frame, conf=0.5, show=False)
    boxes = results[0].boxes.xywh.cpu()

    for box in boxes:
        x, y, w, h = box
        width_cm = w * pixel_to_cm_ratio
        height_cm = h * pixel_to_cm_ratio

        if width_cm >= min_defect_size and height_cm >= min_defect_size:
            cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
            label = f"{width_cm:.2f}x{height_cm:.2f} cm"
            cv2.putText(frame, label, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Button to start detection
start_detection = st.button("Start Detection")

# Stream the video or image with defect detection if start button is pressed
if start_detection:
    if cap:
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, min_defect_size)
            stframe.image(frame, channels="BGR", use_column_width=True)
    else:
        if source:
            image = cv2.imdecode(torch.from_numpy(source.getvalue()), 1)
            frame = process_frame(image, min_defect_size)
            st.image(frame, channels="BGR", use_column_width=True)
        else:
            st.warning("Please select a valid image or video file.")

# Release resources
if cap:
    cap.release()
