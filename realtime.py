import cv2
import streamlit as st
from ultralytics import YOLO
import torch
import time
import pandas as pd
import os
import psutil

# Load the YOLOv8 model (replace 'best.pt' with your model file)
model = YOLO("best.pt")

# Function to save frame
def save_frame(frame, image_folder, image_id):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_path = os.path.join(image_folder, f"frame_{image_id}.jpg")
    cv2.imwrite(image_path, frame)

# Function to process frame
def process_frame(frame, min_defect_size, pixel_to_cm_ratio):
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

# Function to save inference results to a CSV file
def save_results_to_csv(results_folder, frame_id, object_counts, defective_count):
    results_file = os.path.join(results_folder, "detection_results.csv")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    df = pd.DataFrame({
        'frame_id': [frame_id],
        'total_defective_bearings': [defective_count],
        **{cls: [count] for cls, count in object_counts.items()}
    })

    if not os.path.isfile(results_file):
        df.to_csv(results_file, index=False)
    else:
        df.to_csv(results_file, mode='a', header=False, index=False)

def main():
    st.title("Real-time Defective Bearing Detection")

    # Sidebar Configuration
    st.sidebar.title("Configuration")

    input_source = st.sidebar.radio(
        "Select input source",
        ('Webcam', 'Local video')
    )

    # Define the pixel to cm ratio (set this according to your camera setup)
    pixel_to_cm_ratio = 0.01
    
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

    # Sidebar for image-saving options
    image_save_limit = st.sidebar.number_input("Image Save Limit", min_value=1, value=500)
    save_images = st.sidebar.checkbox("Save Images")
    image_folder = "saved_images"

    # Folder to store results
    results_folder = "detection_results"

    # Input source selection
    if input_source.startswith("Local video"):
        video_file_1 = st.sidebar.file_uploader("Select input video 1", type=["mp4", "avi"])
        video_file_2 = st.sidebar.file_uploader("Select input video 2", type=["mp4", "avi"])
        if video_file_1 is not None and video_file_2 is not None:
            save_path_1 = os.path.join(os.getcwd(), video_file_1.name)
            save_path_2 = os.path.join(os.getcwd(), video_file_2.name)
            with open(save_path_1, "wb") as f:
                f.write(video_file_1.getbuffer())
            with open(save_path_2, "wb") as f:
                f.write(video_file_2.getbuffer())
            video_path_1 = save_path_1
            video_path_2 = save_path_2
        else:
            st.warning("Please upload both video files.")
            return
    elif input_source.startswith("Webcam"):
        video_path_1 = 0  # Default webcam 1
        video_path_2 = 1  # Default webcam 2

    # Button to start detection
    start_detection = st.sidebar.button("Start Detection")

    if start_detection:
        # Display system stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cpu_text = st.empty()
        with col2:
            ram_text = st.empty()
        with col3:
            gpu_text = st.empty()
        with col4:
            fps_text = st.empty()

        # Video display
        col5, col6 = st.columns(2)
        with col5:
            frame_display_1 = st.empty()
        with col6:
            frame_display_2 = st.empty()

        # Inference stats
        st.subheader("Inference Stats")
        col7, col8 = st.columns(2)
        with col7:
            st.write("Total Defective Bearings (Camera 1)")
            total_defective_bearings_chart_1 = st.empty()
        with col8:
            st.write("Object Counts (Camera 1)")
            object_counts_chart_1 = st.empty()

        with col7:
            st.write("Total Defective Bearings (Camera 2)")
            total_defective_bearings_chart_2 = st.empty()
        with col8:
            st.write("Object Counts (Camera 2)")
            object_counts_chart_2 = st.empty()

        # System stats
        st.subheader("System Stats")
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            cpu_usage_text = st.empty()
        with col10:
            ram_usage_text = st.empty()
        with col11:
            gpu_usage_text = st.empty()
        with col12:
            fps_count_text = st.empty()

        # Video capture
        cap_1 = cv2.VideoCapture(video_path_1, cv2.CAP_DSHOW)
        cap_2 = cv2.VideoCapture(video_path_2, cv2.CAP_DSHOW)
        
        total_defective_bearings_1 = 0
        total_defective_bearings_2 = 0
        object_counts_1 = {cls: 0 for cls in ['chatter_on_od', 'damage_on_big_radius', 'damage_on_od', 'damage_on_small_radius', 'dent', 'roller', 'rust_on_od', 'spherical']}
        object_counts_2 = {cls: 0 for cls in ['chatter_on_od', 'damage_on_big_radius', 'damage_on_od', 'damage_on_small_radius', 'dent', 'roller', 'rust_on_od', 'spherical']}
        
        start_time_1 = time.time()
        start_time_2 = time.time()
        frame_count_1 = 0
        frame_count_2 = 0
        image_save_counter = 0

        while cap_1.isOpened() and cap_2.isOpened():
            ret_1, frame_1 = cap_1.read()
            ret_2, frame_2 = cap_2.read()

            if ret_1 and ret_2:
                frame_count_1 += 1
                frame_count_2 += 1
                
                frame_1 = process_frame(frame_1, min_defect_size, pixel_to_cm_ratio)
                frame_2 = process_frame(frame_2, min_defect_size, pixel_to_cm_ratio)

                # Create a copy of the image for displaying the detections
                output_image_1 = frame_1
                output_image_2 = frame_2

                # Save frames if the option is enabled
                if save_images:
                    save_frame(output_image_1, image_folder, image_save_counter)
                    save_frame(output_image_2, image_folder, image_save_counter + 1)
                    image_save_counter += 2

                    if image_save_counter >= image_save_limit:
                        st.sidebar.warning("Image save limit reached.")
                        save_images = False  # Stop saving images after the limit is reached

                # Update total defective bearings detected and object counts
                for result in model(frame_1):
                    for detection in result.boxes:
                        confidence = detection.conf[0]
                        class_id = int(detection.cls[0])
                        class_name = model.names[class_id]

                        if confidence >= 0.5:  # Use a fixed confidence threshold for detection
                            object_counts_1[class_name] += 1
                            if class_name != 'roller' and class_name != 'spherical':  # Assuming 'roller' and 'spherical' are not defective classes
                                total_defective_bearings_1 += 1

                for result in model(frame_2):
                    for detection in result.boxes:
                        confidence = detection.conf[0]
                        class_id = int(detection.cls[0])
                        class_name = model.names[class_id]

                        if confidence >= 0.5:  # Use a fixed confidence threshold for detection
                            object_counts_2[class_name] += 1
                            if class_name != 'roller' and class_name != 'spherical':  # Assuming 'roller' and 'spherical' are not defective classes
                                total_defective_bearings_2 += 1

                # Update system stats
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                gpu_usage = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
                gpu_usage = (gpu_usage - torch.cuda.memory_reserved(0)) / gpu_usage * 100 if torch.cuda.is_available() else 0
                fps_count = (frame_count_1 / (time.time() - start_time_1)) if frame_count_1 else 0

                cpu_usage_text.text(f"CPU Usage: {cpu_usage}%")
                ram_usage_text.text(f"RAM Usage: {ram_usage}%")
                gpu_usage_text.text(f"GPU Usage: {gpu_usage:.2f}%")
                fps_count_text.text(f"FPS: {fps_count:.2f}")

                # Display frames
                frame_display_1.image(output_image_1, channels="BGR", use_column_width=True)
                frame_display_2.image(output_image_2, channels="BGR", use_column_width=True)

                # Update inference stats
                total_defective_bearings_chart_1.write(f"Total Defective Bearings: {total_defective_bearings_1}")
                object_counts_chart_1.write(pd.DataFrame(object_counts_1.items(), columns=['Class', 'Count']))

                total_defective_bearings_chart_2.write(f"Total Defective Bearings: {total_defective_bearings_2}")
                object_counts_chart_2.write(pd.DataFrame(object_counts_2.items(), columns=['Class', 'Count']))

                # Save inference results to CSV
                save_results_to_csv(results_folder, frame_count_1, object_counts_1, total_defective_bearings_1)
                save_results_to_csv(results_folder, frame_count_2, object_counts_2, total_defective_bearings_2)

            else:
                break

        cap_1.release()
        cap_2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
