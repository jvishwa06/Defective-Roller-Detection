from ultralytics import YOLO
import os
import cv2
import csv
from datetime import datetime

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")

# Define the video source path
source = r"test_scripts\city.mp4"

# Run inference on the video, filtering for classes 2 and 7 (e.g., cars and trucks)
results = model.predict(source, stream=True, classes=[2, 7])

# Prepare the CSV file
csv_file = 'detected_objects_summary.csv'
csv_headers = ['Date', 'Time', 'Defect Type', 'Class ID', 'Count']

# Check if the CSV file already exists
if not os.path.exists(csv_file):
    # If not, create it and write the headers
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)

# Initialize a dictionary to keep track of counts
object_counts = {}

# Loop through the results
for frame_results in results:
    frame = frame_results.orig_img  # Original frame image

    for box in frame_results.boxes:  # Loop through all detected boxes
        cls_id = int(box.cls)  # Class ID
        class_name = model.names[cls_id]  # Class name

        # Create a directory for the class if it doesn't exist
        os.makedirs(f"detected_objects/{class_name}", exist_ok=True)

        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Crop the detected object from the frame
        cropped_img = frame[y1:y2, x1:x2]

        # Generate a unique filename based on the class and box index
        filename = f"detected_objects/{class_name}/{cls_id}_{x1}_{y1}_{x2}_{y2}.jpg"

        # Save the cropped image
        cv2.imwrite(filename, cropped_img)

        # Update the count for this class
        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1

        print(f"Saved {filename}")

# Write the counts to the CSV file
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    for class_name, count in object_counts.items():
        # Get the current date and time
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        
        # Write the entry to the CSV file
        writer.writerow([date_str, time_str, class_name, cls_id, count])

print("Detection and CSV update completed!")
