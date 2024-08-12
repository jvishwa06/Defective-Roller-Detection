import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import os
import subprocess
import shutil
import yaml

def move_folder(source, destination):

    if not os.path.exists(source):
        print(f"Source folder {source} does not exist.")
        return
    if not os.path.exists(destination):
        os.makedirs(destination)

    for filename in os.listdir(source):
        source_file = os.path.join(source, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination)  # Use shutil.copy2 to preserve metadata
            print(f"Copied file {source_file} to {destination}")  

def convert_to_yolo(json_folder):

    if not os.path.exists(json_folder):
        print(f"Folder {json_folder} does not exist.")
        return
    test_size = '0.1'
    val_size = '0.2'
    subprocess.run(["python","-m","labelme2yolov8","--json_dir", json_folder,'--test_size',test_size,'--val_size',val_size])

def open_labelme(image_path):

    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        return
    
    subprocess.run(["labelme", image_path,"--autosave","--output","json_files"])

def combine_yolo_datasets(big_dataset, small_dataset, output_dataset):
    
    if os.path.exists(output_dataset):
        shutil.rmtree(output_dataset)
        print(f"Deleted folder {output_dataset}")
    else:
        print(f"Folder {output_dataset} does not exist.")
        
    def copy_files(src_dir, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_name in os.listdir(src_dir):
            full_file_name = os.path.join(src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dst_dir)

    def ensure_subdirs_exist(base_dir, subdirs):
        for subdir in subdirs:
            dir_path = os.path.join(base_dir, subdir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def update_yaml(big_yaml_path, small_yaml_path, output_yaml_path):
        # Load big dataset YAML
        with open(big_yaml_path, 'r') as file:
            big_data = yaml.safe_load(file)

        # Load small dataset YAML
        with open(small_yaml_path, 'r') as file:
            small_data = yaml.safe_load(file)

        # Combine classes
        combined_names = list(big_data['names'] + small_data['names'])
        num_classes = len(combined_names)

        # Update the YAML content
        combined_data = {
            'train': os.path.join(output_dataset, 'train/images'),
            'val': os.path.join(output_dataset, 'val/images'),
            'test': os.path.join(output_dataset, 'test/images'),
            'nc': num_classes,
            'names': combined_names
        }

        # Save the updated YAML content to the output YAML file
        with open(output_yaml_path, 'w') as file:
            yaml.safe_dump(combined_data, file)

    # Define the subdirectories to combine
    subdirs = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']

    # Ensure subdirectories exist in the output dataset
    ensure_subdirs_exist(output_dataset, subdirs)

    for subdir in subdirs:
        src_dir_small = os.path.join(small_dataset, subdir)
        dst_dir_big = os.path.join(big_dataset, subdir)
        dst_dir_combined = os.path.join(output_dataset, subdir)

        if os.path.exists(src_dir_small):
            copy_files(src_dir_small, dst_dir_combined)
        if os.path.exists(dst_dir_big):
            copy_files(dst_dir_big, dst_dir_combined)

    # Paths to YAML files
    big_yaml = os.path.join(big_dataset, 'dataset.yaml')
    small_yaml = os.path.join(small_dataset, 'dataset.yaml')
    output_yaml = os.path.join(output_dataset, 'dataset.yaml')

    # Update YAML file
    update_yaml(big_yaml, small_yaml, output_yaml)

    print(f"Datasets from {small_dataset} have been combined into {output_dataset}. Updated YAML file created at {output_yaml}")

def train_yolo_v8(data_yaml=os.path.join(os.getcwd(), 'Yolov8_Datasets', 'final_dataset', 'dataset.yaml'),
                  model='yolov8n.pt', epochs=100, imgsz=640):

    if not os.path.exists(data_yaml):
        print(f"dataset.yaml file at {data_yaml} does not exist.")
        return

    # Load the YAML file
    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)

    # Update paths to absolute paths
    base_dir = os.getcwd()
    data['train'] = os.path.join(base_dir, data['train']).replace('\\', '/')
    data['val'] = os.path.join(base_dir, data['val']).replace('\\', '/')
    data['test'] = os.path.join(base_dir, data['test']).replace('\\', '/')

    # Save the updated YAML content to a new temporary file
    temp_yaml = os.path.join(data_yaml.replace('\dataset.yaml',''), 'temp_dataset.yaml')
    with open(temp_yaml, 'w') as file:
        yaml.safe_dump(data, file)

    # Print the paths to check correctness
    print(f"Using dataset.yaml: {temp_yaml}")

    # Construct the training command
    train_command = [
        "yolo", "detect", "train",
        f"data={temp_yaml}",
        f"model={model}",
        f"epochs={epochs}",
        f"imgsz={imgsz}"
    ]
    print("Running command:", ' '.join(train_command))

    # Run the training command
    process = subprocess.Popen(train_command)
    return process

def stop_yolo_training(process):
    if process:
        process.terminate()
        process.wait()  # Wait for the process to terminate
        
if 'process' not in st.session_state:
    st.session_state.process = None
    
st.title("Custom training")



st.write("Upload image folder for custom training")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

if uploaded_files:
    # Create a directory to store uploaded files
    upload_dir = "uploaded_files"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)


    # Save uploaded files to the directory
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # List the files in the directory
    st.success("Files uploaded")
            
# st.write("Click the button to annoate")
annoate=st.button("Click here to annoate")

if annoate:
    image_path = r"uploaded_files"
    open_labelme(image_path)
dataset=st.button("click here to create dataset")

if dataset:
 
    # Example usage
    json_folder = r"Yolov8_Datasets\added_new_json"
    move_folder('json_files','Yolov8_Datasets/added_new_json')
    # move_folder('json_files','Yolov8_Datasets/labelme_dataset')
    
    convert_to_yolo(json_folder)
    
    big_dataset = 'Yolov8_Datasets\Welvision_Polarized_Lens-2'
    small_dataset = 'Yolov8_Datasets/added_new_json/YOLOv8Dataset'
    output_dataset = 'Yolov8_Datasets/final_dataset'
    combine_yolo_datasets(big_dataset, small_dataset, output_dataset)
    
    st.success("Sucessfully created the dataset")
    
    
train = st.button("Click here to train")
stop_train_button = st.button("Click here to stop training process")

if train:
    st.session_state.process = train_yolo_v8()
    st.write("model training successfully completed")
    
if stop_train_button and st.session_state.process:
    stop_yolo_training(st.session_state.process)
    st.session_state.process = None
    st.write("Training stopped")
        

        