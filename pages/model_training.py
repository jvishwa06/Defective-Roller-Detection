import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import os
import subprocess
import shutil
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from torch.utils.tensorboard import SummaryWriter

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
    subprocess.run(["python", "-m", "labelme2yolov8", "--json_dir", json_folder, '--test_size', test_size, '--val_size', val_size])

def open_labelme(image_path):
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        return
    subprocess.run(["labelme", image_path, "--autosave", "--output", "json_files"])

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
        with open(big_yaml_path, 'r') as file:
            big_data = yaml.safe_load(file)
        with open(small_yaml_path, 'r') as file:
            small_data = yaml.safe_load(file)
        combined_names = list(big_data['names'] + small_data['names'])
        num_classes = len(combined_names)
        combined_data = {
            'train': os.path.join(output_dataset, 'train/images'),
            'val': os.path.join(output_dataset, 'val/images'),
            'test': os.path.join(output_dataset, 'test/images'),
            'nc': num_classes,
            'names': combined_names
        }
        with open(output_yaml_path, 'w') as file:
            yaml.safe_dump(combined_data, file)

    subdirs = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']
    ensure_subdirs_exist(output_dataset, subdirs)
    for subdir in subdirs:
        src_dir_small = os.path.join(small_dataset, subdir)
        dst_dir_big = os.path.join(big_dataset, subdir)
        dst_dir_combined = os.path.join(output_dataset, subdir)
        if os.path.exists(src_dir_small):
            copy_files(src_dir_small, dst_dir_combined)
        if os.path.exists(dst_dir_big):
            copy_files(dst_dir_big, dst_dir_combined)

    big_yaml = os.path.join(big_dataset, 'dataset.yaml')
    small_yaml = os.path.join(small_dataset, 'dataset.yaml')
    output_yaml = os.path.join(output_dataset, 'dataset.yaml')
    update_yaml(big_yaml, small_yaml, output_yaml)
    print(f"Datasets from {small_dataset} have been combined into {output_dataset}. Updated YAML file created at {output_yaml}")

def train_yolo_v8(data_yaml=os.path.join(os.getcwd(), 'Yolov8_Datasets', 'final_dataset', 'dataset.yaml'),
                model='yolov8n.pt', epochs=100, imgsz=640):
    if not os.path.exists(data_yaml):
        print(f"dataset.yaml file at {data_yaml} does not exist.")
        return
    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)
    base_dir = os.getcwd()
    data['train'] = os.path.join(base_dir, data['train']).replace('\\', '/')
    data['val'] = os.path.join(base_dir, data['val']).replace('\\', '/')
    data['test'] = os.path.join(base_dir, data['test']).replace('\\', '/')
    temp_yaml = os.path.join(data_yaml.replace('dataset.yaml', ''), 'temp_dataset.yaml')
    with open(temp_yaml, 'w') as file:
        yaml.safe_dump(data, file)
    print(f"Using dataset.yaml: {temp_yaml}")
    train_command = [
        "yolo", "detect", "train",
        f"data={temp_yaml}",
        f"model={model}",
        f"epochs={epochs}",
        f"imgsz={imgsz}"
    ]
    print("Running command:", ' '.join(train_command))
    process = subprocess.Popen(train_command)
    return process

def stop_yolo_training(process):
    if process:
        process.terminate()
        process.wait()  # Wait for the process to terminate

def run_tensorboard(logdir="runs"):
    command = ["tensorboard", "--logdir", logdir, "--host", "0.0.0.0", "--port", "6006"]
    process = subprocess.Popen(command)
    return process

if 'process' not in st.session_state:
    st.session_state.process = None
if 'tb_process' not in st.session_state:
    st.session_state.tb_process = None

st.title("Custom Training")

st.write("Upload image folder for custom training")
uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

if uploaded_files:
    upload_dir = "uploaded_files"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Annotate"):
        image_path = r"uploaded_files"
        open_labelme(image_path)

with col2:
    if st.button("Create Dataset"):
        json_folder = r"Yolov8_Datasets/added_new_json"
        move_folder('json_files', 'Yolov8_Datasets/added_new_json')
        convert_to_yolo(json_folder)
        big_dataset = 'Yolov8_Datasets/Welvision_Polarized_Lens-2'
        small_dataset = 'Yolov8_Datasets/added_new_json/YOLOv8Dataset'
        output_dataset = 'Yolov8_Datasets/final_dataset'
        combine_yolo_datasets(big_dataset, small_dataset, output_dataset)
        st.success("Successfully created the dataset")

with col3:
    if st.button("Train Model"):
        st.session_state.process = train_yolo_v8()
        st.write("Model training successfully started")

with col4:
    if st.button("Stop Training") and st.session_state.process:
        stop_yolo_training(st.session_state.process)
        st.session_state.process = None
        st.write("Training stopped")

with col5:
    if st.button("TensorBoard"):
        st.session_state.tb_process = run_tensorboard()
        st.write("TensorBoard started")

if st.session_state.tb_process:
    st.write("### TensorBoard")
    st.markdown(
        """
        <iframe src="http://localhost:6006" width="100%" height="800px" frameborder="0"></iframe>
        """,
        unsafe_allow_html=True
    )
