import os
import cv2
from tqdm import tqdm

# Adjust the data_dir and output_dir paths to be one level above the current script location
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../labelled_data_yolo/yolo_labels')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../yolo_formatted_data')
class_id = 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a list of student folders
student_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

for student_folder in tqdm(student_folders, desc="Converting Data"):
    # Create a single output file for the student
    student_output_file = os.path.join(output_dir, f'{student_folder}.txt')
    
    for image_file in os.listdir(os.path.join(data_dir, student_folder)):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(data_dir, student_folder, image_file)
            img = cv2.imread(image_path)
            height, width, _ = img.shape
            
            # Modify these values based on your bounding box format
            x_center = width / 2
            y_center = height / 2
            bbox_width = width
            bbox_height = height
            
            # Append YOLO format to the student's output file
            with open(student_output_file, 'a') as output_file:
                output_file.write(f"{class_id} {x_center/width} {y_center/height} {bbox_width/width} {bbox_height/height}\n")
