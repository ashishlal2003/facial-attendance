import os
from ultralytics import YOLO
import time
from tqdm import tqdm
import cv2

# Path to the YOLO model file
model = YOLO("yolov8n-face.pt")

# Input and output directories
input_directory = "train"
output_directory = "yolo_labels"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Get a list of subfolders in the input directory
subfolders = [subfolder for subfolder in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, subfolder))]

# Iterate through subfolders and process images
for subfolder in subfolders:
    subfolder_path = os.path.join(input_directory, subfolder)
    
    # Create a corresponding subfolder in the output directory
    output_subfolder = os.path.join(output_directory, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)

    # Get a list of image files in the subfolder
    image_files = [image_file for image_file in os.listdir(subfolder_path) if image_file.endswith((".jpg", ".png", ".jpeg"))]

    # Create a tqdm progress bar
    with tqdm(total=len(image_files), desc=f"Processing {subfolder}") as pbar:
        # Iterate through images in the subfolder
        for image_file in image_files:
            image_path = os.path.join(subfolder_path, image_file)
            
            # Predict using the YOLO model
            results = model.predict(source=image_path)

            # Access prediction results for the first image
            for det in results[0]:
                x1, y1, x2, y2, conf, class_id = det.tolist()

                # Do something with the prediction data
                print(f"Bounding Box: ({x1}, {y1}) - ({x2}, {y2}), Confidence: {conf}, Class ID: {class_id}")
            
            # Load the original image
            original_image = cv2.imread(image_path)

            # Draw bounding boxes on the original image
            for det in results[0]:
                x1, y1, x2, y2, conf, class_id = det.tolist()
                color = (0, 255, 0)  # Green color for bounding boxes
                thickness = 2
                cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # Save the labeled image in the output directory with the same structure
            labeled_image_path = os.path.join(output_subfolder, image_file)
            cv2.imwrite(labeled_image_path, original_image)

            pbar.update(1)  # Update the progress bar
            pbar.set_postfix(Processed=image_file)
    
# Wait for 10 seconds before exiting
time.sleep(10)
