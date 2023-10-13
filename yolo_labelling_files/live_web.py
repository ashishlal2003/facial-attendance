from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n-face.pt")

# Initialize OpenCV's VideoCapture for webcam feed
cap = cv2.VideoCapture(0)  # Use the default camera (usually 0) or specify the camera's index if you have multiple cameras

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the OpenCV frame to an image (numpy array)
    image = np.array(frame)

    # Perform face detection on the image using the YOLO model
    results = model(image)

    # Show the results with bounding boxes
    results.show()

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
