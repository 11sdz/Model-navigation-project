import cv2
from ultralytics import YOLO
import torch
from tqdm import tqdm

# Load the model
model = YOLO('runs/kfold/fold4_0/weights/best.pt')  # Adjust your path

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open video file
cap = cv2.VideoCapture('DJI0274.mp4')
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('predicted_DJI0274v2.mp4', fourcc, fps, (width, height))

# Progress bar
with tqdm(total=total_frames, desc="Processing Video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction on the frame
        results = model.predict(frame, device=device, verbose=False)

        # Get the annotated frame
        annotated_frame = results[0].plot()

        # Write to output video
        out.write(annotated_frame)

        pbar.update(1)  # update progress bar by one frame

# Clean up
cap.release()
out.release()
print("Prediction complete. Saved to 'predicted_DJI0149.mp4'")