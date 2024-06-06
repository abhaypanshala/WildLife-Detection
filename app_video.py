import cv2
import torch
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('lion.pt')

# Function to make predictions and display video
def predict_and_display_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make predictions on the current frame
        results = model.predict(frame)

        # Process results and draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates and confidence score
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                # Draw the bounding box and label
                label = f"Lion: {confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame in a popup window
        cv2.imshow('Prediction', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

# Path to the video you want to predict
video_path = '/home/technocrat/Pictures/lion.mp4'

# Run prediction and display
predict_and_display_video(video_path)

