import cv2
import torch
from ultralytics import YOLO

# Loading the trained model
model = YOLO('monkeyUpdated.pt')

# Define class names based on your training data
class_names = ["Lion", "Monkey"]  # Add more classes as needed

# Function to make predictions and display image
def predict_and_display(img_path, confidence_threshold=0.1):

    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not read image {img_path}")
        return

    results = model.predict(image, conf=confidence_threshold)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            label = f"{class_names[class_id]}: {confidence:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = './testImages/monkey4.png'
confidence_threshold = 0.4
predict_and_display(img_path, confidence_threshold)

