import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import os
import pandas as pd

def load_model(model_name, custom_model_path=None):
    """Load the selected model"""
    try:
        from ultralytics import YOLO
        
        if custom_model_path:
            model = YOLO(custom_model_path)
        else:
            # Load pre-trained YOLOv8 model
            version = model_name.lower().replace("yolov8", "")  # get n,s,m,l,x
            model = YOLO(f"yolov8{version}.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    """Process the image and return predictions"""
    if model is None:
        return None
    results = model(image)
    return results[0]  # YOLOv8 returns a list of Results objects

def main():
    st.title("Wildlife Detection System")
    
    # Add a sidebar for model selection
    st.sidebar.title("Model Settings")
    
    # Model source selection
    model_source = st.sidebar.radio(
        "Model Source",
        ["Pre-trained YOLOv8", "Custom Model"]
    )
    
    model = None
    if model_source == "Pre-trained YOLOv8":
        model_name = st.sidebar.selectbox(
            "Choose YOLOv8 Model Size",
            ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]
        )
        model = load_model(model_name)
    else:
        # Custom model upload
        custom_model = st.sidebar.file_uploader(
            "Upload YOLOv8 Model (.pt file)",
            type=['pt']
        )
        
        if custom_model:
            # Save the uploaded model temporarily
            model_path = f"temp_model_{custom_model.name}"
            with open(model_path, "wb") as f:
                f.write(custom_model.getbuffer())
            model = load_model(None, model_path)
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
    
    if model:
        st.sidebar.success("Model loaded successfully!")

    # Add input source selection
    input_source = st.sidebar.radio(
        "Select Input Source",
        ["Upload Image", "Use Camera"]
    )
    
    if input_source == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add a button to trigger detection
            if st.button("Detect Wildlife"):
                with st.spinner("Detecting..."):
                    # Process the image
                    results = process_image(image, model)
                    if results is not None:
                        # Display results
                        plotted_image = results.plot()  # YOLOv8 uses plot() instead of render()
                        st.image(plotted_image, caption="Detection Result", use_container_width=True)
                        
                        # Display detection information
                        boxes = results.boxes
                        if len(boxes) > 0:
                            st.success(f"Found {len(boxes)} objects!")
                            
                            # Create a list of detections
                            detections = []
                            for box in boxes:
                                # Get class name and confidence
                                class_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = results.names[class_id]
                                detections.append({
                                    'name': class_name,
                                    'confidence': conf
                                })
                            
                            # Convert to DataFrame and display
                            df = pd.DataFrame(detections)
                            st.write("Detection Details:")
                            st.dataframe(df)
                        else:
                            st.warning("No wildlife detected in the image.")
    
    else:  # Use Camera
        # Initialize camera
        camera = cv2.VideoCapture(0)
        
        # Create placeholders
        camera_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Add stop button
        stop_button = st.button("Stop Camera")
        
        while not stop_button:
            # Read frame from camera
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access camera!")
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = process_image(frame_rgb, model)
            if results is not None:
                # Get processed frame with detections
                plotted_frame = results.plot()
                
                # Display the frame
                camera_placeholder.image(plotted_frame, channels="RGB", caption="Live Detection", use_container_width=True)
                
                # Update detection information
                boxes = results.boxes
                if len(boxes) > 0:
                    detections = []
                    for box in boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = results.names[class_id]
                        detections.append({
                            'name': class_name,
                            'confidence': conf
                        })
                    
                    df = pd.DataFrame(detections)
                    info_placeholder.dataframe(df)
                else:
                    info_placeholder.empty()
        
        # Release camera when stopped
        camera.release()

if __name__ == "__main__":
    main()
