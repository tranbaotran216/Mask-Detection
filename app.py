import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the model only once
@st.cache_resource
def load_model():
    return YOLO("runs/detect/yolov11-mask-detection/weights/best.pt")

def predict(image, model):
    img_array = np.array(image)
    results = model.predict(img_array)
    return results[0]

def main():
    st.title("Mask Detection using YOLOv11")
    st.write("Upload an image to detect masks.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        model = load_model()
        results = predict(image, model)

        st.write("Detection Results:")
        st.image(results.plot(), caption="Processed Image", use_container_width=True)

if __name__ == "__main__":
    main()
