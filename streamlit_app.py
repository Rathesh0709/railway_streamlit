import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load model (WSL PATH supported)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

st.title("🚆 Railway Track Fault Detection - YOLO11")
st.write("Upload an image or video to detect faults.")

option = st.radio("Choose Input Type", ["Image", "Video"])

# ------------------- IMAGE MODE -------------------
if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(img, caption="Uploaded Image")

        if st.button("Run Detection"):
            results = model.predict(img, conf=0.25)

            annotated = results[0].plot()   # YOLO annotated numpy image
            st.image(annotated, caption="Detection Result")

# ------------------- VIDEO MODE -------------------
else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4","mov","avi"])

    if uploaded_video is not None:
        temp_video = "temp_video.mp4"

        with open(temp_video, "wb") as f:
            f.write(uploaded_video.read())

        st.video(temp_video)

        if st.button("Run Detection"):
            result_gen = model.predict(source=temp_video, stream=True)

            output_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = None

            for r in result_gen:
                frame = r.plot()
                if out is None:
                    h, w, _ = frame.shape
                    out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))
                out.write(frame)

            out.release()
            st.success("Detection Completed!")
            st.video(output_path)
