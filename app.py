import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
from pytube import YouTube
import numpy as np

# Load model YOLO
MODEL_PATH = "best.pt"  # Sesuaikan path model
model = YOLO(MODEL_PATH)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Simpan video hasil deteksi sementara
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_video.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    class_colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{['bahrain', 'bola', 'indonesia'][cls]} ({conf:.2f})"
                text_color = class_colors.get(cls, (255, 255, 255))
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        out.write(frame)

    cap.release()
    out.release()
    return output_path


def download_youtube_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension="mp4").first()
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    stream.download(filename=temp_video.name)
    return temp_video.name


st.title("YOLO Video Detection")
option = st.radio("Pilih sumber video:", ("Upload File", "YouTube Link"))

video_path = None
if option == "Upload File":
    uploaded_file = st.file_uploader("Unggah video MP4", type=["mp4"])
    if uploaded_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name
elif option == "YouTube Link":
    youtube_url = st.text_input("Masukkan link YouTube")
    if youtube_url:
        with st.spinner("Mengunduh video..."):
            video_path = download_youtube_video(youtube_url)
        st.success("Video berhasil diunduh!")

if video_path:
    with st.spinner("Memproses video..."):
        processed_video_path = process_video(video_path)
    st.video(processed_video_path)

    # Remove the temporary files after the video is processed
    # Remove the processed video after displaying
    os.remove(processed_video_path)
    # Remove the uploaded/downloaded video after displaying
    os.remove(video_path)
