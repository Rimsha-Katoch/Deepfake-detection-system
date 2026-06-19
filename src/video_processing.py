import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            file_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(file_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Extracted {saved} frames.")