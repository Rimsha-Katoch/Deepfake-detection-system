import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from audio_predict import predict_audio_from_video

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(128, 2)
)

model = model.to(device)
model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- FRAME PREDICTION ----------------
def predict_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # fallback if no face
    if len(faces) == 0:
        face = frame
    else:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)

    face = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face)
        probs = torch.softmax(outputs, dim=1)

    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    return real_prob, fake_prob


# ---------------- VIDEO PREDICTION ----------------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    real_scores = []
    fake_scores = []
    frame_count = 0
    valid_frames = 0   # ✅ FIXED

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # sample every 5th frame
        if frame_count % 5 != 0:
            continue

        real_prob, fake_prob = predict_frame(frame)

        real_scores.append(real_prob)
        fake_scores.append(fake_prob)
        valid_frames += 1
    cap.release()

    # -------- SAFETY --------
    if valid_frames == 0:
        return {
            "label": "Not Sure",
            "confidence": 0.0,
            "real_prob": 0.0,
            "fake_prob": 0.0
        }

    # -------- AVERAGE --------
    video_real = float(np.mean(real_scores))
    video_fake = float(np.mean(fake_scores))

    # ---------------- AUDIO ----------------
    try:
        audio_label, audio_conf = predict_audio_from_video(video_path)

        if audio_label == "Real":
            audio_real = audio_conf
            audio_fake = 1 - audio_conf
        else:
            audio_fake = audio_conf
            audio_real = 1 - audio_conf

    except Exception as e:
        print("Audio Error:", e)
        audio_real = 0.5
        audio_fake = 0.5

    # ---------------- FUSION ----------------
    final_real = 0.7 * video_real + 0.3 * audio_real
    final_fake = 0.7 * video_fake + 0.3 * audio_fake

    if final_real > final_fake:
        label = "Real"
        confidence = final_real
    else:
        label = "Fake"
        confidence = final_fake

    # ---------------- DEBUG LOGS ----------------
    print("\n===== VIDEO ANALYSIS =====")
    print(f"Video Path   : {video_path}")
    print(f"Label        : {label}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print(f"Real Prob    : {final_real * 100:.2f}%")
    print(f"Fake Prob    : {final_fake * 100:.2f}%")
    print(f"Frames Used  : {valid_frames}")
    print("===========================\n")

    return {
        "label": label,
        "confidence": float(confidence),   # ✅ IMPORTANT: 0–1 ONLY
        "real_prob": float(final_real),
        "fake_prob": float(final_fake)
    }


# ---------------- TEST ----------------
if __name__ == "__main__":
    result = predict_video("test_video.mp4")
    print(result)