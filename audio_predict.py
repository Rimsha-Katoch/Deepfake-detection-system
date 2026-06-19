import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from moviepy import VideoFileClip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
)

model.load_state_dict(torch.load("models/audio_model.pth", map_location=device))
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def extract_audio(video_path):
    audio_path = "temp_audio.wav"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    return audio_path


def create_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(S_db, sr=sr)
    plt.axis('off')
    plt.savefig("temp_spec.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    return "temp_spec.png"


def predict_audio_from_video(video_path):
    audio_path = extract_audio(video_path)
    spec_path = create_spectrogram(audio_path)

    image = Image.open(spec_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, 1)

        label = "Real" if pred.item() == 1 else "Fake"

        return label, confidence.item()